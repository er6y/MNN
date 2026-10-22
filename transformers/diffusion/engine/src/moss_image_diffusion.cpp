//  moss_image_diffusion.cpp - MOSS-Image (OpenMOSS 5B DiT + 8-step DMD turbo) engine.
//  See moss_image_diffusion.hpp for the graph contract. Reference chains:
//  GPU-server work/moss_export_dit.py (export), moss_pt_ref.py (PT ground truth),
//  moss_verify.cpp (MNN parity - FINAL cosine 0.874 @512 / 0.880 @768 dyn).
#include <random>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include "diffusion/moss_image_diffusion.hpp"
#include "diffusion/diffusion_config.hpp"
#include <rapidjson/document.h>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>
#include <MNN/expr/ExecutorScope.hpp>
using namespace CV;

namespace MNN {
namespace DIFFUSION {

namespace {

// MOSS rope, fp64 host build. Per-token 128 floats:
// [sin_r16, sin_f16, sin_h16, sin_w16, cos_r16, cos_f16, cos_h16, cos_w16];
// 16 bands per axis, base 10000 (matches build_rope in moss_export_dit.py).
static const int kRopeBands = 16;

std::vector<float> buildMossRope(const std::vector<float>& coords4, int rows) {
    double bands[kRopeBands];
    for (int j = 0; j < kRopeBands; ++j)
        bands[j] = 1.0 / std::pow(10000.0, (double)j / (double)kRopeBands);
    std::vector<float> rope((size_t)rows * 128, 0.f);
    for (int r = 0; r < rows; ++r) {
        float* out = &rope[(size_t)r * 128];
        for (int a = 0; a < 4; ++a) {
            double pos = coords4[(size_t)r * 4 + a];
            for (int j = 0; j < kRopeBands; ++j) {
                double ang = pos * bands[j];
                out[a * kRopeBands + j] = (float)std::sin(ang);
                out[64 + a * kRopeBands + j] = (float)std::cos(ang);
            }
        }
    }
    return rope;
}

} // namespace

// ===== Constructor =====
MossImageDiffusion::MossImageDiffusion(
    std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode,
    int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU,
    DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode,
    DiffusionCFGMode cfgMode, int numThreads)
    : Diffusion(modelPath, modelType, backendType, memoryMode,
                imageWidth, imageHeight, textEncoderOnCPU, vaeOnCPU,
                gpuMemoryMode, precisionMode, cfgMode, numThreads) {

    loadSchedulerConfig();
    // scheduler_config.json: MOSSImageScheduler(shift=5, training_shift=1)
    for (const auto& schedPath : {mModelPath + "/scheduler/scheduler_config.json",
                                   mModelPath + "/scheduler_config.json"}) {
        std::ifstream f(schedPath.c_str());
        if (!f.good()) continue;
        std::ostringstream oss; oss << f.rdbuf();
        rapidjson::Document doc; doc.Parse(oss.str().c_str());
        if (doc.HasParseError() || !doc.IsObject()) continue;
        if (doc.HasMember("shift") && (doc["shift"].IsFloat() || doc["shift"].IsDouble()))
            mSchedulerShift = doc["shift"].GetDouble();
        if (doc.HasMember("training_shift") && (doc["training_shift"].IsFloat() || doc["training_shift"].IsDouble()))
            mTrainingShift = doc["training_shift"].GetDouble();
        if (doc.HasMember("num_train_timesteps") && doc["num_train_timesteps"].IsInt())
            mTrainTimesteps = doc["num_train_timesteps"].GetInt();
        break;
    }

    // config.json: vae.bn_mean/bn_std (128ch), text_encoder dir/tokenizer,
    // text_max_len, inference.width/height (default render size)
    {
        std::ifstream f((mModelPath + "/config.json").c_str());
        if (f.good()) {
            std::ostringstream oss; oss << f.rdbuf();
            rapidjson::Document doc; doc.Parse(oss.str().c_str());
            if (!doc.HasParseError() && doc.IsObject()) {
                if (doc.HasMember("vae") && doc["vae"].IsObject()) {
                    auto& vae = doc["vae"];
                    if (vae.HasMember("bn_mean") && vae["bn_mean"].IsArray())
                        for (auto& v : vae["bn_mean"].GetArray()) mVaeBnMean.push_back((float)v.GetDouble());
                    if (vae.HasMember("bn_std") && vae["bn_std"].IsArray())
                        for (auto& v : vae["bn_std"].GetArray()) mVaeBnStd.push_back((float)v.GetDouble());
                }
                if (doc.HasMember("text_max_len") && doc["text_max_len"].IsInt())
                    mMaxTextLen = doc["text_max_len"].GetInt();
                if (doc.HasMember("text_encoder") && doc["text_encoder"].IsObject()) {
                    auto& te = doc["text_encoder"];
                    std::string dir, tok;
                    if (te.HasMember("directory") && te["directory"].IsString()) dir = te["directory"].GetString();
                    if (te.HasMember("tokenizer") && te["tokenizer"].IsString()) tok = te["tokenizer"].GetString();
                    mTeDir = mModelPath + "/" + (dir.empty() ? "" : dir + "/");
                    mTokenizerPath = mTeDir + (tok.empty() ? "tokenizer.txt" : tok);
                }
                if (doc.HasMember("inference") && doc["inference"].IsObject()) {
                    auto& inf = doc["inference"];
                    if (inf.HasMember("width") && inf["width"].IsInt()) mConfigImageWidth = inf["width"].GetInt();
                    if (inf.HasMember("height") && inf["height"].IsInt()) mConfigImageHeight = inf["height"].GetInt();
                }
            }
        }
        if (!mVaeBnMean.empty())
            MNN_PRINT("[MossImage] Loaded VAE BN params: %d channels\n", (int)mVaeBnMean.size());
        else
            MNN_PRINT("[MossImage] Warning: VAE BN params not found in config.json\n");
    }
    // TE hidden size from the llmexport llm_config.json
    {
        std::ifstream f((mTeDir + "llm_config.json").c_str());
        if (f.good()) {
            std::ostringstream oss; oss << f.rdbuf();
            rapidjson::Document doc; doc.Parse(oss.str().c_str());
            if (!doc.HasParseError() && doc.IsObject() && doc.HasMember("hidden_size") && doc["hidden_size"].IsInt())
                mHidden = doc["hidden_size"].GetInt();
        }
        mEmbeddingsFile = mTeDir + "embeddings_bf16.bin";
    }

    // Dynamic DiT: any 16-aligned size is valid; config inference.* is the default.
    if (imageWidth > 0 && imageHeight > 0) {
        mImageWidth  = imageWidth;
        mImageHeight = imageHeight;
    } else if (mConfigImageWidth > 0 && mConfigImageHeight > 0) {
        mImageWidth  = mConfigImageWidth;
        mImageHeight = mConfigImageHeight;
    } else {
        mImageWidth = 512; mImageHeight = 512;
    }
    mImageWidth  = ((mImageWidth  + 15) / 16) * 16;
    mImageHeight = ((mImageHeight + 15) / 16) * 16;
    if (mImageWidth  < 256) mImageWidth  = 256;
    if (mImageHeight < 256) mImageHeight = 256;
    mLatentH = mImageHeight / 8;
    mLatentW = mImageWidth  / 8;
    MNN_PRINT("[MossImage] image=%dx%d latent=%dx%d img_seq=%d shift=%.2f\n",
              mImageWidth, mImageHeight, mLatentH, mLatentW,
              (mLatentH / 2) * (mLatentW / 2), mSchedulerShift);
}

MossImageDiffusion::~MossImageDiffusion() {
}

// ===== Scheduler: MOSSImageScheduler (shift grid + nearest training-grid snap) =====
std::vector<float> MossImageDiffusion::getMossSigmas(int numSteps) const {
    const int total = mTrainTimesteps;
    auto grid = [&](int count, double sh) {
        std::vector<double> v(count);
        double lower = sh / ((double)total + sh - 1.0);
        for (int i = 0; i < count; ++i) {
            double t = 1.0 + (lower - 1.0) * (double)i / (double)(count - 1); // linspace(1, lower, count)
            v[i] = sh * t / (1.0 + (sh - 1.0) * t);
        }
        return v; // descending 1 -> ~0
    };
    std::vector<double> sig = grid(numSteps, mSchedulerShift);
    // training grid (training_shift=1), flipped to ascending
    std::vector<double> training = grid(total, mTrainingShift);
    std::reverse(training.begin(), training.end());
    // nearest-grid snap (torch.searchsorted default: first idx with training[idx] >= s)
    std::vector<double> snapped;
    double last = -1.0;
    for (double s : sig) {
        int lo = (int)(std::lower_bound(training.begin(), training.end(), s) - training.begin());
        int upper = std::min(lo, total - 1);
        int lowerI = std::max(upper - 1, 0);
        int nearest = (std::abs(s - training[lowerI]) <= std::abs(training[upper] - s)) ? lowerI : upper;
        double snappedV = training[nearest];
        // unique_consecutive: drop dups in keep-order
        if (snapped.empty() || snappedV != last) {
            snapped.push_back(snappedV);
            last = snappedV;
        }
    }
    std::vector<float> out(snapped.size() + 1);
    for (size_t i = 0; i < snapped.size(); ++i) out[i] = (float)snapped[i];
    out[snapped.size()] = 0.0f; // terminal
    return out;
}

// ===== load() =====
bool MossImageDiffusion::load() {
    AUTOTIME;
    if (!initRuntimeManagers(/*gpuBufferMode=*/true, /*attentionHint=*/8)) return false;

    mDiffConfig.reset(new DiffusionConfig(mModelPath));
    mModules.resize(3);

    if (!ensureTeLoaded()) return false;
    if (mMemoryMode != 0) {
        if (!ensureDiTLoaded()) return false;
        if (!ensureVAELoaded()) return false;
        MNN_PRINT("[MossImage] modules loaded (TE / DiT / VAE)\n");
    } else {
        MNN_PRINT("[MossImage] memory_mode=0: staged load (TE now, DiT/VAE on demand)\n");
    }
    return true;
}

bool MossImageDiffusion::ensureTeLoaded() {
    if (mModules.size() < 3) mModules.resize(3);
    if (mModules[0]) return true;
    auto path = mDiffConfig->text_encoder_model();
    MNN_PRINT("[MossImage] Load text encoder: %s\n", path.c_str());
    Module::Config tec; tec.shapeMutable = true;
    auto& te_runtime = runtime_manager_cpu_ ? runtime_manager_cpu_ : runtime_manager_;
    // llmexport contract (qwenimage21 style): float embeds go into "input_ids",
    // causal mask / positions / logits_index are explicit inputs.
    mModules[0].reset(Module::load({"input_ids", "attention_mask", "position_ids", "logits_index"},
                                   {"logits", "hidden_states"}, path.c_str(), te_runtime, &tec));
    if (!mModules[0]) { MNN_ERROR("[MossImage] Failed to load text encoder\n"); return false; }
    return true;
}

bool MossImageDiffusion::ensureDiTLoaded() {
    if (mModules.size() < 3) mModules.resize(3);
    if (mModules[1]) return true;
    Module::Config mc; mc.shapeMutable = true;
    auto path = mDiffConfig->unet_model();
    MNN_PRINT("[MossImage] Load transformer: %s\n", path.c_str());
    mModules[1].reset(Module::load(
        {"hidden_states", "timestep", "encoder_hidden_states", "rope_img", "rope_txt"},
        {"out_sample"}, path.c_str(), runtime_manager_, &mc));
    if (!mModules[1]) { MNN_ERROR("[MossImage] Failed to load transformer\n"); return false; }
    return true;
}

bool MossImageDiffusion::ensureVAELoaded() {
    if (mModules.size() < 3) mModules.resize(3);
    if (mModules[2]) return true;
    Module::Config mc; mc.shapeMutable = true;
    auto path = mDiffConfig->vae_decoder_model();
    MNN_PRINT("[MossImage] Load VAE decoder: %s\n", path.c_str());
    auto& vae_runtime = runtime_manager_vae_cpu_ ? runtime_manager_vae_cpu_ : runtime_manager_;
    mModules[2].reset(Module::load({"latent_sample"}, {"sample"}, path.c_str(), vae_runtime, &mc));
    if (!mModules[2]) { MNN_ERROR("[MossImage] Failed to load VAE decoder\n"); return false; }
    mModules[2]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
    return true;
}

// ===== Text Encoder =====
// Chat template matches AutoProcessor.apply_chat_template([{"role":"user",
//   "content":prompt}], tokenize=False, add_generation_prompt=True) on the
//   MOSS-Image-SFT tokenizer (enable_thinking): note the trailing "<think>\n".
// GDN is padding-sensitive: EXACT token length, no window, no pad.
bool MossImageDiffusion::textEncode(const std::string& prompt, std::vector<float>& embeds, int& seqLen) {
    AUTOTIME;
    if (!ensureTeLoaded()) return false;
    std::vector<int> inputIds;
#ifdef MNN_BUILD_LLM
    if (!mTokenizer) {
        mTokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(mTokenizerPath));
        if (!mTokenizer) MNN_PRINT("[MossImage] Warning: tokenizer load failed: %s\n", mTokenizerPath.c_str());
    }
    if (mTokenizer) {
        std::string templated = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
        inputIds = mTokenizer->encode(templated);
        MNN_PRINT("[MossImage] Tokens after chat_template: %d\n", (int)inputIds.size());
    }
#endif
    if (inputIds.empty()) {
        MNN_ERROR("[MossImage] tokenize failed\n");
        return false;
    }
    if ((int)inputIds.size() > mMaxTextLen) inputIds.resize(mMaxTextLen);
    const int seq = (int)inputIds.size();

    // bf16 embedding table lookup -> float embeds [seq, hidden].
    // Row-wise seek+read: the full table is ~1.2GB and loading it whole spikes RSS
    // right before the TE forward, which gets the process LMK-killed on phones.
    std::ifstream f(mEmbeddingsFile, std::ios::binary);
    if (!f.good()) { MNN_ERROR("[MossImage] embeddings file missing: %s\n", mEmbeddingsFile.c_str()); return false; }
    f.seekg(0, std::ios::end);
    const size_t vocab = (size_t)f.tellg() / 2 / (size_t)mHidden;
    std::vector<float> embs((size_t)seq * mHidden);
    std::vector<uint16_t> row((size_t)mHidden);
    for (int i = 0; i < seq; ++i) {
        int id = inputIds[i];
        if (id < 0 || (size_t)id >= vocab) { MNN_ERROR("[MossImage] bad token id %d\n", id); return false; }
        f.seekg((std::streamoff)((size_t)id * mHidden * 2));
        f.read((char*)row.data(), (std::streamsize)mHidden * 2);
        for (int j = 0; j < mHidden; ++j) {
            unsigned u = ((unsigned)row[j]) << 16;
            float fval;
            memcpy(&fval, &u, 4);
            embs[(size_t)i * mHidden + j] = fval;
        }
    }

    VARP eVar = _Input({seq, 1, mHidden}, NCHW, halide_type_of<float>());
    memcpy(eVar->writeMap<float>(), embs.data(), sizeof(float) * embs.size());
    // causal mask, built host-side (llmexport graph has no in-graph causal)
    VARP mask = _Input({1, 1, seq, seq}, NCHW, halide_type_of<float>());
    {
        auto mp = mask->writeMap<float>();
        const float NEG = -3.4e38f;
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < seq; ++j) mp[(size_t)i * seq + j] = (j > i) ? NEG : 0.0f;
    }
    VARP pos = _Input({3, seq}, NCHW, halide_type_of<int>());
    {
        auto p = pos->writeMap<int>();
        for (int a = 0; a < 3; ++a)
            for (int i = 0; i < seq; ++i) p[(size_t)a * seq + i] = i;
    }
    VARP lix = _Input({1}, NCHW, halide_type_of<int>());
    lix->writeMap<int>()[0] = seq - 1;

    auto teRes = mModules[0]->onForward({eVar, mask, pos, lix});
    if (teRes.size() < 2 || teRes[1].get() == nullptr) {
        MNN_ERROR("[MossImage] TE forward failed\n");
        return false;
    }
    // read raw (qwenimage21 style - no layout convert on the TE output)
    auto rawInfo = teRes[1]->getInfo();
    size_t total = 1;
    std::string dimStr;
    for (auto d : rawInfo->dim) { total *= d; dimStr += std::to_string(d) + ","; }
    const int outHid = rawInfo->dim[rawInfo->dim.size() - 1];
    if (outHid != mHidden) {
        MNN_ERROR("[MossImage] TE output hidden mismatch: dims=[%s] expect last=%d\n",
                  dimStr.c_str(), mHidden);
        return false;
    }
    const int outSeq = (int)(total / (size_t)outHid);
    auto hsPtr = teRes[1]->readMap<float>();
    embeds.assign(hsPtr, hsPtr + total);
    seqLen = outSeq;
    MNN_PRINT("[MossImage] Text embeds: dims=[%s] -> seq=%d hid=%d\n",
              dimStr.c_str(), outSeq, outHid);
    // GDN TE is huge (~8GB int8): release before the DiT loads (staged mode)
    if (mMemoryMode == 0) {
        mModules[0].reset();
#ifdef MNN_BUILD_LLM
        mTokenizer.reset();
#endif
        MNN_PRINT("[MossImage] Text encoder unloaded (low memory mode)\n");
    }
    return true;
}

// ===== VAE Decoder (BN denorm + unpatchify fused, same as FLUX.3) =====
VARP MossImageDiffusion::vae_decoder(VARP latent) {
    if (mMemoryMode != 1) {
        mModules[1].reset();
        MNN_PRINT("[MossImage] Transformer unloaded\n");
    }
    if (!mModules[2] && !ensureVAELoaded()) return nullptr;
    AUTOTIME;
    auto info = latent->getInfo();
    int B = info->dim[0], C = info->dim[1], H = info->dim[2], W = info->dim[3];
    int LC = C / 4, LH = H * 2, LW = W * 2;
    auto denormVar = _Input({B, LC, LH, LW}, NCHW, halide_type_of<float>());
    {
        const float* src = latent->readMap<float>();
        float* dst = denormVar->writeMap<float>();
        size_t plane = (size_t)H * W;
        bool hasBn = (int)mVaeBnMean.size() == C && (int)mVaeBnStd.size() == C;
        if (!hasBn) MNN_PRINT("[MossImage] Warning: BN params missing on decode, skipping denorm\n");
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < LC; ++c)
                for (int dh = 0; dh < 2; ++dh)
                    for (int dw = 0; dw < 2; ++dw) {
                        int fc = c * 4 + dh * 2 + dw;
                        float mean = hasBn ? mVaeBnMean[fc] : 0.f;
                        float stdv = hasBn ? mVaeBnStd[fc] : 1.f;
                        const float* srcPlane = src + (size_t)b * C * plane + fc * plane;
                        for (int ph = 0; ph < H; ++ph)
                            for (int pw = 0; pw < W; ++pw)
                                dst[((size_t)b * LC + c) * LH * LW + (size_t)(ph * 2 + dh) * LW + (pw * 2 + dw)] =
                                    srcPlane[(size_t)ph * W + pw] * stdv + mean;
                    }
        denormVar.fix(VARP::CONSTANT);
    }
    auto outs = mModules[2]->onForward({denormVar});
    if (outs.empty()) { MNN_PRINT("[MossImage] VAE decode failed\n"); return nullptr; }
    return nchwFloatToHwcBGR(_Convert(outs[0], NCHW));
}

// ===== Denoising loop =====
VARP MossImageDiffusion::denoiseLoop(const std::vector<float>& textEmbeds, int txtSeqLen,
                                     int iterNum, int randomSeed,
                                     std::function<void(int)> progressCallback) {
    if (mMemoryMode == 0 && !ensureDiTLoaded()) return nullptr;
    AUTOTIME;
    int pH = mLatentH / 2, pW = mLatentW / 2;
    int imgSeq = pH * pW;
    const int pC = mPackedC;

    // noise in packed space [1,128,pH,pW] (Philox, PyTorch-aligned), then seq layout
    int noiseSize = pC * pH * pW;
    std::vector<float> noiseData(noiseSize);
    {
        int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
        generateLatentNoise(noiseData.data(), noiseSize, seed);
    }
    std::vector<float> curLatent((size_t)imgSeq * pC);
    for (int c = 0; c < pC; ++c)
        for (int s = 0; s < imgSeq; ++s)
            curLatent[(size_t)s * pC + c] = noiseData[(size_t)c * imgSeq + s];

    auto sigmas = getMossSigmas(iterNum);
    int steps = (int)sigmas.size() - 1;

    // rope: img coords (0, txtLen, h-(pH-1)/2, w-(pW-1)/2); txt coords (0,pos,0,0)
    std::vector<float> imgCoords((size_t)imgSeq * 4, 0.f);
    for (int h = 0; h < pH; ++h)
        for (int w = 0; w < pW; ++w) {
            size_t idx = (size_t)(h * pW + w) * 4;
            imgCoords[idx + 0] = 0.f;
            imgCoords[idx + 1] = (float)txtSeqLen;
            imgCoords[idx + 2] = (float)(h - (pH - 1) / 2.0);
            imgCoords[idx + 3] = (float)(w - (pW - 1) / 2.0);
        }
    std::vector<float> txtCoords((size_t)txtSeqLen * 4, 0.f);
    for (int i = 0; i < txtSeqLen; ++i)
        txtCoords[(size_t)i * 4 + 1] = (float)i;
    std::vector<float> imgRope = buildMossRope(imgCoords, imgSeq);
    std::vector<float> txtRope = buildMossRope(txtCoords, txtSeqLen);

    auto ropeImgVar = _Input({1, imgSeq, 128}, NCHW, halide_type_of<float>());
    auto ropeTxtVar = _Input({1, txtSeqLen, 128}, NCHW, halide_type_of<float>());
    memcpy(ropeImgVar->writeMap<float>(), imgRope.data(), imgRope.size() * sizeof(float));
    memcpy(ropeTxtVar->writeMap<float>(), txtRope.data(), txtRope.size() * sizeof(float));
    ropeImgVar.fix(VARP::CONSTANT);
    ropeTxtVar.fix(VARP::CONSTANT);
    auto embedsVar = _Input({1, txtSeqLen, mHidden}, NCHW, halide_type_of<float>());
    memcpy(embedsVar->writeMap<float>(), textEmbeds.data(), textEmbeds.size() * sizeof(float));
    embedsVar.fix(VARP::CONSTANT);

    MNN_PRINT("[MossImage] Denoising %d steps, sigmas[0]=%.4f sigmas[last]=%.4f\n",
              steps, sigmas[0], sigmas[steps - 1]);

    for (int i = 0; i < steps; ++i) {
        AUTOTIME;
        float dt = sigmas[i + 1] - sigmas[i];

        // seq -> 4D packed [1,128,pH,pW]
        auto hsVar = _Input({1, pC, pH, pW}, NCHW, halide_type_of<float>());
        {
            float* dst = hsVar->writeMap<float>();
            for (int c = 0; c < pC; ++c)
                for (int s = 0; s < imgSeq; ++s)
                    dst[(size_t)c * imgSeq + s] = curLatent[(size_t)s * pC + c];
        }
        auto timestepVar = _Input({1}, NCHW, halide_type_of<float>());
        timestepVar->writeMap<float>()[0] = sigmas[i] * 1000.0f; // raw 0..1000

        auto outs = mModules[1]->onForward({hsVar, timestepVar, embedsVar, ropeImgVar, ropeTxtVar});
        if (outs.empty()) {
            MNN_PRINT("[MossImage] Transformer failed at step %d\n", i);
            return nullptr;
        }
        auto noisePred = _Convert(outs[0], NCHW);
        noisePred.fix(VARP::CONSTANT);
        const float* npPtr = noisePred->readMap<float>();
        // out_sample is the FULL seq [1, imgSeq+txtSeq, 128]; per-row ops only,
        // the first imgSeq rows are the image velocity.
        for (size_t j = 0; j < curLatent.size(); ++j) curLatent[j] += dt * npPtr[j];

        if (mBackendType == MNN_FORWARD_OPENCL)
            MNN::Express::ExecutorScope::Current()->gc(MNN::Express::Executor::PART);
        if (progressCallback) progressCallback((2 + i) * 100 / (steps + 3));
        MNN_PRINT("[MossImage] Step %d/%d t=%.1f dt=%.4f\n", i + 1, steps,
                  sigmas[i] * 1000.0f, dt);
    }

    // seq -> 4D packed for the VAE (BN denorm + unpatchify happen in vae_decoder)
    auto latentVar = _Input({1, pC, pH, pW}, NCHW, halide_type_of<float>());
    {
        float* dst = latentVar->writeMap<float>();
        for (int c = 0; c < pC; ++c)
            for (int s = 0; s < imgSeq; ++s)
                dst[(size_t)c * imgSeq + s] = curLatent[(size_t)s * pC + c];
    }
    latentVar.fix(VARP::CONSTANT);
    return latentVar;
}

// ===== run() overloads =====
bool MossImageDiffusion::run(const std::string prompt, const std::string imagePath,
                             int iterNum, int randomSeed,
                             std::function<void(int)> progressCallback) {
    return run(prompt, imagePath, iterNum, randomSeed, 1.0f, progressCallback, "");
}

bool MossImageDiffusion::run(const std::string prompt, const std::string outputPath,
                             int iterNum, int randomSeed, float cfgScale,
                             std::function<void(int)> progressCallback,
                             const std::string inputImagePath) {
    AUTOTIME;
    if (!inputImagePath.empty()) {
        MNN_PRINT("[MossImage] Error: MOSS-Image does not support image editing\n");
        return false;
    }
    // DMD turbo recipe: 8 steps, no CFG. cfgScale is ignored (single forward).
    if (iterNum < 1)  iterNum = 8;
    if (iterNum > 50) iterNum = 50;

    if (progressCallback) progressCallback(0);
    std::vector<float> textEmbeds;
    int txtSeqLen = 0;
    if (!textEncode(prompt, textEmbeds, txtSeqLen)) {
        MNN_PRINT("[MossImage] Text encode failed\n");
        return false;
    }
    if (progressCallback) progressCallback(1 * 100 / (iterNum + 3));

    auto latent = denoiseLoop(textEmbeds, txtSeqLen, iterNum, randomSeed, progressCallback);
    if (!latent.get()) { MNN_PRINT("[MossImage] Denoise failed\n"); return false; }

    auto image = vae_decoder(latent);
    if (!image.get()) { MNN_PRINT("[MossImage] VAE decode failed\n"); return false; }

    bool res = CV::imwrite(outputPath, image);
    if (res) MNN_PRINT("[MossImage] Saved: %s\n", outputPath.c_str());
    else     MNN_PRINT("[MossImage] Error: imwrite failed: %s\n", outputPath.c_str());

    if (mMemoryMode != 1) {
        mModules[2].reset();
        MNN_PRINT("[MossImage] VAE decoder unloaded\n");
    }
    if (progressCallback) progressCallback(100);
    return res;
}

bool MossImageDiffusion::run(const VARP input_embeds, const std::string& mode,
                             const std::string& inputImagePath, const std::string& outputImagePath,
                             int width, int height, int iterNum, int randomSeed,
                             bool use_cfg, float cfg_scale,
                             std::function<void(int)> progressCallback) {
    MNN_PRINT("[MossImage] input_embeds interface not supported\n");
    return false;
}

} // namespace DIFFUSION
} // namespace MNN
