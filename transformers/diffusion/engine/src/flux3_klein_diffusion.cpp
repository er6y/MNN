//  flux3_klein_diffusion.cpp - FLUX.3 Image Klein (step-distilled flow-map) diffusion model
//  Differences vs FLUX.2-Klein:
//   - DiT is a flow-map model: takes target_timestep (the sigma the step jumps to,
//     0 after the last step) in addition to timestep; Euler update unchanged.
//   - img_ids: (t=0, h*image_rope_scale, w*image_rope_scale, l); references use
//     l=-1,-2,... (FLUX.2 used t=10 offset, unscaled h/w).
//   - VAE decode denormalizes latents (x*bn_std + bn_mean) before decoding;
//     encode normalizes after patchify (same as FLUX.2).
//   - Text window is dynamic: ceil(real_tokens/pad_multiple)*pad_multiple, cap 8192.
#include <random>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include "diffusion/flux3_klein_diffusion.hpp"
#include "diffusion/diffusion_config.hpp"
#include "scheduler.hpp"
#include <rapidjson/document.h>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>
#include <MNN/expr/ExecutorScope.hpp>
using namespace CV;

namespace MNN {
namespace DIFFUSION {

namespace {
// diffusers get_1d_rotary_pos_embed(repeat_interleave_real=True) in fp64,
// Flux3PosEmbed axis concat: ids [S,4] -> cos/sin [S, sum(axes)]. Rope is
// precomputed host-side and fed to the transformer as graph inputs (the
// exported graph has no in-graph pos_embed).
static const int kRopeAxes[4] = {32, 32, 32, 32};
static const float kRopeTheta = 10000.f;

static void buildRope(const float* ids, int S, std::vector<float>& cosOut, std::vector<float>& sinOut) {
    int outDim = 0;
    for (int i = 0; i < 4; i++) outDim += kRopeAxes[i];
    cosOut.assign((size_t)S * outDim, 0.f);
    sinOut.assign((size_t)S * outDim, 0.f);
    int off = 0;
    for (int a = 0; a < 4; a++) {
        int dim = kRopeAxes[a], half = dim / 2;
        std::vector<double> freqs(half);
        for (int j = 0; j < half; j++) freqs[j] = 1.0 / std::pow((double)kRopeTheta, (2.0 * j) / (double)dim);
        for (int s = 0; s < S; s++) {
            double pos = ids[(size_t)s * 4 + a];
            for (int j = 0; j < half; j++) {
                double ang = pos * freqs[j];
                float c = (float)std::cos(ang), sn = (float)std::sin(ang);
                cosOut[(size_t)s * outDim + off + 2 * j] = c;
                cosOut[(size_t)s * outDim + off + 2 * j + 1] = c;
                sinOut[(size_t)s * outDim + off + 2 * j] = sn;
                sinOut[(size_t)s * outDim + off + 2 * j + 1] = sn;
            }
        }
        off += dim;
    }
}
} // namespace

// ===== Constructor =====
Flux3KleinDiffusion::Flux3KleinDiffusion(
    std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode,
    int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU,
    DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode,
    DiffusionCFGMode cfgMode, int numThreads)
    : Diffusion(modelPath, modelType, backendType, memoryMode,
                imageWidth, imageHeight, textEncoderOnCPU, vaeOnCPU,
                gpuMemoryMode, precisionMode, cfgMode, numThreads) {

    loadSchedulerConfig();
    for (const auto& schedPath : {mModelPath + "/scheduler/scheduler_config.json",
                                   mModelPath + "/scheduler_config.json"}) {
        std::ifstream f(schedPath.c_str());
        if (!f.good()) continue;
        std::ostringstream oss; oss << f.rdbuf();
        rapidjson::Document doc; doc.Parse(oss.str().c_str());
        if (doc.HasParseError() || !doc.IsObject()) continue;
        if (doc.HasMember("base_shift") && (doc["base_shift"].IsFloat() || doc["base_shift"].IsDouble()))
            mBaseShift = (float)doc["base_shift"].GetDouble();
        if (doc.HasMember("max_shift") && (doc["max_shift"].IsFloat() || doc["max_shift"].IsDouble()))
            mMaxShift = (float)doc["max_shift"].GetDouble();
        if (doc.HasMember("base_image_seq_len") && doc["base_image_seq_len"].IsInt())
            mBaseImageSeqLen = doc["base_image_seq_len"].GetInt();
        if (doc.HasMember("max_image_seq_len") && doc["max_image_seq_len"].IsInt())
            mMaxImageSeqLen = doc["max_image_seq_len"].GetInt();
        break;
    }

    // config.json: vae.bn_mean/bn_std (128ch, std precomputed with eps) +
    // dit.image_rope_scale + text window params
    {
        std::ifstream f((mModelPath + "/config.json").c_str());
        if (f.good()) {
            std::ostringstream oss; oss << f.rdbuf();
            rapidjson::Document doc; doc.Parse(oss.str().c_str());
            if (!doc.HasParseError() && doc.IsObject()) {
                if (doc.HasMember("vae") && doc["vae"].IsObject()) {
                    auto& vae = doc["vae"];
                    if (vae.HasMember("bn_mean") && vae["bn_mean"].IsArray()) {
                        for (auto& v : vae["bn_mean"].GetArray())
                            mVaeBnMean.push_back((float)v.GetDouble());
                    }
                    if (vae.HasMember("bn_std") && vae["bn_std"].IsArray()) {
                        for (auto& v : vae["bn_std"].GetArray())
                            mVaeBnStd.push_back((float)v.GetDouble());
                    }
                }
                if (doc.HasMember("dit") && doc["dit"].IsObject()) {
                    auto& dit = doc["dit"];
                    if (dit.HasMember("image_rope_scale") && (dit["image_rope_scale"].IsFloat() || dit["image_rope_scale"].IsDouble()))
                        mImageRopeScale = (float)dit["image_rope_scale"].GetDouble();
                }
                if (doc.HasMember("text_pad_multiple") && doc["text_pad_multiple"].IsInt())
                    mPadMultiple = doc["text_pad_multiple"].GetInt();
                if (doc.HasMember("max_text_len") && doc["max_text_len"].IsInt())
                    mMaxTextLen = doc["max_text_len"].GetInt();
                // fixed text window (static DiT export): pad every prompt to exactly this
                if (doc.HasMember("text_len") && doc["text_len"].IsInt() && doc["text_len"].GetInt() > 0)
                    mFixedTextLen = doc["text_len"].GetInt();
                // LongCat-style layout: tokenizer lives in the text_encoder directory
                if (doc.HasMember("text_encoder") && doc["text_encoder"].IsObject()) {
                    auto& te = doc["text_encoder"];
                    std::string dir, tok;
                    if (te.HasMember("directory") && te["directory"].IsString())
                        dir = te["directory"].GetString();
                    if (te.HasMember("tokenizer") && te["tokenizer"].IsString())
                        tok = te["tokenizer"].GetString();
                    if (!tok.empty()) {
                        mTokenizerPath = mModelPath + "/" + (dir.empty() ? "" : dir + "/") + tok;
                    }
                }
                // inference.height/width: the DiT is exported statically per size
                // (img_seq is baked into the graph), so the package config — not
                // the runtime request — decides the render size
                if (doc.HasMember("inference") && doc["inference"].IsObject()) {
                    auto& inf = doc["inference"];
                    if (inf.HasMember("width") && inf["width"].IsInt())
                        mConfigImageWidth = inf["width"].GetInt();
                    if (inf.HasMember("height") && inf["height"].IsInt())
                        mConfigImageHeight = inf["height"].GetInt();
                }
            }
        }
        if (!mVaeBnMean.empty())
            MNN_PRINT("[Flux3Klein] Loaded VAE BN params: %d channels, rope_scale=%.3f, pad=%d\n",
                      (int)mVaeBnMean.size(), mImageRopeScale, mPadMultiple);
        else
            MNN_PRINT("[Flux3Klein] Warning: VAE BN params not found in config.json\n");
    }

    // The DiT is exported with a dynamic image sequence, so any size the caller
    // asks for is valid. config.json inference.width/height is only the default
    // used when the request does not carry a size.
    if (imageWidth > 0 && imageHeight > 0) {
        mImageWidth  = imageWidth;
        mImageHeight = imageHeight;
        MNN_PRINT("[Flux3Klein] runtime size %dx%d\n", mImageWidth, mImageHeight);
    } else if (mConfigImageWidth > 0 && mConfigImageHeight > 0) {
        mImageWidth  = mConfigImageWidth;
        mImageHeight = mConfigImageHeight;
        MNN_PRINT("[Flux3Klein] default size from config: %dx%d\n", mImageWidth, mImageHeight);
    } else {
        mImageWidth = 1024; mImageHeight = 1024;
        MNN_PRINT("[Flux3Klein] no size given, fallback 1024x1024\n");
    }
    // Must align to 16: VAE scale=8 then patchify /2
    mImageWidth  = ((mImageWidth  + 15) / 16) * 16;
    mImageHeight = ((mImageHeight + 15) / 16) * 16;
    if (mImageWidth  < 256) mImageWidth  = 256;
    if (mImageHeight < 256) mImageHeight = 256;
    mLatentH = mImageHeight / 8;
    mLatentW = mImageWidth  / 8;
    MNN_PRINT("[Flux3Klein] image=%dx%d latent=%dx%d packed_seq=%d\n",
              mImageWidth, mImageHeight, mLatentH, mLatentW, (mLatentH/2)*(mLatentW/2));
}

Flux3KleinDiffusion::~Flux3KleinDiffusion() {
}

// ===== Scheduler helpers =====
float Flux3KleinDiffusion::computeEmpiricalMu(int imageSeqLen, int numSteps) const {
    float m = (mMaxShift - mBaseShift) / (float)(mMaxImageSeqLen - mBaseImageSeqLen);
    float b = mBaseShift - m * (float)mBaseImageSeqLen;
    return m * (float)imageSeqLen + b;
}

std::vector<float> Flux3KleinDiffusion::getSigmas(int numSteps, int imageSeqLen) const {
    std::vector<float> sigmas(numSteps + 1);
    for (int i = 0; i < numSteps; ++i) {
        float frac = (numSteps == 1) ? 0.f : (float)i / (float)(numSteps - 1);
        sigmas[i] = 1.0f - frac * (1.0f - 1.0f / (float)numSteps);
    }
    sigmas[numSteps] = 0.0f;
    if (mSchedulerConfig.useDynamicShifting) {
        float mu = computeEmpiricalMu(imageSeqLen, numSteps);
        float expMu = std::exp(mu);
        for (int i = 0; i < numSteps; ++i) {
            float s = sigmas[i];
            sigmas[i] = expMu * s / (expMu * s + (1.0f - s));
        }
    } else if (mSchedulerConfig.shift != 1.0f) {
        for (int i = 0; i < numSteps; ++i) {
            float s = sigmas[i];
            sigmas[i] = mSchedulerConfig.shift * s / (1.0f + (mSchedulerConfig.shift - 1.0f) * s);
        }
    }
    return sigmas;
}

// ===== Latent packing/unpacking =====
void Flux3KleinDiffusion::packLatentsToSeq(const float* src, float* dst,
                                            int B, int C, int H, int W) const {
    int pH = H/2, pW = W/2, pC = C*4;
    for (int b = 0; b < B; ++b)
        for (int ph = 0; ph < pH; ++ph)
            for (int pw = 0; pw < pW; ++pw) {
                int si = ph*pW + pw;
                for (int c = 0; c < C; ++c)
                    for (int dh = 0; dh < 2; ++dh)
                        for (int dw = 0; dw < 2; ++dw)
                            dst[b*pH*pW*pC + si*pC + c*4+dh*2+dw] =
                                src[b*C*H*W + c*H*W + (ph*2+dh)*W + (pw*2+dw)];
            }
}

void Flux3KleinDiffusion::unpackLatentsToPatchified(const float* src, float* dst,
                                                     int B, int C, int H, int W, int seqLen) const {
    int pH = H/2, pW = W/2, pC = C*4;
    for (int b = 0; b < B; ++b)
        for (int ph = 0; ph < pH; ++ph)
            for (int pw = 0; pw < pW; ++pw) {
                int si = ph*pW + pw;
                for (int fc = 0; fc < pC; ++fc)
                    dst[b*pC*pH*pW + fc*pH*pW + ph*pW+pw] =
                        src[b*seqLen*pC + si*pC + fc];
            }
}

void Flux3KleinDiffusion::prepareImgIds(float* dst, int H, int W, int seqOffset, float lCoord) const {
    // ids format: (t=0, h*scale, w*scale, l)
    int pH = H/2, pW = W/2;
    for (int h = 0; h < pH; ++h)
        for (int w = 0; w < pW; ++w) {
            int idx = seqOffset + h*pW + w;
            dst[idx*ID_DIM+0]=0.f;                      // t=0
            dst[idx*ID_DIM+1]=(float)h*mImageRopeScale; // h scaled
            dst[idx*ID_DIM+2]=(float)w*mImageRopeScale; // w scaled
            dst[idx*ID_DIM+3]=lCoord;                   // l: 0 for noise, -1.. for refs
        }
}

void Flux3KleinDiffusion::prepareTxtIds(float* dst, int seqLen) const {
    for (int i = 0; i < seqLen; ++i) {
        dst[i*ID_DIM+0]=0.f;
        dst[i*ID_DIM+1]=0.f;
        dst[i*ID_DIM+2]=0.f;
        dst[i*ID_DIM+3]=(float)i;
    }
}

// ===== load() =====
bool Flux3KleinDiffusion::load() {
    AUTOTIME;
    // Flash attention (block 64) for the long joint sequences
    if (!initRuntimeManagers(/*gpuBufferMode=*/true, /*attentionHint=*/8)) return false;

    DiffusionConfig diff_config(mModelPath);
    // 0=TE, 1=DiT(T2I), 2=VAE dec, 3=VAE enc, 4=DiT(edit, image_cond variant)
    mModules.resize(5);

    // [0] Text encoder
    {
        auto path = diff_config.text_encoder_model();
        MNN_PRINT("[Flux3Klein] Load text encoder: %s\n", path.c_str());
        Module::Config tec; tec.shapeMutable = true;
        auto& te_runtime = runtime_manager_cpu_ ? runtime_manager_cpu_ : runtime_manager_;
        pointWeightPoolAt("te");
        mModules[0].reset(Module::load({"input_ids","attention_mask","position_ids"},{"prompt_embeds"}, path.c_str(), te_runtime, &tec));
        if (!mModules[0]) { MNN_ERROR("[Flux3Klein] Failed to load text encoder\n"); return false; }
    }

    if (mMemoryMode != 0) {
        Module::Config mc; mc.shapeMutable = true;
        auto& vae_runtime = runtime_manager_vae_cpu_ ? runtime_manager_vae_cpu_ : runtime_manager_;
        {
            auto path = diff_config.unet_model();
            MNN_PRINT("[Flux3Klein] Load transformer: %s\n", path.c_str());
            pointWeightPoolAt("dit");
            mModules[1].reset(Module::load(
                {"hidden_states","timestep","encoder_hidden_states","target_timestep",
                 "txt_rope_cos","txt_rope_sin","img_rope_cos","img_rope_sin"},
                {"out_sample"}, path.c_str(), runtime_manager_, &mc));
            if (!mModules[1]) { MNN_ERROR("[Flux3Klein] Failed to load transformer\n"); return false; }
        }
        {
            auto path = diff_config.vae_decoder_model();
            MNN_PRINT("[Flux3Klein] Load VAE decoder: %s\n", path.c_str());
            pointWeightPoolAt("vae_dec");
            mModules[2].reset(Module::load({"latent_sample"},{"sample"}, path.c_str(), vae_runtime, &mc));
            if (!mModules[2]) { MNN_ERROR("[Flux3Klein] Failed to load VAE decoder\n"); return false; }
            mModules[2]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
        }
        {
            auto path = diff_config.vae_encoder_model();
            MNN_PRINT("[Flux3Klein] Load VAE encoder: %s\n", path.c_str());
            pointWeightPoolAt("vae_enc");
            mModules[3].reset(Module::load({"sample"},{"latent_sample"}, path.c_str(), vae_runtime, &mc));
            if (!mModules[3]) { MNN_ERROR("[Flux3Klein] Failed to load VAE encoder\n"); return false; }
            mModules[3]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
        }
    } else {
        MNN_PRINT("[Flux3Klein] Low memory mode: transformer, VAE decoder, VAE encoder will be loaded on demand\n");
    }
    mDiffConfig.reset(new DiffusionConfig(mModelPath));
    return true;
}

// ===== Text Encoder =====
// Hardcoded Qwen3 chat_template (user-only, no thinking, no system prompt),
// matches AutoProcessor.apply_chat_template([{"role":"user","content":prompt}],
//   tokenize=False, add_generation_prompt=True)
VARP Flux3KleinDiffusion::text_encoder_llm(const std::string& prompt) {
    AUTOTIME;
    if (!mModules[0]) { MNN_PRINT("[Flux3Klein] Error: text encoder not loaded\n"); return nullptr; }

    std::vector<int> inputIds;

#ifdef MNN_BUILD_LLM
    if (!mTokenizer) {
        std::string tokPath = mTokenizerPath.empty() ? (mModelPath + "/tokenizer.txt") : mTokenizerPath;
        mTokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(tokPath));
        if (mTokenizer)
            MNN_PRINT("[Flux3Klein] Tokenizer loaded: %s\n", tokPath.c_str());
        else
            MNN_PRINT("[Flux3Klein] Warning: tokenizer load failed: %s\n", tokPath.c_str());
    }
    if (mTokenizer) {
        std::string templated = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        inputIds = mTokenizer->encode(templated);
        MNN_PRINT("[Flux3Klein] Tokens after chat_template: %d\n", (int)inputIds.size());
    }
#endif

    if (inputIds.empty()) {
        MNN_PRINT("[Flux3Klein] Warning: tokenize failed, using dummy tokens\n");
        inputIds.resize(mPadMultiple, 0);
    }

    int realLen = (int)inputIds.size();
    if (realLen > mMaxTextLen) {
        inputIds.resize(mMaxTextLen);
        realLen = mMaxTextLen;
    }
    // window: fixed (static DiT export) when configured, else the reference
    // ceil-to-pad-multiple rule
    int seqLen;
    if (mFixedTextLen > 0) {
        if (realLen > mFixedTextLen) {
            inputIds.resize(mFixedTextLen);
            realLen = mFixedTextLen;
        }
        seqLen = mFixedTextLen;
    } else {
        seqLen = ((realLen + mPadMultiple - 1) / mPadMultiple) * mPadMultiple;
        if (seqLen < mPadMultiple) seqLen = mPadMultiple;
    }
    inputIds.resize(seqLen, 0);

    INTS idShape = {1, seqLen};
    auto idsVar  = _Input(idShape, NCHW, halide_type_of<int>());
    auto maskVar = _Input(idShape, NCHW, halide_type_of<int>());
    auto posVar  = _Input(idShape, NCHW, halide_type_of<int>());
    auto idsPtr  = idsVar->writeMap<int>();
    auto maskPtr = maskVar->writeMap<int>();
    auto posPtr  = posVar->writeMap<int>();
    for (int i = 0; i < seqLen; ++i) {
        idsPtr[i]  = inputIds[i];
        maskPtr[i] = (i < realLen) ? 1 : 0;
        // reference: position_ids = arange(S).masked_fill(pad -> 1)
        posPtr[i]  = (i < realLen) ? i : 1;
    }

    auto outs = mModules[0]->onForward({idsVar, maskVar, posVar});
    if (outs.empty()) { MNN_PRINT("[Flux3Klein] Text encoder forward failed\n"); return nullptr; }

    auto embeds = _Convert(outs[0], NCHW);
    embeds.fix(VARP::CONSTANT);
    auto fi = embeds->getInfo();
    MNN_PRINT("[Flux3Klein] Text embeds: [%d,%d,%d]\n", fi->dim[0], fi->dim[1], fi->dim[2]);
    return embeds;
}

// ===== VAE Decoder =====
// Denoised patchified latents live in BN-normalized space: denormalize
// (x * bn_std + bn_mean) BEFORE decode.
VARP Flux3KleinDiffusion::vae_decoder(VARP latent) {
    if (mMemoryMode != 1) {
        mModules[1].reset();
        MNN_PRINT("[Flux3Klein] Transformer unloaded\n");
    }
    if (!mModules[2]) {
        Module::Config mc; mc.shapeMutable = true;
        auto path = mDiffConfig->vae_decoder_model();
        MNN_PRINT("[Flux3Klein] Load VAE decoder on demand: %s\n", path.c_str());
        pointWeightPoolAt("vae_dec");
        auto& vae_runtime = runtime_manager_vae_cpu_ ? runtime_manager_vae_cpu_ : runtime_manager_;
        mModules[2].reset(Module::load({"latent_sample"},{"sample"}, path.c_str(), vae_runtime, &mc));
        if (mModules[2]) mModules[2]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
        if (!mModules[2]) { MNN_PRINT("[Flux3Klein] Failed to load VAE decoder\n"); return nullptr; }
    }
    AUTOTIME;
    auto info = latent->getInfo();
    int B = info->dim[0], C = info->dim[1], H = info->dim[2], W = info->dim[3];
    // Input: packed patchified latents [1,128,pH,pW] in BN-normalized space.
    // Denormalize, then unpatchify -> [1,32,2pH,2pW] for the VAE decoder.
    int LC = C / 4, LH = H * 2, LW = W * 2;
    auto denormVar = _Input({B, LC, LH, LW}, NCHW, halide_type_of<float>());
    {
        const float* src = latent->readMap<float>();
        float* dst = denormVar->writeMap<float>();
        size_t plane = (size_t)H * W;
        bool hasBn = (int)mVaeBnMean.size() == C && (int)mVaeBnStd.size() == C;
        if (!hasBn)
            MNN_PRINT("[Flux3Klein] Warning: BN params missing on decode, skipping denorm\n");
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < LC; ++c)
                for (int dh = 0; dh < 2; ++dh)
                    for (int dw = 0; dw < 2; ++dw) {
                        int fc = c*4 + dh*2 + dw;
                        float mean = hasBn ? mVaeBnMean[fc] : 0.f;
                        float stdv = hasBn ? mVaeBnStd[fc] : 1.f;
                        const float* srcPlane = src + (size_t)b*C*plane + fc*plane;
                        for (int ph = 0; ph < H; ++ph)
                            for (int pw = 0; pw < W; ++pw)
                                dst[((size_t)b*LC + c)*LH*LW + (size_t)(ph*2+dh)*LW + (pw*2+dw)] =
                                    srcPlane[ph*W + pw] * stdv + mean;
                    }
        denormVar.fix(VARP::CONSTANT);
    }
    auto outs = mModules[2]->onForward({denormVar});
    if (outs.empty()) { MNN_PRINT("[Flux3Klein] VAE decode failed\n"); return nullptr; }
    return nchwFloatToHwcBGR(_Convert(outs[0], NCHW));
}

// ===== VAE Encoder =====
// vae.encode -> mode -> patchify -> BN normalize -> [1, 128, H/16, W/16]
VARP Flux3KleinDiffusion::vae_encoder(VARP image) {
    if (!mModules[3]) {
        Module::Config mc; mc.shapeMutable = true;
        auto path = mDiffConfig->vae_encoder_model();
        MNN_PRINT("[Flux3Klein] Load VAE encoder on demand: %s\n", path.c_str());
        pointWeightPoolAt("vae_enc");
        auto& vae_runtime = runtime_manager_vae_cpu_ ? runtime_manager_vae_cpu_ : runtime_manager_;
        mModules[3].reset(Module::load({"sample"},{"latent_sample"}, path.c_str(), vae_runtime, &mc));
        if (!mModules[3]) { MNN_PRINT("[Flux3Klein] Failed to load VAE encoder\n"); return nullptr; }
    }
    AUTOTIME;
    auto outs = mModules[3]->onForward({image});
    if (outs.empty()) { MNN_PRINT("[Flux3Klein] VAE encode failed\n"); return nullptr; }
    auto lat = _Convert(outs[0], NCHW);
    lat.fix(VARP::CONSTANT);

    auto info = lat->getInfo();
    int B = info->dim[0], C = info->dim[1], H = info->dim[2], W = info->dim[3];
    int pH = H/2, pW = W/2, pC = C*4;
    std::vector<float> patchData(B * pC * pH * pW);
    const float* src = lat->readMap<float>();
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int ph = 0; ph < pH; ++ph)
                for (int pw = 0; pw < pW; ++pw)
                    for (int dh = 0; dh < 2; ++dh)
                        for (int dw = 0; dw < 2; ++dw) {
                            int srcIdx = b*C*H*W + c*H*W + (ph*2+dh)*W + (pw*2+dw);
                            int fc = c*4 + dh*2 + dw;
                            int dstIdx = b*pC*pH*pW + fc*pH*pW + ph*pW + pw;
                            patchData[dstIdx] = src[srcIdx];
                        }

    if ((int)mVaeBnMean.size() == pC && (int)mVaeBnStd.size() == pC) {
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < pC; ++c) {
                float mean = mVaeBnMean[c], bnStd = mVaeBnStd[c];
                for (int i = 0; i < pH*pW; ++i)
                    patchData[b*pC*pH*pW + c*pH*pW + i] = (patchData[b*pC*pH*pW + c*pH*pW + i] - mean) / bnStd;
            }
    } else {
        MNN_PRINT("[Flux3Klein] Warning: BN params missing, skipping BN normalize\n");
    }

    auto patchVar = _Input({B, pC, pH, pW}, NCHW, halide_type_of<float>());
    memcpy(patchVar->writeMap<float>(), patchData.data(), patchData.size()*sizeof(float));
    patchVar.fix(VARP::CONSTANT);
    MNN_PRINT("[Flux3Klein] VAE encoded+patchified: [%d,%d,%d,%d]\n", B, pC, pH, pW);
    return patchVar;
}

// ===== UNet (Denoising Loop, flow-map) =====
VARP Flux3KleinDiffusion::unet(VARP textEmbeds, VARP imageLatents,
                                int iterNum, int randomSeed,
                                std::function<void(int)> progressCallback) {
#ifdef MNN_BUILD_LLM
    if (mMemoryMode != 1 && mTokenizer) {
        mTokenizer.reset();
        MNN_PRINT("[Flux3Klein] Tokenizer unloaded\n");
    }
#endif
    if (mMemoryMode != 1 && textEmbeds.get()) {
        auto info = textEmbeds->getInfo();
        if (info) {
            size_t n = 1; for (auto d : info->dim) n *= d;
            std::vector<float> buf(n);
            const float* src = textEmbeds->readMap<float>();
            if (src) memcpy(buf.data(), src, n * sizeof(float));
            VARP tmp = _Input(info->dim, info->order, halide_type_of<float>());
            memcpy(tmp->writeMap<float>(), buf.data(), n * sizeof(float));
            tmp.fix(VARP::CONSTANT);
            textEmbeds = tmp;
        }
        mModules[0].reset();
        MNN_PRINT("[Flux3Klein] Text encoder unloaded\n");
    }
    if (!mModules[1]) {
        Module::Config mc; mc.shapeMutable = true;
        auto path = mDiffConfig->unet_model();
        MNN_PRINT("[Flux3Klein] Load transformer on demand: %s\n", path.c_str());
        pointWeightPoolAt("dit");
        mModules[1].reset(Module::load(
            {"hidden_states","timestep","encoder_hidden_states","target_timestep",
             "txt_rope_cos","txt_rope_sin","img_rope_cos","img_rope_sin"},
            {"out_sample"}, path.c_str(), runtime_manager_, &mc));
        if (!mModules[1]) { MNN_PRINT("[Flux3Klein] Failed to load transformer\n"); return nullptr; }
    }
    bool isT2I = !imageLatents.get();
    // edit DiT (image_cond variant): separate graph with cond-token inputs,
    // loaded on demand (the T2I graph cannot take reference tokens — FLUX.3
    // gives references their own embedder/modulations/mode blocks)
    if (!isT2I && !mModules[4]) {
        auto editPath = mDiffConfig->unet_edit_model();
        if (editPath.empty()) {
            MNN_PRINT("[Flux3Klein] Error: edit requested but config has no transformer.edit_model\n");
            return nullptr;
        }
        Module::Config mc; mc.shapeMutable = true;
        MNN_PRINT("[Flux3Klein] Load edit transformer on demand: %s\n", editPath.c_str());
        mModules[4].reset(Module::load(
            {"hidden_states","timestep","encoder_hidden_states","target_timestep",
             "txt_rope_cos","txt_rope_sin","img_rope_cos","img_rope_sin",
             "image_cond_hidden_states","cond_rope_cos","cond_rope_sin"},
            {"out_sample"}, editPath.c_str(), runtime_manager_, &mc));
        if (!mModules[4]) { MNN_PRINT("[Flux3Klein] Failed to load edit transformer\n"); return nullptr; }
    }
    int pH = mLatentH/2, pW = mLatentW/2;
    int singleSeq = pH * pW;
    int imgSeqLen = singleSeq;
    int condSeqLen = isT2I ? 0 : singleSeq;
    int pC = mPackedC;

    MNN_PRINT("[Flux3Klein] %s: singleSeq=%d condSeq=%d\n",
              isT2I ? "T2I" : "Edit", singleSeq, condSeqLen);

    int noiseSize = pC * pH * pW;
    std::vector<float> noiseData(noiseSize);
    {
        int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
        generateLatentNoise(noiseData.data(), noiseSize, seed);
    }

    auto sigmas = getSigmas(iterNum, singleSeq);

    int txtSeqLen = textEmbeds->getInfo()->dim[1];
    std::vector<float> txtIdsData(txtSeqLen * ID_DIM, 0.f);
    prepareTxtIds(txtIdsData.data(), txtSeqLen);

    std::vector<float> imgIdsData(imgSeqLen * ID_DIM, 0.f);
    // noise latents: l=0; reference image tokens sit on the same grid with l=-1
    // (fed as a separate rope input to the edit graph)
    prepareImgIds(imgIdsData.data(), mLatentH, mLatentW, 0, 0.f);
    std::vector<float> condIdsData;
    if (!isT2I) {
        condIdsData.assign((size_t)condSeqLen * ID_DIM, 0.f);
        prepareImgIds(condIdsData.data(), mLatentH, mLatentW, 0, -1.f);
    }
    // (ids feed host-side rope only; the exported DiT graph takes rope cos/sin)

    std::vector<float> imageLatentsPacked;
    if (!isT2I) {
        auto imgInfo = imageLatents->getInfo();
        int iB = imgInfo->dim[0], iC = imgInfo->dim[1], iH = imgInfo->dim[2], iW = imgInfo->dim[3];
        int imgSeq = iH * iW;
        MNN_ASSERT(imgSeq == singleSeq && iC == pC);
        imageLatentsPacked.resize(singleSeq * pC);
        const float* imgSrc = imageLatents->readMap<float>();
        for (int c = 0; c < pC; ++c)
            for (int i = 0; i < singleSeq; ++i)
                imageLatentsPacked[i * pC + c] = imgSrc[c * singleSeq + i];
    }

    // FLUX.3 reference draws noise directly in packed space [1,128,pH,pW] then
    // flatten(2).transpose(1,2): token (ph,pw) major, packed channel c minor
    std::vector<float> latentSeq(singleSeq * pC);
    for (int c = 0; c < pC; ++c)
        for (int i = 0; i < singleSeq; ++i)
            latentSeq[i * pC + c] = noiseData[c * singleSeq + i];

    // Precomputed rope (host-side): per-sequence cos/sin [seq,128]; ids fixed across steps
    std::vector<float> txtCos, txtSin, imgCosAll, imgSinAll;
    buildRope(txtIdsData.data(), txtSeqLen, txtCos, txtSin);
    buildRope(imgIdsData.data(), imgSeqLen, imgCosAll, imgSinAll);
    std::vector<float> condCosAll, condSinAll;
    if (!isT2I) buildRope(condIdsData.data(), condSeqLen, condCosAll, condSinAll);
    auto txtCosVar = _Input({txtSeqLen, 128}, NCHW, halide_type_of<float>());
    auto txtSinVar = _Input({txtSeqLen, 128}, NCHW, halide_type_of<float>());
    auto imgCosVar = _Input({imgSeqLen, 128}, NCHW, halide_type_of<float>());
    auto imgSinVar = _Input({imgSeqLen, 128}, NCHW, halide_type_of<float>());
    memcpy(txtCosVar->writeMap<float>(), txtCos.data(), txtCos.size()*sizeof(float));
    memcpy(txtSinVar->writeMap<float>(), txtSin.data(), txtSin.size()*sizeof(float));
    memcpy(imgCosVar->writeMap<float>(), imgCosAll.data(), imgCosAll.size()*sizeof(float));
    memcpy(imgSinVar->writeMap<float>(), imgSinAll.data(), imgSinAll.size()*sizeof(float));
    txtCosVar.fix(VARP::CONSTANT);
    txtSinVar.fix(VARP::CONSTANT);
    imgCosVar.fix(VARP::CONSTANT);
    imgSinVar.fix(VARP::CONSTANT);
    VARP condVar, condCosVar, condSinVar;
    if (!isT2I) {
        condVar = _Input({1, condSeqLen, pC}, NCHW, halide_type_of<float>());
        memcpy(condVar->writeMap<float>(), imageLatentsPacked.data(), imageLatentsPacked.size()*sizeof(float));
        condCosVar = _Input({condSeqLen, 128}, NCHW, halide_type_of<float>());
        condSinVar = _Input({condSeqLen, 128}, NCHW, halide_type_of<float>());
        memcpy(condCosVar->writeMap<float>(), condCosAll.data(), condCosAll.size()*sizeof(float));
        memcpy(condSinVar->writeMap<float>(), condSinAll.data(), condSinAll.size()*sizeof(float));
        condVar.fix(VARP::CONSTANT);
        condCosVar.fix(VARP::CONSTANT);
        condSinVar.fix(VARP::CONSTANT);
    }

    std::vector<float> curLatent = latentSeq;

    MNN_PRINT("[Flux3Klein] Denoising %d steps (flow-map), sigma[0]=%.4f\n", iterNum, sigmas[0]);

    for (int i = 0; i < iterNum; ++i) {
        AUTOTIME;
        float sigma      = sigmas[i];
        float sigma_next = sigmas[i + 1];
        float dt         = sigma_next - sigma;

        std::vector<float> sampleData(imgSeqLen * pC);
        memcpy(sampleData.data(), curLatent.data(), singleSeq * pC * sizeof(float));

        auto sampleVar   = _Input({1, imgSeqLen, pC}, NCHW, halide_type_of<float>());
        auto timestepVar = _Input({1}, NCHW, halide_type_of<float>());
        auto targetVar   = _Input({1}, NCHW, halide_type_of<float>());
        memcpy(sampleVar->writeMap<float>(), sampleData.data(), sampleData.size()*sizeof(float));
        timestepVar->writeMap<float>()[0] = sigma;
        // flow-map: the sigma this step jumps to; 0 after the last step
        targetVar->writeMap<float>()[0] = (i + 1 < iterNum) ? sigma_next : 0.f;

        std::vector<VARP> inputs = {sampleVar, timestepVar, textEmbeds, targetVar,
                                    txtCosVar, txtSinVar, imgCosVar, imgSinVar};
        if (!isT2I) {
            // edit graph: reference tokens are a separate input; the output is
            // already the noise-part velocity (graph slices internally)
            inputs.push_back(condVar);
            inputs.push_back(condCosVar);
            inputs.push_back(condSinVar);
        }
        auto& ditModule = isT2I ? mModules[1] : mModules[4];
        auto outs = ditModule->onForward(inputs);
        if (outs.empty()) {
            MNN_PRINT("[Flux3Klein] Transformer failed at step %d\n", i);
            return nullptr;
        }
        auto noisePred = _Convert(outs[0], NCHW);
        noisePred.fix(VARP::CONSTANT);

        const float* npPtr = noisePred->readMap<float>();
        for (int j = 0; j < singleSeq * pC; ++j)
            curLatent[j] += dt * npPtr[j];

        if (mBackendType == MNN_FORWARD_OPENCL)
            MNN::Express::ExecutorScope::Current()->gc(MNN::Express::Executor::PART);

        if (progressCallback) progressCallback((2 + i) * 100 / (iterNum + 3));
        MNN_PRINT("[Flux3Klein] Step %d/%d sigma=%.4f target=%.4f\n", i+1, iterNum, sigma,
                  (i + 1 < iterNum) ? sigma_next : 0.f);
    }

    int pH2 = mLatentH/2, pW2 = mLatentW/2;
    if (!isT2I && mMemoryMode != 1) {
        mModules[4].reset();
        MNN_PRINT("[Flux3Klein] Edit transformer unloaded\n");
    }
    std::vector<float> patchifiedData(pC * pH2 * pW2);
    unpackLatentsToPatchified(curLatent.data(), patchifiedData.data(),
                              1, mLatentC, mLatentH, mLatentW, singleSeq);

    auto patchifiedVar = _Input({1, pC, pH2, pW2}, NCHW, halide_type_of<float>());
    memcpy(patchifiedVar->writeMap<float>(), patchifiedData.data(), patchifiedData.size()*sizeof(float));
    patchifiedVar.fix(VARP::CONSTANT);
    return patchifiedVar;
}

// ===== run() overloads =====
bool Flux3KleinDiffusion::run(const std::string prompt, const std::string imagePath,
                               int iterNum, int randomSeed,
                               std::function<void(int)> progressCallback) {
    return run(prompt, imagePath, iterNum, randomSeed, 1.0f, progressCallback, "");
}

bool Flux3KleinDiffusion::run(const std::string prompt, const std::string outputPath,
                               int iterNum, int randomSeed, float cfgScale,
                               std::function<void(int)> progressCallback,
                               const std::string inputImagePath) {
    AUTOTIME;
    bool isT2I = inputImagePath.empty();
    VARP imageLatents = nullptr;

    if (!isT2I) {
        MNN_PRINT("[Flux3Klein] Image Edit: %s\n", inputImagePath.c_str());
        auto rawImage = CV::imread(inputImagePath);
        if (!rawImage.get()) {
            MNN_PRINT("[Flux3Klein] Error: cannot load %s\n", inputImagePath.c_str());
            return false;
        }
        auto processed = resizeAndCenterCrop(rawImage, mImageWidth, mImageHeight);
        auto rgbImage  = bgrToRgb(processed);
        auto inputNorm = hwcToNchw(rgbImage, true);
        imageLatents = vae_encoder(inputNorm);
        if (!imageLatents.get()) { MNN_PRINT("[Flux3Klein] VAE encode failed\n"); return false; }
        {
            auto info = imageLatents->getInfo();
            if (info) {
                size_t n = 1; for (auto d : info->dim) n *= d;
                std::vector<float> buf(n);
                const float* src = imageLatents->readMap<float>();
                if (src) memcpy(buf.data(), src, n * sizeof(float));
                VARP tmp = _Input(info->dim, info->order, halide_type_of<float>());
                memcpy(tmp->writeMap<float>(), buf.data(), n * sizeof(float));
                tmp.fix(VARP::CONSTANT);
                imageLatents = tmp;
            }
        }
        if (mMemoryMode != 1) mModules[3].reset();
    }

    if (iterNum < 1)  iterNum = 16;
    if (iterNum > 50) iterNum = 50;

    if (progressCallback) progressCallback(0);

    auto textEmbeds = text_encoder_llm(prompt);
    if (!textEmbeds.get()) { MNN_PRINT("[Flux3Klein] Text encode failed\n"); return false; }
    if (progressCallback) progressCallback(1 * 100 / (iterNum + 3));
    auto latent = unet(textEmbeds, imageLatents, iterNum, randomSeed, progressCallback);
    if (!latent.get()) { MNN_PRINT("[Flux3Klein] UNet failed\n"); return false; }

    auto image = vae_decoder(latent);
    if (!image.get()) { MNN_PRINT("[Flux3Klein] VAE decode failed\n"); return false; }

    bool res = CV::imwrite(outputPath, image);
    if (res) MNN_PRINT("[Flux3Klein] Saved: %s\n", outputPath.c_str());
    else     MNN_PRINT("[Flux3Klein] Error: imwrite failed: %s\n", outputPath.c_str());

    if (mMemoryMode != 1) {
        mModules[2].reset();
        MNN_PRINT("[Flux3Klein] VAE decoder unloaded\n");
    }
    if (progressCallback) progressCallback(100);
    return res;
}

bool Flux3KleinDiffusion::run(const VARP input_embeds, const std::string& mode,
                               const std::string& inputImagePath, const std::string& outputImagePath,
                               int width, int height, int iterNum, int randomSeed,
                               bool use_cfg, float cfg_scale,
                               std::function<void(int)> progressCallback) {
    MNN_PRINT("[Flux3Klein] input_embeds interface not supported\n");
    return false;
}

} // namespace DIFFUSION
} // namespace MNN
