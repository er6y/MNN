//  qwenimage21_diffusion.cpp - Qwen-Image-2.1 text-to-image engine.
//  Behavior mirrors the verified offline chain (test_pipeline.cpp):
//  token ids -> bf16 embedding lookup -> TE(llm.mnn) -> drop system prefix (14 rows)
//  -> pad to text_len 1024 -> DiT N-step flow-match euler -> de-normalize
//  -> VAE(dynamic) -> BGR png.
//
//  Expected model dir config.json (all paths relative to the model dir):
//  {
//    "text_encoder_dir": "te", "text_encoder_model": "llm.mnn",
//    "embeddings_bf16": "te/embeddings_bf16.bin", "tokenizer": "te/tokenizer.txt",
//    "transformer": "dit/unet.mnn", "vae_decoder": "vae/vae_decoder.mnn",
//    "latent_stats": "vae/latent_stats.f32", "text_len": 1024, "drop_idx": 14,
//    "weight_mmap_dir": "mmap_cache", "weight_mmap_cached": true
//  }
#include "diffusion/qwenimage21_diffusion.hpp"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <rapidjson/document.h>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>
#include <MNN/expr/ExecutorScope.hpp>

namespace MNN {
namespace DIFFUSION {

using namespace Express;
using namespace CV;

// Qwen-Image-2.1 T2I chat template system message (verified against te_full_ids.txt,
// the token ids the whole MNN chain was validated with).
static const char* kSysPrompt = "Comprehend and analyze the provided prompt.";

// M-RoPE indices for T2I (mirrors diffusers QwenImage21Rope.forward for one image block):
// text tokens advance all three axes together; image tokens freeze frame at the position
// reached by the text and lay h/w on centred grids. Negative h/w indices are PRE-MAPPED
// positive (mod 9216, the freqs-table rows) because the exported DiT gathers the tables
// directly — MNN lowers even int32 Mod to a float64 binary it cannot execute.
static void buildRopeIndices(int textLen, int latentH, int latentW,
                             std::vector<int32_t>* frame, std::vector<int32_t>* h,
                             std::vector<int32_t>* w) { /* ROIPE-I32 */
    const int imgSeq = latentH * latentW;
    const int32_t kRopeTable = 9216;
    auto rem = [&](int32_t v) { return ((v % kRopeTable) + kRopeTable) % kRopeTable; };
    frame->clear(); h->clear(); w->clear();
    frame->reserve(textLen + imgSeq);
    h->reserve(textLen + imgSeq);
    w->reserve(textLen + imgSeq);
    const int pos = textLen; // frame value for image tokens: position after the text
    for (int i = 0; i < textLen; i++) {
        frame->push_back(i);
        h->push_back(i);
        w->push_back(i);
    }
    for (int r = 0; r < latentH; r++) {
        for (int c = 0; c < latentW; c++) {
            frame->push_back(pos);
            h->push_back(rem(r - (latentH - latentH / 2)));
            w->push_back(rem(c - (latentW - latentW / 2)));
        }
    }
}

// --- torch CPU randn replica (verified vs pipeline_io/latents_init.f32, cosine 0.9999986) ---
// torch.Generator("cpu").manual_seed(s) -> MT19937; randn(float32) -> 24-bit uniforms +
// normal_fill_16 Box-Muller pairs over the contiguous tensor.
namespace torchcpu {
struct MT19937 {
    static const int NN = 624, MM = 397;
    uint32_t mt[NN];
    int mti;
    explicit MT19937(uint64_t seed) {
        mt[0] = (uint32_t)(seed & 0xffffffffULL);
        for (mti = 1; mti < NN; mti++)
            mt[mti] = (1812433253U * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    }
    uint32_t next() {
        uint32_t y;
        if (mti >= NN) {
            const uint32_t MATRIX_A = 0x9908b0dfU, UPPER = 0x80000000U, LOWER = 0x7fffffffU;
            int kk;
            for (kk = 0; kk < NN - MM; kk++) {
                y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + MM] ^ (y >> 1) ^ ((y & 1) ? MATRIX_A : 0);
            }
            for (; kk < NN - 1; kk++) {
                y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + (MM - NN)] ^ (y >> 1) ^ ((y & 1) ? MATRIX_A : 0);
            }
            y = (mt[NN - 1] & UPPER) | (mt[0] & LOWER);
            mt[NN - 1] = mt[MM - 1] ^ (y >> 1) ^ ((y & 1) ? MATRIX_A : 0);
            mti = 0;
        }
        y = mt[mti++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }
};
static inline float uniform01(MT19937& g) {
    return (float)(g.next() & 0xFFFFFF) * (1.0f / (float)(1 << 24));
}
} // namespace torchcpu

static void torchCpuRandn(float* data, size_t size, uint64_t seed) {
    torchcpu::MT19937 g(seed);
    for (size_t i = 0; i + 15 < size; i += 16) {
        for (int j = 0; j < 16; j++) data[i + j] = torchcpu::uniform01(g);
        for (int j = 0; j < 8; j++) {
            const float u1 = 1 - data[i + j]; // (0, 1]
            const float u2 = data[i + j + 8]; // [0, 1)
            const float radius = std::sqrt(-2.0f * std::log(u1));
            const float theta  = 2.0f * 3.14159265358979323846 * u2;
            data[i + j]     = radius * std::cos(theta);
            data[i + j + 8] = radius * std::sin(theta);
        }
    }
}

QwenImage21Diffusion::QwenImage21Diffusion(std::string modelPath, DiffusionModelType modelType,
                                           MNNForwardType backendType, int memoryMode)
    : Diffusion(modelPath, modelType, backendType, memoryMode) {
    readEngineConfig();
}

QwenImage21Diffusion::QwenImage21Diffusion(std::string modelPath, DiffusionModelType modelType,
                                           MNNForwardType backendType, int memoryMode, int imageWidth,
                                           int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU,
                                           DiffusionGpuMemoryMode gpuMemoryMode,
                                           DiffusionPrecisionMode precisionMode, DiffusionCFGMode cfgMode,
                                           int numThreads)
    : Diffusion(modelPath, modelType, backendType, memoryMode, imageWidth, imageHeight, textEncoderOnCPU,
                vaeOnCPU, gpuMemoryMode, precisionMode, cfgMode, numThreads) {
    readEngineConfig();
}

QwenImage21Diffusion::~QwenImage21Diffusion() {
    mModules.clear();
}

void QwenImage21Diffusion::readEngineConfig() {
    // engine-specific fields (weight_mmap_* are consumed by the Diffusion base ctor)
    std::string path = mModelPath + "/config.json";
    std::ifstream f(path);
    mEmbeddingsFile = mModelPath + "/" + mTeDir + "/embeddings_bf16.bin";
    mTokenizerFile  = mModelPath + "/" + mTeDir + "/tokenizer.txt";
    if (!f.is_open()) {
        MNN_PRINT("[QwenImage21] no config.json at %s, using defaults\n", path.c_str());
        return;
    }
    rapidjson::IStreamWrapper isw(f);
    rapidjson::Document doc;
    doc.ParseStream(isw);
    if (doc.HasParseError() || !doc.IsObject()) {
        MNN_PRINT("[QwenImage21] config.json parse failed, using defaults\n");
        return;
    }
    auto getStr = [&](const char* k, std::string& v) {
        if (doc.HasMember(k) && doc[k].IsString()) v = doc[k].GetString();
    };
    auto getInt = [&](const char* k, int& v) {
        if (doc.HasMember(k) && doc[k].IsInt()) v = doc[k].GetInt();
    };
    getStr("text_encoder_dir", mTeDir);
    getStr("text_encoder_model", mTeModelFile);
    getStr("transformer", mDiTFile);
    getStr("vae_decoder", mVaeFile);
    getStr("embeddings_bf16", mEmbeddingsFile);
    getStr("tokenizer", mTokenizerFile);
    getStr("latent_stats", mStatsFile);
    getStr("sigmas_file", mSigmasFile);
    getInt("text_len", mTextLen);
    getInt("drop_idx", mDropIdx);
    if (doc.HasMember("rope_index_inputs") && doc["rope_index_inputs"].IsBool())
        mRopeIndexInputs = doc["rope_index_inputs"].GetBool();
    auto absPath = [&](std::string& v) {
        if (!v.empty() && v[0] != '/') v = mModelPath + "/" + v;
    };
    absPath(mEmbeddingsFile);
    absPath(mTokenizerFile);
    absPath(mStatsFile);
}

bool QwenImage21Diffusion::readFileBytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    f.seekg(0, std::ios::end);
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    out.resize(bytes);
    f.read((char*)out.data(), bytes);
    return true;
}

bool QwenImage21Diffusion::load() {
    // Runtime hints (weight mmap) come from config.json via the Diffusion base ctor; they must
    // be in place before the module loads so static weights land in the mmap pool.
    if (!initRuntimeManagers(true /*gpuBufferMode*/, 0)) {
        return false;
    }
    mModules.resize(3);

    // TE and VAE each get a dedicated CPU runtime manager. Sharing one manager across the
    // three stages miscomputes the DiT image branch (text rows bit-exact, image rows garbage)
    // — same failure family as the fork's earlier shared-allocator useCount bug.
    auto makeCpuRuntime = [&]() -> std::shared_ptr<Executor::RuntimeManager> {
        ScheduleConfig cfg;
        cfg.type          = MNN_FORWARD_CPU;
        cfg.numThread     = mNumThreads;
        BackendConfig bc;
        bc.memory    = BackendConfig::Memory_Low;
        bc.precision = BackendConfig::Precision_Normal;
        cfg.backendConfig = &bc;
        std::shared_ptr<Executor::RuntimeManager> rt(Executor::RuntimeManager::createRuntimeManager(cfg));
        rt->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 0);
        if (!mWeightMmapPath.empty()) {
            rt->setExternalPath(mWeightMmapPath, Interpreter::EXTERNAL_WEIGHT_DIR);
            if (mUseCachedMmap) rt->setHint(Interpreter::USE_CACHED_MMAP, 1);
            if (mMmapFileSizeMB > 0) rt->setHint(Interpreter::MMAP_FILE_SIZE, mMmapFileSizeMB);
        }
        return rt;
    };
    std::shared_ptr<Executor::RuntimeManager> teRt = makeCpuRuntime();
    mVaeRt = makeCpuRuntime();

    std::string teModelPath = mModelPath + "/" + mTeDir + "/" + mTeModelFile;
    {
        Module::Config tec;
        tec.shapeMutable = true;
        mModules[0].reset(Module::load({"input_ids", "attention_mask", "position_ids", "logits_index",
                                        "deepstack_embeds"},
                                       {"logits", "hidden_states"}, teModelPath.c_str(), teRt, &tec));
    }
    if (!mModules[0]) {
        MNN_ERROR("[QwenImage21] Failed to load text encoder: %s\n", teModelPath.c_str());
        return false;
    }

    // memory_mode==0 (low): staged loading - TE only here; DiT loads after textEncode
    // released the TE (its ~8GB weights are never used again), VAE loads right before
    // decode. Peak memory becomes max(TE stage, DiT stage) instead of the sum of all three.
    if (mMemoryMode == 0) {
        MNN_PRINT("[QwenImage21] memory_mode=0: staged load (TE now, DiT/VAE on demand)\n");
        return true;
    }
    if (!ensureDiTLoaded() || !ensureVAELoaded()) return false;
    MNN_PRINT("[QwenImage21] modules loaded (TE / DiT / VAE)\n");
    return true;
}

bool QwenImage21Diffusion::ensureDiTLoaded() {
    if (mModules.size() < 3) mModules.resize(3);
    if (mModules[1]) return true;
    {
        std::string ditPath = mModelPath + "/" + mDiTFile;
        Module::Config dc;
        dc.shapeMutable = true;
        if (mRopeIndexInputs) {
            // dynamic-size DiT: rope indices + additive attention mask are inputs
            // (freqs tables are graph constants; the causal mask is built host-side)
            mModules[1].reset(Module::load({"hidden_states", "encoder_hidden_states", "timestep",
                                            "frame_idx", "height_idx", "width_idx",
                                            "attention_mask", "target_token_mask"},
                                           {"noise_pred"}, ditPath.c_str(), runtime_manager_, &dc));
        } else {
            mModules[1].reset(Module::load({"hidden_states", "encoder_hidden_states", "encoder_mask",
                                            "timestep", "img_mask"},
                                           {"noise_pred"}, ditPath.c_str(), runtime_manager_, &dc));
        }
        if (!mModules[1]) {
            MNN_ERROR("[QwenImage21] Failed to load transformer: %s\n", ditPath.c_str());
            return false;
        }
    }
    MNN_PRINT("[QwenImage21] stage-load: DiT ready\n");
    return true;
}

bool QwenImage21Diffusion::ensureVAELoaded() {
    if (mModules.size() < 3) mModules.resize(3);
    if (mModules[2]) return true;
    {
        std::string vaePath = mModelPath + "/" + mVaeFile;
        Module::Config vc;
        vc.shapeMutable = true;
        mModules[2].reset(Module::load({"latent_sample"}, {"sample"}, vaePath.c_str(), mVaeRt, &vc));
        if (!mModules[2]) {
            MNN_ERROR("[QwenImage21] Failed to load VAE decoder: %s\n", vaePath.c_str());
            return false;
        }
        mModules[2]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
    }
    MNN_PRINT("[QwenImage21] stage-load: VAE ready\n");
    return true;
}

bool QwenImage21Diffusion::textEncode(const std::string& prompt, std::vector<float>& enc,
                                      std::vector<int>& tmask) {
    AUTOTIME;
    std::vector<int> ids;
#ifdef MNN_BUILD_LLM
    if (!mTokenizer) {
        mTokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(mTokenizerFile));
        if (!mTokenizer) {
            MNN_ERROR("[QwenImage21] tokenizer load failed: %s\n", mTokenizerFile.c_str());
            return false;
        }
    }
    std::string templated = std::string("<|im_start|>system\n") + kSysPrompt + "<|im_end|>\n" +
                            "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    ids = mTokenizer->encode(templated);
#else
    MNN_ERROR("[QwenImage21] built without MNN_BUILD_LLM, no tokenizer\n");
    return false;
#endif
    const int seq = (int)ids.size();
    MNN_PRINT("[QwenImage21] prompt tokens: %d (drop first %d)\n", seq, mDropIdx);
    if (seq <= mDropIdx) {
        MNN_ERROR("[QwenImage21] prompt too short after template\n");
        return false;
    }

    // bf16 embedding table lookup (the exported TE consumes float embeds directly)
    std::vector<uint8_t> tableBytes;
    if (!readFileBytes(mEmbeddingsFile, tableBytes)) {
        MNN_ERROR("[QwenImage21] embeddings file missing: %s\n", mEmbeddingsFile.c_str());
        return false;
    }
    const uint16_t* table = (const uint16_t*)tableBytes.data();
    const size_t vocab    = tableBytes.size() / 2 / mHidden;
    std::vector<float> embeds((size_t)seq * mHidden);
    for (int i = 0; i < seq; i++) {
        int id = ids[i];
        if (id < 0 || (size_t)id >= vocab) {
            MNN_ERROR("[QwenImage21] bad token id %d\n", id);
            return false;
        }
        for (int j = 0; j < mHidden; j++) {
            unsigned u = ((unsigned)table[(size_t)id * mHidden + j]) << 16;
            float f;
            memcpy(&f, &u, 4);
            embeds[(size_t)i * mHidden + j] = f;
        }
    }

    // TE forward
    if (getenv("MNN_Q21_SKIPTETORWARD") != nullptr) {
        // bisect hook: skip the TE forward entirely (enc stays zero)
        MNN_PRINT("[QwenImage21] SKIPTETORWARD: enc=zeros, %d tokens\n", seq);
        enc.assign((size_t)mTextLen * mHidden, 0.0f);
        tmask.assign(mTextLen, 0);
        for (int i = 0; i < seq - mDropIdx && i < mTextLen; i++) tmask[i] = 1;
        return true;
    }
    VARP eVar = _Input({seq, 1, mHidden}, NCHW, halide_type_of<float>());
    memcpy(eVar->writeMap<float>(), embeds.data(), sizeof(float) * embeds.size());
    VARP mask = _Input({1, 1, seq, seq}, NCHW, halide_type_of<float>());
    {
        auto mp = mask->writeMap<float>();
        const float NEG = -3.4e38f;
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++) mp[i * seq + j] = (j > i) ? NEG : 0.0f;
    }
    VARP pos = _Input({3, seq}, NCHW, halide_type_of<int>());
    {
        auto p = pos->writeMap<int>();
        for (int a = 0; a < 3; a++)
            for (int i = 0; i < seq; i++) p[a * seq + i] = i;
    }
    VARP lix = _Input({1}, NCHW, halide_type_of<int>());
    lix->writeMap<int>()[0] = seq - 1;
    VARP ds = _Input({3, 1, mHidden}, NCHW, halide_type_of<float>());
    memset(ds->writeMap<float>(), 0, sizeof(float) * 3 * mHidden);

    auto teRes = mModules[0]->onForward({eVar, mask, pos, lix, ds});
    if (teRes.size() < 2 || teRes[1].get() == nullptr) {
        MNN_ERROR("[QwenImage21] TE forward failed\n");
        return false;
    }
    auto hsPtr = teRes[1]->readMap<float>();

    // drop system prefix rows, right-pad to text_len
    const int kept = seq - mDropIdx;
    enc.assign((size_t)mTextLen * mHidden, 0.0f);
    for (int i = 0; i < kept && i < mTextLen; i++)
        memcpy(&enc[(size_t)i * mHidden], &hsPtr[(size_t)(mDropIdx + i) * mHidden],
               sizeof(float) * mHidden);
    tmask.assign(mTextLen, 0);
    for (int i = 0; i < kept && i < mTextLen; i++) tmask[i] = 1;
    return true;
}

void QwenImage21Diffusion::computeSchedule(int steps, int imgSeqLen, std::vector<float>& sigmas) {
    // diffusers FlowMatchEulerDiscreteScheduler for Qwen-Image-2.1:
    // use_dynamic_shifting=true, time_shift_type="exponential", shift_terminal=0.02,
    // sigma_max=1.0, sigma_min=0.001, num_train_timesteps=1000.
    const float baseSeq = 256.f, maxSeq = 4096.f, baseShift = 0.5f, maxShift = 1.15f;
    const float m = (maxShift - baseShift) / (maxSeq - baseSeq);
    const float b = baseShift - m * baseSeq;
    const float mu       = imgSeqLen * m + b;
    const float expMu    = std::exp(mu);
    const float terminal = 0.02f;
    sigmas.clear();
    for (int i = 0; i < steps; i++) {
        float t = 1000.f - (1000.f - 1.f) * (float)i / (float)(steps > 1 ? steps - 1 : 1); // linspace(1000, 1)
        float sigma = t / 1000.f;
        sigma = expMu / (expMu + 1.f / sigma - 1.f);
        sigmas.push_back(sigma);
    }
    // stretch so the schedule terminates at shift_terminal (steps==1 has no interior
    // schedule to stretch; guard keeps the debug 1-step mode from dividing by zero)
    if (steps > 1) {
        const float scale = (1.f - sigmas.back()) / (1.f - terminal);
        for (auto& s : sigmas) s = 1.f - (1.f - s) / scale;
    }
    sigmas.push_back(0.0f);
}

bool QwenImage21Diffusion::run(const std::string prompt, const std::string outputPath, int iterNum,
                               int randomSeed, std::function<void(int)> progressCallback) {
    AUTOTIME;
    const char* injDir = getenv("MNN_Q21_INJECT");
    if (mMemoryMode == 0) {
        // staged mode: TE must exist unless inputs are injected; DiT/VAE load on demand
        if (!injDir && (mModules.size() < 3 || !mModules[0])) {
            MNN_ERROR("[QwenImage21] call load() first\n");
            return false;
        }
    } else if (mModules.size() < 3 || !mModules[0] || !mModules[1] || !mModules[2]) {
        MNN_ERROR("[QwenImage21] call load() first\n");
        return false;
    }
    int steps            = getenv("MNN_Q21_STEPS") ? atoi(getenv("MNN_Q21_STEPS"))
                                                   : (iterNum > 0 ? iterNum : 8);
    const int latentW    = (mImageWidth  > 0 ? mImageWidth  : 512) / 16;
    const int latentH    = (mImageHeight > 0 ? mImageHeight : 512) / 16;
    const int imgSeq     = latentW * latentH;
    const int slots      = imgSeq / 4;
    const int ch         = 64;

    // 1. text encode (or inject dumped inputs: MNN_Q21_INJECT=<dir> bisect hook)
    std::vector<float> enc;
    std::vector<int> tmask;
    std::vector<int> imask(mTextLen + slots, 0);
    std::vector<float> lat;
    if (injDir != nullptr) {
        std::vector<uint8_t> eb, mb, lb;
        readFileBytes(std::string(injDir) + "/enc.f32", eb);
        enc.resize(eb.size() / 4);
        memcpy(enc.data(), eb.data(), eb.size());
        readFileBytes(std::string(injDir) + "/in_masks.i32", mb);
        tmask.assign((int*)mb.data(), (int*)mb.data() + mTextLen);
        imask.assign((int*)mb.data() + mTextLen, (int*)mb.data() + mTextLen + slots);
        readFileBytes(std::string(injDir) + "/in_hs.f32", lb);
        lat.resize(lb.size() / 4);
        memcpy(lat.data(), lb.data(), lb.size());
        MNN_PRINT("[QwenImage21] INJECT mode: enc/lat/masks from %s\n", injDir);
        if (progressCallback) progressCallback(10);
    } else {
        if (!textEncode(prompt, enc, tmask)) return false;
        for (int i = mTextLen; i < mTextLen + slots; i++) imask[i] = 1;
        if (progressCallback) progressCallback(5);
    }

    // Debug bisect: MNN_Q21_DUMP=<dir> writes enc/lat0/npN for comparison with pipeline_io
    const char* dbgDir = getenv("MNN_Q21_DUMP");
    auto dumpF32 = [&](const std::string& name, const float* p, size_t n) {
        if (dbgDir == nullptr) return;
        std::ofstream f(std::string(dbgDir) + "/" + name, std::ios::binary);
        f.write((const char*)p, sizeof(float) * n);
        MNN_PRINT("[QwenImage21] dumped %s (%zu floats)\n", name.c_str(), n);
    };
    dumpF32("enc.f32", enc.data(), enc.size());

    // staged low-memory mode: TE weights are dead after textEncode (enc[] owns the output);
    // release them before the DiT loads so peak memory = max(TE, DiT), not the sum
    if (mMemoryMode == 0) {
        mModules[0].reset();
        if (!ensureDiTLoaded()) return false;
        MNN_PRINT("[QwenImage21] stage-load: TE released, DiT active\n");
    }

    // 2. schedule + noise (skipped in INJECT mode: lat already loaded)
    std::vector<float> sigmas;
    if (!mSigmasFile.empty()) {
        // checkpoint-embedded schedule (Turbo): sigmas dumped from the PT pipeline
        std::string sp = mSigmasFile;
        if (sp[0] != '/') sp = mModelPath + "/" + sp;
        std::vector<uint8_t> sb;
        if (!readFileBytes(sp, sb) || sb.size() < sizeof(float) * (steps + 1)) {
            MNN_ERROR("[QwenImage21] sigmas_file missing/too small: %s (%zu bytes)\n", sp.c_str(),
                      sb.size());
            return false;
        }
        const float* sf = (const float*)sb.data();
        size_t n = sb.size() / sizeof(float);
        sigmas.assign(sf, sf + n);
        MNN_PRINT("[QwenImage21] schedule from %s: %zu sigmas (steps=%d)\n", sp.c_str(), n, steps);
        if ((int)n < steps + 1) steps = (int)n - 1;
    } else {
        computeSchedule(steps, imgSeq, sigmas);
    }
    if (injDir == nullptr) {
        // Initial noise follows the reference chain bit-for-bit-ish: torch CPU
        // randn([1,1,64,h,w], manual_seed) (MT19937 + 24-bit uniform + normal_fill_16
        // Box-Muller pairs), packed to (B, H*W, C). The base-class PhiloxRNG targets the
        // CUDA/llm convention and does NOT match torch-CPU dumps (cosine 0.0006 vs 0.9999986).
        std::vector<float> noise((size_t)ch * latentW * latentH);
        torchCpuRandn(noise.data(), noise.size(), (uint64_t)randomSeed);
        lat.resize((size_t)imgSeq * ch);
        for (int c = 0; c < ch; c++)
            for (int y = 0; y < latentH; y++)
                for (int x = 0; x < latentW; x++)
                    lat[((size_t)y * latentW + x) * ch + c] =
                        noise[((size_t)c * latentH + y) * latentW + x];
        if (progressCallback) progressCallback(10);
        dumpF32("lat0.f32", lat.data(), lat.size());
    }

    // IMG2IMG (SDEdit): init from a reference image's normalized latents at
    // sigmas[startStep], then run the tail of the schedule.
    const char* initPath = getenv("MNN_Q21_INIT_LATENTS");
    int startStep = getenv("MNN_Q21_START_STEP") ? atoi(getenv("MNN_Q21_START_STEP")) : 0;
    if (injDir == nullptr && initPath != nullptr) {
        std::vector<uint8_t> ib;
        if (!readFileBytes(initPath, ib) || ib.size() < lat.size() * 4) {
            MNN_ERROR("[QwenImage21] init latents short: %s\n", initPath);
            return false;
        }
        std::vector<float> x0((float*)ib.data(), (float*)ib.data() + lat.size());
        std::vector<float> inoise((size_t)ch * latentW * latentH);
        torchCpuRandn(inoise.data(), inoise.size(), (uint64_t)randomSeed);
        if (startStep < 0 || startStep >= steps) startStep = 0;
        const float sk = sigmas[startStep];
        for (int c = 0; c < ch; c++)
            for (int y = 0; y < latentH; y++)
                for (int x = 0; x < latentW; x++) {
                    size_t li = ((size_t)y * latentW + x) * ch + c;
                    size_t ni = ((size_t)c * latentH + y) * latentW + x;
                    lat[li] = (1.0f - sk) * x0[li] + sk * inoise[ni];
                }
        MNN_PRINT("[QwenImage21] img2img init: startStep=%d sigma=%.4f\n", startStep, sk);
    }

    // 3. latent de-normalization stats (64 mean + 64 std)
    std::vector<uint8_t> statsBytes;
    {
        std::string statsPath = mStatsFile;
        if (statsPath.find('/') == std::string::npos) statsPath = mModelPath + "/" + mStatsFile;
        if (!readFileBytes(statsPath, statsBytes) || statsBytes.size() < sizeof(float) * 2 * ch) {
            MNN_ERROR("[QwenImage21] latent stats missing: %s\n", statsPath.c_str());
            return false;
        }
    }
    const float* stats = (const float*)statsBytes.data();

    // 4. DiT sampling
    VARP amV, tmV;
    if (mRopeIndexInputs) {
        // additive attention mask [1,1,S,S] built host-side (the exported graph
        // takes it as an input — MNN cannot execute the dynamic shape ops an
        // in-graph mask would need): text rows causal over valid text keys,
        // image rows attend to everything.
        const int seq = mTextLen + imgSeq;
        std::vector<float> amask((size_t)seq * seq);
        for (int q = 0; q < seq; q++) {
            for (int k = 0; k < seq; k++) {
                float v = 0.f;
                if (q < mTextLen && k > q) v -= 1e9f;
                // padded text keys are penalized for ALL rows (PT keys=[valid,ones])
                if (k < mTextLen && tmask[k] == 0) v -= 1e9f;
                amask[(size_t)q * seq + k] = v;
            }
        }
        amV = _Input({1, 1, seq, seq}, NCHW, halide_type_of<float>());
        memcpy(amV->writeMap<float>(), amask.data(), sizeof(float) * amask.size());
        // target_token_mask [1,seq]: 1.0 on target-image rows, 0.0 on the text
        // prefix (host-built input, same philosophy as the attention mask)
        tmV = _Input({1, seq}, NCHW, halide_type_of<float>());
        {
            float* tp = tmV->writeMap<float>();
            for (int q = 0; q < seq; q++) tp[q] = (q >= mTextLen) ? 1.0f : 0.0f;
        }
    }
    int s0 = (initPath != nullptr) ? startStep : 0; /* IMG2IMG */
    for (int s = s0; s < steps; s++) {
        VARP hs = _Input({1, imgSeq, ch}, NCHW, halide_type_of<float>());
        memcpy(hs->writeMap<float>(), lat.data(), sizeof(float) * lat.size());
        VARP encV = _Input({1, mTextLen, mHidden}, NCHW, halide_type_of<float>());
        memcpy(encV->writeMap<float>(), enc.data(), sizeof(float) * enc.size());
        VARP tsV = _Input({1}, NCHW, halide_type_of<float>());
        // the exported DiT consumes SIGMA (0..1), i.e. timesteps.f32/1000 — same as
        // test_pipeline's tsteps[s]/1000.0f. Feeding sigma*1000 saturates the timestep
        // embedding and corrupts the image branch (text rows stay bit-exact).
        tsV->writeMap<float>()[0] = sigmas[s];
        VARP out;
        if (mRopeIndexInputs) {
            // dynamic-size DiT: rope indices [1, textLen+imgSeq] int64 (pre-mapped
            // positive; the graph gathers the freqs tables directly)
            std::vector<int32_t> fi, hi, wi; /* ROIPE-I32 */
            buildRopeIndices(mTextLen, latentH, latentW, &fi, &hi, &wi);
            const size_t total = fi.size();
            VARP fV = _Input({1, (int)total}, NCHW, halide_type_of<int32_t>());
            VARP hV = _Input({1, (int)total}, NCHW, halide_type_of<int32_t>());
            VARP wV = _Input({1, (int)total}, NCHW, halide_type_of<int32_t>());
            memcpy(fV->writeMap<int32_t>(), fi.data(), sizeof(int32_t) * total);
            memcpy(hV->writeMap<int32_t>(), hi.data(), sizeof(int32_t) * total);
            memcpy(wV->writeMap<int32_t>(), wi.data(), sizeof(int32_t) * total);
            out = mModules[1]->onForward({hs, encV, tsV, fV, hV, wV, amV, tmV})[0];
        } else {
            VARP emV = _Input({1, mTextLen}, NCHW, halide_type_of<int>());
            memcpy(emV->writeMap<int>(), tmask.data(), sizeof(int) * mTextLen);
            VARP imV = _Input({1, mTextLen + slots}, NCHW, halide_type_of<int>());
            memcpy(imV->writeMap<int>(), imask.data(), sizeof(int) * (mTextLen + slots));
            out = mModules[1]->onForward({hs, encV, emV, tsV, imV})[0];
        }
        if (out.get() == nullptr) {
            MNN_ERROR("[QwenImage21] DiT step %d failed\n", s);
            return false;
        }
        auto info = out->getInfo();
        size_t elems = 1;
        for (int d : info->dim) elems *= d;
        auto ptr  = out->readMap<float>();
        const size_t rowElems = (size_t)imgSeq * ch;
        if (dbgDir != nullptr && s == 0) {
            std::string dims;
            for (int d : info->dim) dims += " " + std::to_string(d);
            MNN_PRINT("[QwenImage21] DiT out dims:%s (elems %zu, rowElems %zu)\n", dims.c_str(),
                      elems, rowElems);
            dumpF32("np0_full.f32", ptr, elems);
        }
        dumpF32("np" + std::to_string(s) + ".f32", ptr + elems - rowElems, rowElems);
        const float dt = sigmas[s + 1] - sigmas[s];
        const float* np = ptr + elems - rowElems;
        for (size_t i = 0; i < lat.size(); i++) lat[i] += dt * np[i];
        MNN_PRINT("[QwenImage21] step %d/%d (sigma %.4f)\n", s + 1, steps, sigmas[s]);
        if (progressCallback) progressCallback(10 + (int)(85.0f * (s + 1) / steps));
    }

    // 5. unpack [imgSeq,64] -> [1,64,1,h,w] + de-normalize
    std::vector<float> lat5d((size_t)ch * latentH * latentW);
    for (int c = 0; c < ch; c++)
        for (int p = 0; p < imgSeq; p++)
            lat5d[((size_t)c * latentH + p / latentW) * latentW + p % latentW] =
                lat[(size_t)p * ch + c] * stats[ch + c] + stats[c];

    // 6. VAE decode + save
    if (mMemoryMode == 0) {
        mModules[1].reset(); // DiT is done; free its weights before the decoder loads
        if (!ensureVAELoaded()) return false;
        MNN_PRINT("[QwenImage21] stage-load: DiT released, VAE active\n");
    }
    VARP lv = _Input({1, ch, 1, latentH, latentW}, NCHW, halide_type_of<float>());
    memcpy(lv->writeMap<float>(), lat5d.data(), sizeof(float) * lat5d.size());
    auto vout = mModules[2]->onForward({lv});
    if (vout.empty() || vout[0].get() == nullptr) {
        MNN_ERROR("[QwenImage21] VAE forward failed\n");
        return false;
    }
    // dyn VAE output is 5D [1,4,1,H,W] (video-style latent); squeeze the T axis, map to
    // [0,1], composite RGBA over white in-graph (the calibrated grid_compose metric), and
    // feed imwrite in BGR order (it swaps to RGB on write).
    auto img = _Squeeze(_Convert(vout[0], NCHW), {2}); // [1,4,H,W]
    img = _Relu6(img * _Const(0.5f) + _Const(0.5f), 0.f, 1.f);
    auto parts = _Split(img, {1, 1, 1, 1}, 1);
    auto comp = _Concat({parts[2] * parts[3] + (_Const(1.f) - parts[3]),
                         parts[1] * parts[3] + (_Const(1.f) - parts[3]),
                         parts[0] * parts[3] + (_Const(1.f) - parts[3])}, 1);
    img = _Squeeze(_Transpose(comp, {0, 2, 3, 1}));
    img = _Cast(_Round(img * _Const(255.f)), halide_type_of<uint8_t>());
    img.fix(VARP::CONSTANT);
    bool ok = CV::imwrite(outputPath, img);
    if (ok)
        MNN_PRINT("[QwenImage21] saved %s\n", outputPath.c_str());
    else
        MNN_ERROR("[QwenImage21] imwrite failed: %s\n", outputPath.c_str());
    if (progressCallback) progressCallback(100);
    return ok;
}

bool QwenImage21Diffusion::run(const VARP input_embeds, const std::string& mode,
                               const std::string& inputImagePath, const std::string& outputImagePath,
                               int width, int height, int iterNum, int randomSeed, bool use_cfg,
                               float cfg_scale, std::function<void(int)> progressCallback) {
    (void)input_embeds;
    (void)mode;
    (void)inputImagePath;
    (void)use_cfg;
    (void)cfg_scale;
    mImageWidth  = width > 0 ? width : mImageWidth;
    mImageHeight = height > 0 ? height : mImageHeight;
    return run("", outputImagePath, iterNum, randomSeed, progressCallback);
}

} // namespace DIFFUSION
} // namespace MNN
