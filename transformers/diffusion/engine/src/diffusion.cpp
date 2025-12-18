#include <random>
#include <fstream>
#include <chrono>
#include "diffusion/diffusion.hpp"
#include "tokenizer.hpp"
#include "scheduler.hpp"
#ifdef MNN_BUILD_LLM
#include "llm/tokenizer.hpp"
#endif
#include <rapidjson/document.h>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>
#include <sstream>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdlib>

#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif

//#define MNN_DUMP_DATA
//#define MNN_DIFFUSION_DEBUG_STATS  // Disabled: verbose debug output

using namespace CV;

namespace MNN {
namespace DIFFUSION {

static inline bool diffusion_debug_enabled() {
    static int enabled = -1;
    if (enabled >= 0) {
        return enabled != 0;
    }
    const char* env = ::getenv("MNN_DIFFUSION_DEBUG");
    if (env && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0')) {
        enabled = 1;
    } else {
        enabled = 0;
    }
    return enabled != 0;
}

template <typename T>
static void print_array_stats(const std::string& name, const T* data, size_t size, size_t max_elems = 8) {
    if (!diffusion_debug_enabled()) {
        return;
    }
    if (!data || size == 0) {
        MNN_PRINT("[STAT] %s: empty, size=0\n", name.c_str());
        return;
    }
    size_t nanCount = 0;
    size_t infCount = 0;
    size_t finiteCount = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    long double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float v = static_cast<float>(data[i]);
        if (std::isnan(v)) {
            nanCount += 1;
            continue;
        }
        if (std::isinf(v)) {
            infCount += 1;
            continue;
        }
        finiteCount += 1;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += v;
    }
    float mean_v = finiteCount > 0 ? static_cast<float>(sum / static_cast<long double>(finiteCount)) : std::numeric_limits<float>::quiet_NaN();
    if (finiteCount > 0) {
        MNN_PRINT("[STAT] %s: size=%zu, finite=%zu, nan=%zu, inf=%zu, min=%f, max=%f, mean=%f\n",
                  name.c_str(), size, finiteCount, nanCount, infCount, min_v, max_v, mean_v);
    } else {
        MNN_PRINT("[STAT] %s: size=%zu, finite=%zu, nan=%zu, inf=%zu\n",
                  name.c_str(), size, finiteCount, nanCount, infCount);
    }

    size_t k = std::min(max_elems, size);
    // Build first_values string
    std::ostringstream oss;
    oss << "[STAT] " << name << ": first_values[" << k << "] = [";
    for (size_t i = 0; i < k; ++i) {
        if (i) oss << ", ";
        oss << std::fixed << std::setprecision(6) << static_cast<float>(data[i]);
    }
    oss << "]";
    MNN_PRINT("%s\n", oss.str().c_str());
}

static void print_var_stats(const std::string& name, VARP var, size_t max_elems = 8) {
    if (!diffusion_debug_enabled()) {
        return;
    }
    if (!var.get()) {
        MNN_PRINT("[STAT] %s: null VARP\n", name.c_str());
        return;
    }
    auto info = var->getInfo();
    if (!info) {
        MNN_PRINT("[STAT] %s: no info\n", name.c_str());
        return;
    }
    // Build shape string
    std::ostringstream oss;
    oss << "[STAT] " << name << ": shape=(";
    for (int i = 0; i < info->dim.size(); ++i) {
        if (i) oss << ", ";
        oss << info->dim[i];
    }
    oss << ")";
    MNN_PRINT("%s\n", oss.str().c_str());

    VARP vf = var;
    if (!(info->type.code == halide_type_float && info->type.bits == 32)) {
        vf = _Cast(var, halide_type_of<float>());
        vf.fix(VARP::CONSTANT);
        info = vf->getInfo();
        if (!info) {
            MNN_PRINT("[STAT] %s: cast-to-float failed (no info)\n", name.c_str());
            return;
        }
    }
    size_t size = static_cast<size_t>(info->size);
    const float* data = vf->readMap<float>();
    if (!data || size == 0) {
        MNN_PRINT("[STAT] %s: empty or unreadable data\n", name.c_str());
        return;
    }
    print_array_stats(name, data, size, max_elems);
}

static inline const char* dim_format_to_string(Dimensionformat format) {
    switch (format) {
        case NHWC:
            return "NHWC";
        case NC4HW4:
            return "NC4HW4";
        case NCHW:
            return "NCHW";
        default:
            return "UNKNOWN";
    }
}

static inline void log_var_shape_mnn(const char* name, VARP var) {
    if (nullptr == name) {
        return;
    }
    if (!var.get()) {
        MNN_PRINT("[SHAPE] %s: null\n", name);
        return;
    }
    auto info = var->getInfo();
    if (!info) {
        MNN_PRINT("[SHAPE] %s: no info\n", name);
        return;
    }
    char line[1024];
    int offset = snprintf(line, sizeof(line), "[SHAPE] %s: order=%s, dims=(",
                          name, dim_format_to_string(info->order));
    if (offset < 0) {
        MNN_PRINT("[SHAPE] %s: order=%s, dims=(...)\n", name, dim_format_to_string(info->order));
        return;
    }
    for (int i = 0; i < (int)info->dim.size(); ++i) {
        if (offset >= (int)sizeof(line) - 4) {
            break;
        }
        if (i) {
            offset += snprintf(line + offset, sizeof(line) - offset, ", ");
        }
        offset += snprintf(line + offset, sizeof(line) - offset, "%d", info->dim[i]);
    }
    if (offset < (int)sizeof(line) - 2) {
        snprintf(line + offset, sizeof(line) - offset, ")");
    } else {
        line[sizeof(line) - 2] = ')';
        line[sizeof(line) - 1] = '\0';
    }
    MNN_PRINT("%s\n", line);
}

static inline void log_module_io_mnn(const char* name, const std::shared_ptr<Module>& module) {
    if (nullptr == name) {
        return;
    }
    if (!module) {
        MNN_PRINT("[MODEL] %s: null\n", name);
        return;
    }
    auto info = module->getInfo();
    if (!info) {
        MNN_PRINT("[MODEL] %s: no info\n", name);
        return;
    }
    MNN_PRINT("[MODEL] %s: defaultFormat=%s, inputCount=%d, outputCount=%d\n",
              name,
              dim_format_to_string(info->defaultFormat),
              (int)info->inputs.size(),
              (int)info->outputNames.size());
    for (int i = 0; i < (int)info->inputs.size(); ++i) {
        const auto& in = info->inputs[i];
        char line[1024];
        const char* inName = (i < (int)info->inputNames.size()) ? info->inputNames[i].c_str() : "";
        int offset = 0;
        if (inName[0] != '\0') {
            offset = snprintf(line, sizeof(line), "[MODEL] %s input[%d] name=%s: order=%s, dims=(",
                              name, i, inName, dim_format_to_string(in.order));
        } else {
            offset = snprintf(line, sizeof(line), "[MODEL] %s input[%d]: order=%s, dims=(",
                              name, i, dim_format_to_string(in.order));
        }
        if (offset < 0) {
            MNN_PRINT("[MODEL] %s input[%d]: order=%s\n", name, i, dim_format_to_string(in.order));
            continue;
        }
        for (int d = 0; d < (int)in.dim.size(); ++d) {
            if (offset >= (int)sizeof(line) - 4) {
                break;
            }
            if (d) {
                offset += snprintf(line + offset, sizeof(line) - offset, ", ");
            }
            offset += snprintf(line + offset, sizeof(line) - offset, "%d", in.dim[d]);
        }
        if (offset < (int)sizeof(line) - 2) {
            snprintf(line + offset, sizeof(line) - offset, ")");
        } else {
            line[sizeof(line) - 2] = ')';
            line[sizeof(line) - 1] = '\0';
        }
        MNN_PRINT("%s\n", line);
    }
}

#ifdef MNN_BUILD_LLM
class LlmTokenizerWrapper : public Tokenizer {
public:
    // Internal helper that can access protected load/encode members.
    class DiffusionHuggingfaceTokenizer : public MNN::Transformer::HuggingfaceTokenizer {
    public:
        bool loadFromFile(const std::string& filename) {
            std::ifstream file(filename.c_str());
            if (!file.is_open()) {
                return false;
            }
            load_special(file);
            return load_vocab(file);
        }

        std::vector<int> encodeWithSpecial(const std::string& str) {
            // Important: call the public Tokenizer::encode(str) which performs atomic special-token matching.
            return MNN::Transformer::Tokenizer::encode(str);
        }
    };

    LlmTokenizerWrapper() = default;
    virtual ~LlmTokenizerWrapper() = default;

    virtual bool load(const std::string& filePath) override {
        std::string tokPath = filePath + "/tokenizer.txt";
        std::ifstream check(tokPath.c_str());
        if (!check.good()) {
            MNN_PRINT("Error: tokenizer.txt not found at %s\n", tokPath.c_str());
            return false;
        }
        check.close();

        std::shared_ptr<DiffusionHuggingfaceTokenizer> impl(new DiffusionHuggingfaceTokenizer);
        if (!impl->loadFromFile(tokPath)) {
            MNN_PRINT("Error: failed to load tokenizer vocab from %s\n", tokPath.c_str());
            return false;
        }
        mTokenizer = impl;
        return true;
    }

    virtual std::vector<int> encode(const std::string& sentence, int maxlen = 0) override {
        std::vector<int> packed;
        auto impl = std::static_pointer_cast<DiffusionHuggingfaceTokenizer>(mTokenizer);
        if (!impl) {
            return packed;
        }

        std::vector<int> baseIds = impl->encodeWithSpecial(sentence);

        if (maxlen <= 0) {
            maxlen = (int)baseIds.size();
        }

        packed.assign(maxlen * 2, 0);
        int n = std::min((int)baseIds.size(), maxlen);
        for (int i = 0; i < n; ++i) {
            packed[i] = baseIds[i];
            packed[maxlen + i] = 1;
        }
        return packed;
    }

private:
    std::shared_ptr<DiffusionHuggingfaceTokenizer> mTokenizer;
};
#endif

void display_progress(int cur, int total){
    putchar('\r');
    MNN_PRINT("[");
    for (int i = 0; i < cur; i++) putchar('#');
    for (int i = 0; i < total - cur; i++) putchar('-');
    MNN_PRINT("]");
    fprintf(stdout, "  [%3d%%]", cur * 100 / total);
    if (cur == total) putchar('\n');
    fflush(stdout);
}

Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode) {
    Diffusion* diffusion = new Diffusion(modelPath, modelType, backendType, memoryMode);
    return diffusion;
}

Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize) {
    Diffusion* diffusion = new Diffusion(modelPath, modelType, backendType, memoryMode, imageSize);
    return diffusion;
}

Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU) {
    Diffusion* diffusion = new Diffusion(modelPath, modelType, backendType, memoryMode, imageSize, textEncoderOnCPU);
    return diffusion;
}

Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads) {
    Diffusion* diffusion = new Diffusion(modelPath, modelType, backendType, memoryMode, imageSize, textEncoderOnCPU, gpuMemoryMode, precisionMode, numThreads);
    return diffusion;
}

Diffusion::Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode) :
Diffusion(modelPath, modelType, backendType, memoryMode, 0, true, GPU_MEMORY_AUTO, PRECISION_AUTO) {
}

Diffusion::Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize) :
Diffusion(modelPath, modelType, backendType, memoryMode, imageSize, true, GPU_MEMORY_AUTO, PRECISION_AUTO) {
}

Diffusion::Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU) :
Diffusion(modelPath, modelType, backendType, memoryMode, imageSize, textEncoderOnCPU, GPU_MEMORY_AUTO, PRECISION_AUTO) {
}

Diffusion::Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads) :
mModelPath(modelPath), mModelType(modelType), mBackendType(backendType), mMemoryMode(memoryMode), mImageSize(imageSize), mTextEncoderOnCPU(textEncoderOnCPU), mGpuMemoryMode(gpuMemoryMode), mPrecisionMode(precisionMode), mNumThreads(numThreads) {
    if (modelType == STABLE_DIFFUSION_1_5) {
        mMaxTextLen = 77;
    } else if (modelType == STABLE_DIFFUSION_TAIYI_CHINESE) {
        mMaxTextLen = 512;
    } else if (modelType == STABLE_DIFFUSION_ZIMAGE) {
        // ZImage pipeline typically uses a longer context; align with Python side (e.g. max_length=128).
        mMaxTextLen = 128;
    }
    if (modelType == STABLE_DIFFUSION_ZIMAGE) {
        // Default FlowMatch Euler config; can be overridden by scheduler_config.json.
        mTrainTimestepsNum = 1000;
        mFlowShift = 3.0f;
        mUseDynamicShifting = false;
        std::string schedPath1 = mModelPath + "/scheduler/scheduler_config.json";
        std::string schedPath2 = mModelPath + "/scheduler_config.json";
        std::string schedPath;
        std::ifstream sfile;
        sfile.open(schedPath1.c_str());
        if (sfile.good()) {
            schedPath = schedPath1;
        } else {
            sfile.close();
            sfile.open(schedPath2.c_str());
            if (sfile.good()) {
                schedPath = schedPath2;
            }
        }
        if (!schedPath.empty()) {
            std::ostringstream oss;
            oss << sfile.rdbuf();
            sfile.close();
            std::string jsonStr = oss.str();
            rapidjson::Document doc;
            doc.Parse(jsonStr.c_str());
            if (!doc.HasParseError() && doc.IsObject()) {
                if (doc.HasMember("num_train_timesteps") && doc["num_train_timesteps"].IsInt()) {
                    mTrainTimestepsNum = doc["num_train_timesteps"].GetInt();
                }
                if (doc.HasMember("shift") && (doc["shift"].IsFloat() || doc["shift"].IsDouble())) {
                    mFlowShift = static_cast<float>(doc["shift"].GetDouble());
                }
                if (doc.HasMember("use_dynamic_shifting") && doc["use_dynamic_shifting"].IsBool()) {
                    mUseDynamicShifting = doc["use_dynamic_shifting"].GetBool();
                }
            } else {
                MNN_PRINT("Warning: failed to parse %s, using default FlowMatch scheduler config.\n", schedPath.c_str());
            }
        } else {
            MNN_PRINT("Warning: scheduler_config.json not found at %s or %s, using default FlowMatch scheduler config.\n", schedPath1.c_str(), schedPath2.c_str());
        }
    } else {
        // compute timesteps alphas for SD1.5/Taiyi (PNDM + PLMS)
        std::unique_ptr<Scheduler> scheduler;
        scheduler.reset(new PNDMScheduler);
        mAlphas = scheduler->get_alphas();
    }

    if (modelType == STABLE_DIFFUSION_ZIMAGE) {
        // Keep legacy behavior when user does not override image size.
        // This must stay consistent with the previous implementation (no imageSize option).
        if (mImageSize <= 0) {
            mLatentC = 16;
            mLatentH = 128;
            mLatentW = 128;
            MNN_PRINT("[ZIMAGE] latentScale=8, imageSize(user=default), latent=(1,%d,%d,%d)\n", mLatentC, mLatentH, mLatentW);
        } else {
            int normalizedSize = mImageSize;
            if (normalizedSize != 512 && normalizedSize != 640 && normalizedSize != 768 && normalizedSize != 896 && normalizedSize != 1024) {
                normalizedSize = 1024;
            }
            mImageSize = normalizedSize;
            mLatentC = 16;
            mLatentH = normalizedSize / 8;
            mLatentW = normalizedSize / 8;
            MNN_PRINT("[ZIMAGE] latentScale=8, imageSize(user=%d), latent=(1,%d,%d,%d)\n", mImageSize, mLatentC, mLatentH, mLatentW);
        }
    }
}

Diffusion::~Diffusion() {
    mModules.clear();
    runtime_manager_.reset();
}

bool Diffusion::load() {
    AUTOTIME;
    ScheduleConfig config;
    BackendConfig backendConfig;
    config.type = mBackendType;
    if(config.type == MNN_FORWARD_CPU) {
        config.numThread = mNumThreads;
    } else if(config.type == MNN_FORWARD_OPENCL) {
        // Configure GPU memory mode based on user setting or auto-detect
        int gpuMode = MNN_GPU_TUNING_FAST;
        if (mGpuMemoryMode == GPU_MEMORY_BUFFER) {
            gpuMode |= MNN_GPU_MEMORY_BUFFER;
        } else if (mGpuMemoryMode == GPU_MEMORY_IMAGE) {
            gpuMode |= MNN_GPU_MEMORY_IMAGE;
        } else {
            // AUTO: Use BUFFER mode for all models
            // SD1.5 contains FmhaV2 operator which is only supported in BUFFER mode
            // ZImage also works fine with BUFFER mode
            gpuMode |= MNN_GPU_MEMORY_BUFFER;
        }
        config.mode = gpuMode;
    } else {
        config.numThread = 1;
    }
    backendConfig.memory = BackendConfig::Memory_Low;
    
    // Configure precision based on user setting or auto-detect
    if (mPrecisionMode == PRECISION_LOW) {
        backendConfig.precision = BackendConfig::Precision_Low;
    } else if (mPrecisionMode == PRECISION_NORMAL) {
        backendConfig.precision = BackendConfig::Precision_Normal;
    } else if (mPrecisionMode == PRECISION_HIGH) {
        backendConfig.precision = BackendConfig::Precision_High;
    } else {
        // AUTO: Select precision based on model type
        // ZImage uses -inf constants in attention mask, requires FP32 on Adreno GPU
        // SD1.5/Taiyi works fine with FP16
        // See SPEC.md "Diffusion OpenCL 后端配置" for details
        if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
            // ZImage requires FP32 for OpenCL (Adreno GPU -inf handling)
            if (config.type == MNN_FORWARD_OPENCL) {
                backendConfig.precision = BackendConfig::Precision_High;  // FP32
            } else {
                backendConfig.precision = BackendConfig::Precision_Normal;  // FP32 for CPU too
            }
        } else {
            // SD1.5/Taiyi: FP16 works fine, better performance
            backendConfig.precision = BackendConfig::Precision_Low;  // FP16
        }
    }
    MNN_PRINT("[DIFFUSION] Backend: type=%d, gpuMemMode=%d, precision=%s (userPrecision=%d)\n",
              config.type, mGpuMemoryMode,
              backendConfig.precision == BackendConfig::Precision_High ? "High(FP32)" :
              (backendConfig.precision == BackendConfig::Precision_Normal ? "Normal(FP32)" : "Low(FP16)"),
              mPrecisionMode);
    config.backendConfig = &backendConfig;

    auto exe = ExecutorScope::Current();
    exe->lazyEval = false;
    exe->setGlobalExecutorConfig(config.type, backendConfig, config.numThread);

    Module::Config module_config;
    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        module_config.shapeMutable = true;
    } else {
        module_config.shapeMutable = false;
    }
    // module_config.rearrange = true;
    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));

    if (config.type == MNN_FORWARD_OPENCL) {
        const char* cacheFileName = ".tempcache";
        runtime_manager_->setCache(cacheFileName);
    }
    // need to consider memory
    if(mMemoryMode == 0) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
    } else if(mMemoryMode == 2) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 1);
    }
    if(config.type == MNN_FORWARD_CPU) {
        if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
            runtime_manager_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 0);
        } else {
            runtime_manager_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 2);
        }
    }

    // Create a separate CPU RuntimeManager for text_encoder to avoid OpenCL CL_INVALID_BUFFER_SIZE error
    // when loading large quantized weights that exceed GPU's CL_DEVICE_MAX_MEM_ALLOC_SIZE limit
    if (mTextEncoderOnCPU && (config.type == MNN_FORWARD_OPENCL || config.type == MNN_FORWARD_VULKAN)) {
        ScheduleConfig cpuConfig;
        BackendConfig cpuBackendConfig;
        cpuConfig.type = MNN_FORWARD_CPU;
        cpuConfig.numThread = mNumThreads;
        cpuBackendConfig.memory = BackendConfig::Memory_Low;
        cpuBackendConfig.precision = BackendConfig::Precision_Normal;
        cpuConfig.backendConfig = &cpuBackendConfig;
        runtime_manager_cpu_.reset(Executor::RuntimeManager::createRuntimeManager(cpuConfig));
        if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
            runtime_manager_cpu_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 0);
        } else {
            runtime_manager_cpu_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 2);
        }
        MNN_PRINT("[DIFFUSION] text_encoder forced on CPU (mTextEncoderOnCPU=true)\n");
    } else {
        MNN_PRINT("[DIFFUSION] text_encoder on same backend as UNet (mTextEncoderOnCPU=false)\n");
    }
    mLatentVar = _Input({1, mLatentC, mLatentH, mLatentW}, NCHW, halide_type_of<float>());
    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        mPromptVar = _Input({1, mMaxTextLen}, NCHW, halide_type_of<int>());
        mAttentionMaskVar = _Input({1, mMaxTextLen}, NCHW, halide_type_of<int>());
        // ZImage UNet expects float32 timesteps, consistent with Python MNN tests
        mTimestepVar = _Input({1}, NCHW, halide_type_of<float>());
    } else {
        mPromptVar = _Input({2, mMaxTextLen}, NCHW, halide_type_of<int>());
        mTimestepVar = _Input({1}, NCHW, halide_type_of<int>());
    }
    mLatentVar->writeMap<float>();
    mPromptVar->writeMap<int>();
    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        mAttentionMaskVar->writeMap<int>();
    }
    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        mTimestepVar->writeMap<float>();
    } else {
        mTimestepVar->writeMap<int>();
    }

    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        // ZImage uses single batch UNet without classifier-free guidance.
        mSampleVar = mLatentVar;
    } else {
        mSampleVar = _Concat({mLatentVar, mLatentVar}, 0);
    }

    if(mMemoryMode > 0) {
        MNN_PRINT("First time initilizing may cost a few seconds to create cachefile, please wait ...\n");
    }

    VARP text_embeddings;
    mModules.resize(3);
    // load text_encoder model
    {
        std::string model_path = mModelPath + "/text_encoder.mnn";
        MNN_PRINT("Load %s\n", model_path.c_str());
        std::vector<std::string> textInputs;
        std::vector<std::string> textOutputs;
        if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
            // ZImage text encoder expects input_ids + attention_mask and outputs last_hidden_state only.
            textInputs = {"input_ids", "attention_mask"};
            textOutputs = {"last_hidden_state"};
        } else {
            textInputs = {"input_ids"};
            textOutputs = {"last_hidden_state", "pooler_output"};
        }
        // Use CPU runtime for text_encoder when GPU backend is selected, to avoid CL_INVALID_BUFFER_SIZE error
        auto& te_runtime = runtime_manager_cpu_ ? runtime_manager_cpu_ : runtime_manager_;
        mModules[0].reset(Module::load(textInputs, textOutputs, model_path.c_str(), te_runtime, &module_config));
    }
    // load unet model
    {
        std::string model_path = mModelPath + "/unet.mnn";
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[1].reset(Module::load(
                                       {"sample", "timestep", "encoder_hidden_states"}, {"out_sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    // load vae_decoder model
    {
        std::string model_path = mModelPath + "/vae_decoder.mnn";
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[2].reset(Module::load(
                                       {"latent_sample"}, {"sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }

    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        log_module_io_mnn("ZIMAGE_text_encoder", mModules[0]);
        log_module_io_mnn("ZIMAGE_unet", mModules[1]);
        log_module_io_mnn("ZIMAGE_vae_decoder", mModules[2]);
    }

    // tokenizer loading
    if(mModelType == STABLE_DIFFUSION_1_5) {
        mTokenizer.reset(new CLIPTokenizer);
    } else if(mModelType == STABLE_DIFFUSION_TAIYI_CHINESE) {
        mTokenizer.reset(new BertTokenizer);
    } else if(mModelType == STABLE_DIFFUSION_ZIMAGE) {
#ifdef MNN_BUILD_LLM
        mTokenizer.reset(new LlmTokenizerWrapper);
#else
        MNN_PRINT("Error: STABLE_DIFFUSION_ZIMAGE requires MNN_BUILD_LLM enabled.\n");
        return false;
#endif
    }
    if (!mTokenizer) {
        MNN_PRINT("Error: tokenizer not initialized for model type %d.\n", (int)mModelType);
        return false;
    }
    if (!mTokenizer->load(mModelPath)) {
        MNN_PRINT("Error: failed to load tokenizer from path %s.\n", mModelPath.c_str());
        return false;
    }
    
    // Resize fix
    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        for (int i = 0; i < (int)mModules.size(); ++i) {
            if (i == 1 || !mModules[i]) {
                continue; // skip UNet for ZImage due to dynamic NonZero/Expand shapes
            }
            mModules[i]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
        }
    } else {
        for (auto& m : mModules) {
            m->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
        }
    }
    // text encoder warmup (skip for ZIMAGE to avoid dynamic NonZero/Expand with all-zero masks)
    {
        if (mModelType != STABLE_DIFFUSION_ZIMAGE) {
            auto outputs = mModules[0]->onForward({mPromptVar});
            text_embeddings = _Convert(outputs[0], NCHW);
        }
    }
    
    if(mMemoryMode > 0) {
        if (mModelType != STABLE_DIFFUSION_ZIMAGE) {
            // unet
            {
                auto outputs = mModules[1]->onForward({mSampleVar, mTimestepVar, text_embeddings});
                auto output = _Convert(outputs[0], NCHW);
                output->readMap<float>();
            }
        }
    }
    if(mMemoryMode == 1) {
        // vae decoder
        {
            auto outputs = mModules[2]->onForward({mLatentVar});
            auto output = _Convert(outputs[0], NCHW);
            output->readMap<float>();
        }
    }
    
    return true;
}

VARP Diffusion::text_encoder(const std::vector<int>& ids) {
    AUTOTIME;
    
    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        int expected = mMaxTextLen * 2;
        MNN_PRINT("[ZIMAGE] text_encoder ids.size=%d expected=%d\n", (int)ids.size(), expected);
        if ((int)ids.size() < expected) {
            MNN_PRINT("Warning: ZIMAGE tokenizer ids size %d smaller than expected %d, remaining positions will be zero-padded.\n", (int)ids.size(), expected);
        }
        auto promptPtr = mPromptVar->writeMap<int>();
        auto maskPtr   = mAttentionMaskVar->writeMap<int>();
        for (int i = 0; i < mMaxTextLen; ++i) {
            int idIndex   = i;
            int maskIndex = mMaxTextLen + i;
            int tokenId   = (idIndex   < (int)ids.size()) ? ids[idIndex]   : 0;
            int maskVal   = (maskIndex < (int)ids.size()) ? ids[maskIndex] : 0;
            promptPtr[i]  = tokenId;
            maskPtr[i]    = maskVal;
        }
        log_var_shape_mnn("ZIMAGE_TE_input_ids", mPromptVar);
        log_var_shape_mnn("ZIMAGE_TE_attention_mask", mAttentionMaskVar);
        print_array_stats("TE_input_ids", promptPtr, static_cast<size_t>(mMaxTextLen));
        print_array_stats("TE_attention_mask", maskPtr, static_cast<size_t>(mMaxTextLen));
        auto outputs = mModules[0]->onForward({mPromptVar, mAttentionMaskVar});
        auto output  = outputs[0];  // keep original [B, L, D] layout for UNet
        log_var_shape_mnn("ZIMAGE_TE_hidden", output);
        print_var_stats("TE_hidden", output);
        {
            auto info = output->getInfo();
            if (info && !info->dim.empty()) {
                int hidden = info->dim.back();
                const float* data = output->readMap<float>();
                if (data && hidden > 0) {
                    print_array_stats("TE_prompt_embed[0]", data, static_cast<size_t>(hidden));
                }
            }
        }
        output.fix(VARP::CONSTANT);
        return output;
    }

    memcpy((void *)mPromptVar->writeMap<int>(), ids.data(), 2*mMaxTextLen*sizeof(int));
    
    auto outputs = mModules[0]->onForward({mPromptVar});
    auto output = _Convert(outputs[0], NCHW);
    output.fix(VARP::CONSTANT);
    
#ifdef MNN_DUMP_DATA
    auto xx = output->readMap<float>();
    for(int i=0; i<10; i+=2) {
        MNN_PRINT("%f %f ", xx[i], xx[i+mMaxTextLen*768]);
    }
    MNN_PRINT("\n\n");
#endif
    return output;
}

VARP Diffusion::step_plms(VARP sample, VARP model_output, int index) {
    int timestep = mTimeSteps[index];
    int prev_timestep = 0;
    if (index + 1 < mTimeSteps.size()) {
        prev_timestep = mTimeSteps[index + 1];
    }
    if (index != 1) {
        if (mEts.size() >= 4) {
            mEts[mEts.size() - 4] = nullptr;
        }
        mEts.push_back(model_output);
    } else {
        timestep = mTimeSteps[0];
        prev_timestep = mTimeSteps[1];
    }
    int ets = mEts.size() - 1;
    if (index == 0) {
        mSample = sample;
    } else if (index == 1) {
        model_output = (model_output + mEts[ets]) * _Const(0.5);
        sample = mSample;
    } else if (ets == 1) {
        model_output = (_Const(3.0) * mEts[ets] - mEts[ets-1]) * _Const(0.5);
    } else if (ets == 2) {
        model_output = (_Const(23.0) * mEts[ets] - _Const(16.0) * mEts[ets-1] + _Const(5.0) * mEts[ets-2]) * _Const(1.0 / 12.0);
    } else if (ets >= 3) {
        model_output = _Const(1. / 24.) * (_Const(55.0) * mEts[ets] - _Const(59.0) * mEts[ets-1] + _Const(37.0) * mEts[ets-2] - _Const(9.0) * mEts[ets-3]);
    }
    auto alpha_prod_t = mAlphas[timestep];
    auto alpha_prod_t_prev = mAlphas[prev_timestep];
    auto beta_prod_t = 1 - alpha_prod_t;
    auto beta_prod_t_prev = 1 - alpha_prod_t_prev;
    auto sample_coeff = std::sqrt(alpha_prod_t_prev / alpha_prod_t);
    auto model_output_denom_coeff = alpha_prod_t * std::sqrt(beta_prod_t_prev) + std::sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);
    auto prev_sample = _Scalar(sample_coeff) * sample - _Scalar((alpha_prod_t_prev - alpha_prod_t)/model_output_denom_coeff) * model_output;
    return prev_sample;
}

VARP Diffusion::unet(VARP text_embeddings, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback) {
    if(mMemoryMode != 1) {
        mModules[0].reset();
    }
    int latentSize = mLatentC * mLatentH * mLatentW;
    if (mInitNoise.size() != latentSize) {
        mInitNoise.resize(latentSize);
    }
    bool loadedNoise = false;
    {
        std::string binPath = mModelPath + "/init_noise.bin";
        std::ifstream bin(binPath.c_str(), std::ios::binary);
        if (bin.good()) {
            bin.read(reinterpret_cast<char*>(mInitNoise.data()), latentSize * sizeof(float));
            loadedNoise = (bin.gcount() == (std::streamsize)(latentSize * sizeof(float)));
        }
        if (!loadedNoise) {
            std::string txtPath = mModelPath + "/init_noise.txt";
            std::ifstream txt(txtPath.c_str());
            if (txt.good()) {
                int count = 0;
                for (; count < latentSize && (txt >> mInitNoise[count]); ++count) {
                }
                loadedNoise = (count == latentSize);
            }
        }
    }

#ifdef MNN_DUMP_DATA
    if (!loadedNoise) {
        std::ostringstream fileName;
        fileName << "random.txt";
        std::ifstream input(fileName.str().c_str());
        if (input.good()) {
            int count = 0;
            for (; count < latentSize && (input >> mInitNoise[count]); ++count) {
            }
            loadedNoise = (count == latentSize);
        }
    }
#endif

    if (!loadedNoise) {
        int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
        std::mt19937 rng;
        rng.seed(seed);

        std::normal_distribution<float> normal(0, 1);
        for (int i = 0; i < latentSize; i++) {
            mInitNoise[i] = normal(rng);
        }
    }
    
    memcpy((void *)mLatentVar->writeMap<float>(), mInitNoise.data(), latentSize * sizeof(float));
    
    // CFG scale variable - use user-provided cfgScale instead of hardcoded 7.5
    VARP scalevar;
    if (mModelType != STABLE_DIFFUSION_ZIMAGE) {
        scalevar = _Input({1}, NCHW, halide_type_of<float>());
        auto scaleptr = scalevar->writeMap<float>();
        scaleptr[0] = cfgScale;  // Use user-provided CFG scale
    }
    
    // Log CFG scale being used
    MNN_PRINT("[UNet] Using CFG scale: %.2f (model=%d)\n", cfgScale, (int)mModelType);
    
    auto floatVar = _Input({1}, NCHW, halide_type_of<float>());
    auto ptr = floatVar->writeMap<float>();
    auto plms = mLatentVar;
    
    int step_index = 0;
    for (int i = 0; i < mTimeSteps.size(); i++) {
        AUTOTIME;
        //display_progress(i, mTimeSteps.size());
        
        if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
            // FlowMatch Euler discrete scheduler for ZImage.
            if (mSigmas.empty() || mSigmas.size() < mTimeSteps.size() + 1) {
                MNN_PRINT("Error: FlowMatch sigma schedule not initialized correctly.\n");
                return nullptr;
            }

            float sigma = mSigmas[i];
            float sigmaNext = mSigmas[i + 1];
            // For FlowMatch Euler, the UNet "t" input aligns with 1 - sigma in Python tests.
            float t = 1.0f - sigma;
            ptr[0] = t;

            mTimestepVar->input(floatVar);
            // CRITICAL: Use _Add with zero to create a new VARP instead of direct assignment
            // Direct assignment (mSampleVar = plms) causes OpenCL buffer invalidation when plms
            // is updated later, leading to setArg err -38 on subsequent onForward() calls
            mSampleVar = plms + _Scalar(0.0f);
            step_index += 1;
            if (step_index == 1) {
                log_var_shape_mnn("ZIMAGE_UNet_in_sample", mSampleVar);
                log_var_shape_mnn("ZIMAGE_UNet_in_timestep", mTimestepVar);
                log_var_shape_mnn("ZIMAGE_UNet_in_encoder_hidden_states", text_embeddings);
                print_var_stats("ZIMAGE_UNet_in_sample_stat", mSampleVar);
                print_var_stats("ZIMAGE_UNet_in_timestep_stat", mTimestepVar);
                print_var_stats("ZIMAGE_UNet_in_encoder_hidden_states_stat", text_embeddings);
            }
            // Debug: print step info
            MNN_PRINT("[UNet] step=%d, sigma=%f, sigmaNext=%f, t=%f, dt=%f\n", 
                      step_index, sigma, sigmaNext, t, sigmaNext - sigma);
            
            // Note: Removed debug readMap() calls - they caused GPU timeout on OpenCL
            // and led to SIGSEGV on subsequent UNet calls
            
            MNN_PRINT("[UNet] Calling mModules[1]->onForward()...\n");
            auto outputs = mModules[1]->onForward({mSampleVar, mTimestepVar, text_embeddings});
            MNN_PRINT("[UNet] onForward() returned, outputs.size()=%zu\n", outputs.size());
            
            if (outputs.empty() || !outputs[0].get()) {
                MNN_PRINT("[UNet] ERROR: outputs is empty or null!\n");
                return nullptr;
            }
            
            // Note: Removed POST-CHECK debug code that called readMap() on full output tensor
            // It caused ~150s delay on OpenCL backend and led to GPU resource timeout/corruption
            
            auto output = _Convert(outputs[0], NCHW);
            // Python side treats model_output as the negative of raw UNet output: noise_pred = -UNet_raw_output
            auto noise_pred = _Scalar(-1.0f) * output;
            print_var_stats("UNet_out_sample", noise_pred);
            
            // Apply CFG scale for ZImage (Flow Matching models)
            // When CFG > 1.0, amplify the noise prediction to strengthen guidance
            // When CFG < 1.0, reduce the noise prediction for softer guidance
            // CFG = 1.0 means no modification (default for ZImage)
            if (std::abs(cfgScale - 1.0f) > 0.001f) {
                noise_pred = _Scalar(cfgScale) * noise_pred;
                MNN_PRINT("[UNet] ZImage CFG applied: scale=%.2f\n", cfgScale);
            }

            float dt = sigmaNext - sigma;
            auto dtVar = _Scalar(dt);
            plms = plms + dtVar * noise_pred;

#ifdef MNN_DUMP_DATA
            auto xx = output->readMap<float>();
            auto yy = mSampleVar->readMap<float>();
            auto zz = text_embeddings->readMap<float>();
            const float* mmF = nullptr;
            const int* mmI = nullptr;
            if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
                mmF = mTimestepVar->readMap<float>();
            } else {
                mmI = mTimestepVar->readMap<int>();
            }

            for(int i=0; i<6; i+=2) {
                if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
                    MNN_PRINT("(0)%f (1)%f (2)%f (3)%f ", xx[i], yy[i], zz[i] ,mmF ? mmF[0] : 0.0f);
                } else {
                    MNN_PRINT("(0)%f (1)%f (2)%f (3)%d ", xx[i], yy[i], zz[i] ,mmI ? mmI[0] : 0);
                }
            }
            MNN_PRINT("\n\n");
#endif
        } else {
            // SD1.5 path: set timestep value for this iteration
            int timestep = mTimeSteps[i];
            ptr[0] = timestep;
            auto temp = _Cast(floatVar, halide_type_of<int>());
            mTimestepVar->input(temp);
            mSampleVar = _Concat({plms, plms}, 0);
            auto outputs = mModules[1]->onForward({mSampleVar, mTimestepVar, text_embeddings});
            auto output = _Convert(outputs[0], NCHW);
            
            auto noise_pred = output;
            
            auto splitvar = _Split(noise_pred, {2}, 0);
            auto noise_pred_uncond = splitvar[0];
            auto noise_pred_text = splitvar[1];
            
            noise_pred = scalevar * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond;
            
            plms = step_plms(plms, noise_pred, i);
            
#ifdef MNN_DUMP_DATA
            auto xx = output->readMap<float>();
            auto yy = mSampleVar->readMap<float>();
            auto zz = text_embeddings->readMap<float>();
            auto mm = mTimestepVar->readMap<int>();

            for(int i=0; i<6; i+=2) {
                MNN_PRINT("(0)%f (1)%f (2)%f (3)%d ", xx[i], yy[i], zz[i] ,mm[0]);
            }
            MNN_PRINT("\n");
            for(int i=0; i<6; i+=2) {
                MNN_PRINT("(0)%f (1)%f (2)%f ", xx[16384+i], yy[16384+i], zz[mMaxTextLen*768+i]);
            }
            MNN_PRINT("\n\n");
#endif
        }
        if (progressCallback) {
            progressCallback((2 + i) * 100 / (iterNum + 3)); // percent
        }
        
    }
    plms.fix(VARP::CONSTANT);
    
#ifdef MNN_DUMP_DATA
    auto xx = plms->readMap<float>();
    for(int i=0; i<10; i+=2) {
        MNN_PRINT("%f ", xx[i]);
    }
    MNN_PRINT("\n\n");
#endif
    return plms;
}

VARP Diffusion::vae_decoder(VARP latent) {
    if(mMemoryMode != 1) {
        mModules[1].reset();
    }
    latent = latent * _Const(1 / 0.18215);
    
    AUTOTIME;
    print_var_stats("VAE_in_latents", latent);
    auto outputs = mModules[2]->onForward({latent});
    auto output = _Convert(outputs[0], NCHW);
    print_var_stats("VAE_out_sample", output);
    
#ifdef MNN_DUMP_DATA
    auto xx = output->readMap<float>();
    for(int i=0; i<320; i+=32) {
        MNN_PRINT("%f ", xx[i]);
    }
    MNN_PRINT("\n\n");
#endif
    
    auto image = output;
    image = _Relu6(image * _Const(0.5) + _Const(0.5), 0, 1);
    print_var_stats("VAE_postprocess_float", image);
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));
    image = _Cast(_Round(image * _Const(255.0)), halide_type_of<uint8_t>());
    // Debug: check pixel range before/after color conversion
    print_var_stats("VAE_postprocess_u8", _Cast(image, halide_type_of<float>()));
    image = cvtColor(image, COLOR_BGR2RGB);
    print_var_stats("VAE_postprocess_u8_rgb", _Cast(image, halide_type_of<float>()));
    image.fix(VARP::CONSTANT);
    return image;
}

bool Diffusion::run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback) {
    AUTOTIME;
    mEts.clear();
 
    if(iterNum > 50) {
        iterNum = 50;
        MNN_PRINT("too much number of iterations, iterations will be set to 50.\n");
    }
    if(iterNum < 1) {
        iterNum = 10;
        MNN_PRINT("illegal number of iterations, iterations will be set to 10.\n");
    }
    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        // Build FlowMatch Euler sigma schedule for ZImage based on iterNum from CLI.
        FlowMatchEulerScheduler scheduler(mTrainTimestepsNum, mFlowShift, mUseDynamicShifting);
        mSigmas = scheduler.get_sigmas(iterNum);

        mTimeSteps.resize(iterNum);
        for (int i = 0; i < iterNum; ++i) {
            mTimeSteps[i] = i; // only used for loop length in unet()
        }
    } else {
        mTimeSteps.resize(iterNum);
        int step = 1000 / iterNum;
        for(int i = iterNum - 1; i >= 0; i--) {
            mTimeSteps[i] = 1 + (iterNum - 1 - i) * step;
        }
    }

    std::string promptForTokenizer = prompt;
    if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        // Mirror Python tokenizer.apply_chat_template for a single user message:
        //   messages=[{"role":"user","content": prompt}], add_generation_prompt=True, enable_thinking=True
        // This ensures C++ input_ids/attention_mask align with the Python MNN pipeline.
        promptForTokenizer = std::string("<|im_start|>user\n") + prompt +
                            std::string("<|im_end|>\n<|im_start|>assistant\n<think>\n");
    }

    auto ids = mTokenizer->encode(promptForTokenizer, mMaxTextLen);

    auto text_embeddings = text_encoder(ids);
     
    if (progressCallback) {
        progressCallback(1 * 100 / (iterNum + 3)); // percent
    }
    auto latent = unet(text_embeddings, iterNum, randomSeed, cfgScale, progressCallback);
    print_var_stats("UNet_out_latent", latent);
     
    auto image = vae_decoder(latent);
    bool res = imwrite(imagePath, image);
    if (res) {
        MNN_PRINT("SUCCESS! write generated image to %s\n", imagePath.c_str());
    }

    if(mMemoryMode != 1) {
        mModules[2].reset();
    }
     
    if (progressCallback) {
        progressCallback(100); // percent
    }
    return res;
}
}
}
