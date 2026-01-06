//
//  diffusion.hpp
//
//  Created by MNN on 2025/01/12.
//  MNN
//
#ifndef MNN_DIFFUSION_HPP
#define MNN_DIFFUSION_HPP

#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Module.hpp>
#include <cv/cv.hpp>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

using namespace MNN;
using namespace MNN::Express;

namespace MNN {
namespace DIFFUSION {

class Tokenizer;

// DiffusionConfig: Parse model file paths from config.json
// Fallback to hardcoded defaults if config.json not found (for backward compatibility)
class DiffusionConfig {
public:
    std::string base_dir_;
    rapidjson::Document config_;
    bool config_loaded_ = false;
    
    DiffusionConfig(const std::string& model_path) {
        base_dir_ = model_path;
        if (!base_dir_.empty() && base_dir_.back() != '/') {
            base_dir_ += "/";
        }
        
        // Try to load config.json
        std::string config_path = model_path + "/config.json";
        std::ifstream config_file(config_path);
        if (config_file.is_open()) {
            rapidjson::IStreamWrapper isw(config_file);
            config_.ParseStream(isw);
            if (!config_.HasParseError() && config_.IsObject()) {
                config_loaded_ = true;
                MNN_PRINT("[DiffusionConfig] Loaded config.json from %s\n", config_path.c_str());
            } else {
                MNN_PRINT("[DiffusionConfig] Failed to parse config.json, using defaults\n");
            }
        } else {
            MNN_PRINT("[DiffusionConfig] config.json not found, using hardcoded defaults\n");
        }
    }
    
    // Get text encoder model path
    std::string text_encoder_model() const {
        if (!config_loaded_) {
            return base_dir_ + "text_encoder.mnn";
        }
        
        if (config_.HasMember("text_encoder") && config_["text_encoder"].IsObject()) {
            const auto& te_config = config_["text_encoder"];
            std::string dir = "";
            std::string prefix = "";
            
            if (te_config.HasMember("directory") && te_config["directory"].IsString()) {
                dir = te_config["directory"].GetString();
                if (!dir.empty()) {
                    prefix = dir + "/";
                }
            }
            
            // LongCat: text_encoder.llm.model
            if (te_config.HasMember("llm") && te_config["llm"].IsObject()) {
                const auto& llm_config = te_config["llm"];
                if (llm_config.HasMember("model") && llm_config["model"].IsString()) {
                    return base_dir_ + prefix + llm_config["model"].GetString();
                }
            }
            
            // ZImage: text_encoder.model
            if (te_config.HasMember("model") && te_config["model"].IsString()) {
                return base_dir_ + prefix + te_config["model"].GetString();
            }
        }
        
        return base_dir_ + "text_encoder.mnn";  // Default fallback
    }
    
    // Get UNet model path
    std::string unet_model() const {
        if (!config_loaded_) {
            return base_dir_ + "unet.mnn";
        }
        
        // Try transformer (LongCat style)
        if (config_.HasMember("transformer") && config_["transformer"].IsObject()) {
            const auto& transformer_config = config_["transformer"];
            if (transformer_config.HasMember("model") && transformer_config["model"].IsString()) {
                return base_dir_ + transformer_config["model"].GetString();
            }
        }
        
        // Try unet (ZImage/SD1.5 style)
        if (config_.HasMember("unet") && config_["unet"].IsObject()) {
            const auto& unet_config = config_["unet"];
            if (unet_config.HasMember("model") && unet_config["model"].IsString()) {
                return base_dir_ + unet_config["model"].GetString();
            }
        }
        
        return base_dir_ + "unet.mnn";  // Default fallback
    }
    
    // Get VAE decoder model path
    std::string vae_decoder_model() const {
        if (!config_loaded_) {
            return base_dir_ + "vae_decoder.mnn";
        }
        
        if (config_.HasMember("vae") && config_["vae"].IsObject()) {
            const auto& vae_config = config_["vae"];
            if (vae_config.HasMember("decoder_model") && vae_config["decoder_model"].IsString()) {
                return base_dir_ + vae_config["decoder_model"].GetString();
            }
        }
        
        return base_dir_ + "vae_decoder.mnn";  // Default fallback
    }
    
    // Get VAE encoder model path
    std::string vae_encoder_model() const {
        if (!config_loaded_) {
            return base_dir_ + "vae_encoder.mnn";
        }
        
        if (config_.HasMember("vae") && config_["vae"].IsObject()) {
            const auto& vae_config = config_["vae"];
            if (vae_config.HasMember("encoder_model") && vae_config["encoder_model"].IsString()) {
                return base_dir_ + vae_config["encoder_model"].GetString();
            }
        }
        
        return base_dir_ + "vae_encoder.mnn";  // Default fallback
    }
};
typedef enum {
    STABLE_DIFFUSION_1_5 = 0,
    STABLE_DIFFUSION_TAIYI_CHINESE = 1,
    STABLE_DIFFUSION_ZIMAGE = 2,
    LONGCAT_IMAGE_EDIT = 3,
} DiffusionModelType;

// GPU memory mode for OpenCL backend
typedef enum {
    GPU_MEMORY_AUTO = 0,    // Auto select based on backend and model type
    GPU_MEMORY_BUFFER = 1,  // Use OpenCL buffer mode
    GPU_MEMORY_IMAGE = 2,   // Use OpenCL image mode (recommended for Adreno)
} DiffusionGpuMemoryMode;

// Precision mode for inference
typedef enum {
    PRECISION_AUTO = 0,     // Auto select (FP32 for OpenCL on Adreno, FP16 otherwise)
    PRECISION_LOW = 1,      // FP16 precision
    PRECISION_NORMAL = 2,   // FP32 precision (normal)
    PRECISION_HIGH = 3,     // FP32 precision (high)
} DiffusionPrecisionMode;

// CFG (Classifier-Free Guidance) mode for dual-UNet models (e.g., LongCat)
// Controls the sigma range where CFG is applied (Limited Interval CFG)
typedef enum {
    CFG_MODE_AUTO = 0,      // Model default (LongCat: 0.1~0.8)
    CFG_MODE_WIDE = 1,      // 0.1~0.9 (widest, strongest guidance)
    CFG_MODE_STANDARD = 2,  // 0.1~0.8 (standard)
    CFG_MODE_MEDIUM = 3,    // 0.15~0.7 (medium)
    CFG_MODE_NARROW = 4,    // 0.2~0.6 (narrow)
    CFG_MODE_MINIMAL = 5,   // 0.25~0.5 (minimal guidance)
} DiffusionCFGMode;

// LLM Text Encoder configuration for multimodal models (e.g., LongCat)
// These parameters control hidden states slicing and padding
struct LLMEncoderConfig {
    int prefixLen = 67;       // Tokens to remove from beginning (system prompt overhead)
    int suffixLen = 5;        // Tokens to remove from end (assistant prompt tokens)
    int targetSeqLen = 838;   // Target sequence length after padding (Image Edit mode)
    int tokenizerMaxLength = 512; // Tokenizer max length for T2I mode (same for both T2I and Image Edit)
    int visionResizeSize = 512; // Image resize target for vision encoder
    int hiddenSize = 3584;    // Hidden state dimension (model-specific)
    
    // Default config for LongCat
    static LLMEncoderConfig longcat() {
        LLMEncoderConfig cfg;
        cfg.prefixLen = 67;
        cfg.suffixLen = 5;
        cfg.targetSeqLen = 838;
        cfg.tokenizerMaxLength = 512;
        cfg.visionResizeSize = 512;
        cfg.hiddenSize = 3584;
        return cfg;
    }
};

// ===== Unified Scheduler API Types =====
// Scheduler type for Layer 2 update
enum SchedulerType {
    SCHEDULER_PLMS,   // SD1.5: 4-order PLMS with history
    SCHEDULER_EULER   // Z-Image, LongCat: FlowMatch Euler 1-order
};

// UNet output preprocessing function type for Layer 1
// LongCat: GPU slice+unpack; SD1.5/Z-Image: pass-through
using UNetPreprocessFunc = std::function<VARP(VARP)>;

class MNN_PUBLIC Diffusion {
public:
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize);
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU);
    // New constructor with GPU memory mode, precision control and thread count
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads = 4);
    // Constructor with separate width and height for non-square aspect ratios (e.g., 1280x720, 768x1024)
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads = 4);
    virtual ~Diffusion();
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize);
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU);
    // New factory method with GPU memory mode, precision control and thread count
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads = 4);
    // Factory method with separate width and height for non-square aspect ratios
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads = 4);
    // Full factory method with all options including VAE on CPU and CFG mode
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, DiffusionCFGMode cfgMode, int numThreads = 4);

    bool run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback, const std::string inputImagePath = "");
    bool load();

    // ===== Image Processing Utility Functions =====
    // These functions are model-agnostic and can be used by SD1.5, Z-image, LongCat, etc.
    
    // Resize image preserving aspect ratio, then center crop to target size
    // This is the standard preprocessing for diffusion models
    static VARP resizeAndCenterCrop(VARP image, int targetW, int targetH);
    
    // Convert BGR to RGB (OpenCV uses BGR by default)
    // Input/Output: HWC format [H, W, 3]
    static VARP bgrToRgb(VARP bgrImage);
    
    // Convert RGB to BGR (for OpenCV imwrite)
    // Input/Output: HWC format [H, W, 3]
    static VARP rgbToBgr(VARP rgbImage);
    
    // Convert HWC format to NCHW format for neural network input
    // normalize: if true, apply (x / 127.5 - 1.0) normalization
    static VARP hwcToNchw(VARP hwcImage, bool normalize = false);
    
    // Convert NCHW format to HWC format for image output
    // denormalize: if true, apply (x + 1.0) * 127.5 denormalization
    static VARP nchwToHwc(VARP nchwImage, bool denormalize = false);
    
    // ===== LongCat-specific Latent Packing Functions =====
    // These functions handle the packed sequence format used by Flux-like models
    
    // Pack latents from NCHW to packed sequence format
    // Input:  [B, C, H, W] - standard NCHW latents
    // Output: [B, H/2*W/2, C*4] - packed sequence format
    // seqOffset: offset for concatenating multiple latent sources (e.g., noise + image)
    static void packLatents(const float* src, float* dst, int B, int C, int H, int W, int seqOffset = 0);
    
    // Unpack latents from packed sequence format to NCHW
    // Input:  [B, seq, 64] - packed sequence format
    // Output: [B, C, H, W] - standard NCHW latents
    // seqLen: total sequence length (for validation)
    static void unpackLatents(const float* src, float* dst, int B, int C, int H, int W, int seqLen);

private:
    VARP step_plms(VARP sample, VARP model_output, int index);
    VARP text_encoder(const std::vector<int>& ids);
    VARP text_encoder_llm(const std::string& prompt, VARP preprocessedImage);  // LLM-based text encoder for LongCat
    VARP unet(VARP text_embeddings, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback);
    VARP vae_decoder(VARP latent);
    void getCFGSigmaRange(float& sigmaLow, float& sigmaHigh) const;  // Get CFG sigma range based on mode
    
    // ===== Unified Scheduler API Functions =====
    // GPU-side unpack for LongCat (pure GPU, no CPU sync)
    VARP unpackLatentsGPU(VARP packed, int B, int C, int H, int W);
    // Scheduler update functions
    VARP applyPLMSUpdate(VARP sample, VARP noise_pred, float dt, int step);
    VARP applyEulerUpdate(VARP sample, VARP noise_pred, float dt);
private:
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_cpu_; // CPU runtime for text_encoder to avoid OpenCL buffer size limit
    std::vector<std::shared_ptr<Module>> mModules;
    // step_plms / schedulers
    std::vector<int> mTimeSteps;
    std::vector<float> mAlphas;
    std::vector<float> mSigmas; // FlowMatch Euler sigmas for ZImage
    std::vector<VARP> mEts;
    VARP mSample;
    VARP mTxtIdsVar, mImgIdsVar;  // LongCat position IDs
    VARP mImageLatentsVar;  // LongCat image latents from VAE encoder
    VARP mLatentVar, mPromptVar, mAttentionMaskVar, mTimestepVar, mSampleVar;
    std::vector<float> mInitNoise;
    
    // ===== Unified Scheduler API State =====
    UNetPreprocessFunc mUNetPreprocess;  // Layer 1: UNet output preprocessing
    SchedulerType mSchedulerType;        // Layer 2: Scheduler type
    
private:
    std::string mModelPath;
    DiffusionModelType mModelType;
    int mMaxTextLen = 77;
    int mTrainTimestepsNum = 1000;
    float mFlowShift = 3.0f;
    bool mUseDynamicShifting = false;
    int mImageSize = 0;    // Legacy: single size for square images
    int mImageWidth = 0;   // Output image width (must be multiple of 8)
    int mImageHeight = 0;  // Output image height (must be multiple of 8)
    int mLatentC = 4;
    int mLatentH = 64;
    int mLatentW = 64;
    /* 0 -> memory saving mode, for memory stictly limited application
        1 -> memory enough mode, for better image generation speed
        2 -> balance mode for memory and generation speed.
     */
    int mMemoryMode;
    MNNForwardType mBackendType;
    bool mTextEncoderOnCPU = true; // Force text_encoder to run on CPU to avoid GPU buffer size limit
    bool mVaeOnCPU = false;  // Force VAE encoder/decoder to run on CPU
    DiffusionGpuMemoryMode mGpuMemoryMode = GPU_MEMORY_AUTO;
    DiffusionPrecisionMode mPrecisionMode = PRECISION_AUTO;
    DiffusionCFGMode mCFGMode = CFG_MODE_AUTO;  // CFG mode for dual-UNet models
    int mNumThreads = 4;  // CPU thread count, configurable from settings
    std::unique_ptr<Tokenizer> mTokenizer;
    
    // LongCat config from config.json
    std::string mTextEncoderDir = "text_encoder";
    float mDefaultCfgScale = 4.5f;
    
    // LLM encoder config (model-specific parameters)
    LLMEncoderConfig mLlmEncoderConfig;
    
    // LLM for LongCat text encoder (lazy loaded)
    void* mLlm = nullptr;  // Llm* pointer, void* to avoid header dependency
};

}
}
#endif
