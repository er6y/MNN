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
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Module.hpp>

using namespace MNN;
using namespace MNN::Express;

namespace MNN {
namespace DIFFUSION {

class Tokenizer;
typedef enum {
    STABLE_DIFFUSION_1_5 = 0,
    STABLE_DIFFUSION_TAIYI_CHINESE = 1,
    STABLE_DIFFUSION_ZIMAGE = 2,
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

class MNN_PUBLIC Diffusion {
public:
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize);
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU);
    // New constructor with GPU memory mode, precision control and thread count
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads = 4);
    virtual ~Diffusion();
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize);
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU);
    // New factory method with GPU memory mode, precision control and thread count
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads = 4);

    bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback);
    bool load();
private:
    VARP step_plms(VARP sample, VARP model_output, int index);
    VARP text_encoder(const std::vector<int>& ids);
    VARP unet(VARP text_embeddings, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback);
    VARP vae_decoder(VARP latent);
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
    VARP mLatentVar, mPromptVar, mAttentionMaskVar, mTimestepVar, mSampleVar;
    std::vector<float> mInitNoise;
    
private:
    std::string mModelPath;
    DiffusionModelType mModelType;
    int mMaxTextLen = 77;
    int mTrainTimestepsNum = 1000;
    float mFlowShift = 3.0f;
    bool mUseDynamicShifting = false;
    int mImageSize = 0;
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
    DiffusionGpuMemoryMode mGpuMemoryMode = GPU_MEMORY_AUTO;
    DiffusionPrecisionMode mPrecisionMode = PRECISION_AUTO;
    int mNumThreads = 4;  // CPU thread count, configurable from settings
    std::unique_ptr<Tokenizer> mTokenizer;
};

}
}
#endif
