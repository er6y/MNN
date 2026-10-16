//  qwenimage21_diffusion.hpp - Qwen-Image-2.1 text-to-image
//  TE: Qwen3-VL 7B (llm.mnn + external bf16 embedding table), DiT: 7B flow-match,
//  VAE: AutoencoderKLQwenImage21 dynamic decoder. CFG-free (guidance distilled, true_cfg=1.0).
#ifndef MNN_QWEN_IMAGE_21_DIFFUSION_HPP
#define MNN_QWEN_IMAGE_21_DIFFUSION_HPP

#include "diffusion.hpp"
#include "diffusion_config.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#ifdef MNN_BUILD_LLM
#include "llm/tokenizer.hpp"
#endif

namespace MNN {
namespace DIFFUSION {

class MNN_PUBLIC QwenImage21Diffusion : public Diffusion {
public:
    QwenImage21Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType,
                         int memoryMode);
    QwenImage21Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType,
                         int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU,
                         bool vaeOnCPU, DiffusionGpuMemoryMode gpuMemoryMode,
                         DiffusionPrecisionMode precisionMode, DiffusionCFGMode cfgMode, int numThreads);
    ~QwenImage21Diffusion() override;

    bool load() override;
    bool run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed,
             std::function<void(int)> progressCallback) override;
    // Sana-style interface: embeds ignored, delegates to the prompt run.
    bool run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
             const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
             bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) override;

private:
    void readEngineConfig();
    // prompt -> packed text embedding [textLen x hidden] + validity mask [textLen]
    bool textEncode(const std::string& prompt, std::vector<float>& enc, std::vector<int>& tmask);
    static bool readFileBytes(const std::string& path, std::vector<uint8_t>& out);
    // low-memory staged loading (memory_mode==0): DiT/VAE load on demand
    bool ensureDiTLoaded();
    bool ensureVAELoaded();

#ifdef MNN_BUILD_LLM
    std::unique_ptr<MNN::Transformer::Tokenizer> mTokenizer;
#endif

    // module slots: 0 = TE, 1 = DiT, 2 = VAE decoder
    int mTextLen = 1024;
    int mHidden  = 4096;
    int mDropIdx = 14;
    // Resolved model paths, shared DiffusionConfig convention (ZImage/LongCat):
    // text_encoder.{directory,model} + embeddings/tokenizer inside that directory,
    // transformer.model (default unet.mnn), vae.decoder_model (default vae_decoder.mnn)
    std::string mTeModelPath;
    std::string mEmbeddingsFile;
    std::string mTokenizerFile;
    std::string mDiTModelPath;
    std::string mVaeDecPath;
    bool mRopeIndexInputs = false; // dynamic-size DiT takes (frame,height,width) index inputs
    std::shared_ptr<Executor::RuntimeManager> mVaeRt; // kept for staged VAE load (memory_mode==0)
};

} // namespace DIFFUSION
} // namespace MNN
#endif
