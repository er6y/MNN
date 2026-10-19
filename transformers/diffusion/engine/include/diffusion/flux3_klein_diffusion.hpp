//  flux3_klein_diffusion.hpp - FLUX.3 Image Klein (step-distilled flow-map) diffusion model
#ifndef MNN_FLUX3_KLEIN_DIFFUSION_HPP
#define MNN_FLUX3_KLEIN_DIFFUSION_HPP

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

class MNN_PUBLIC Flux3KleinDiffusion : public Diffusion {
public:
    Flux3KleinDiffusion(std::string modelPath, DiffusionModelType modelType,
                        MNNForwardType backendType, int memoryMode,
                        int imageWidth, int imageHeight,
                        bool textEncoderOnCPU, bool vaeOnCPU,
                        DiffusionGpuMemoryMode gpuMemoryMode,
                        DiffusionPrecisionMode precisionMode,
                        DiffusionCFGMode cfgMode, int numThreads = 4);
    virtual ~Flux3KleinDiffusion();

    virtual bool load() override;

    virtual bool run(const std::string prompt, const std::string imagePath,
                     int iterNum, int randomSeed,
                     std::function<void(int)> progressCallback) override;

    virtual bool run(const std::string prompt, const std::string outputPath,
                     int iterNum, int randomSeed, float cfgScale,
                     std::function<void(int)> progressCallback,
                     const std::string inputImagePath = "") override;

    virtual bool run(const VARP input_embeds, const std::string& mode,
                     const std::string& inputImagePath, const std::string& outputImagePath,
                     int width, int height, int iterNum, int randomSeed,
                     bool use_cfg, float cfg_scale,
                     std::function<void(int)> progressCallback) override;

private:
    VARP text_encoder_llm(const std::string& prompt);
    VARP unet(VARP textEmbeds, VARP imageLatents,
              int iterNum, int randomSeed,
              std::function<void(int)> progressCallback);
    VARP vae_decoder(VARP latent);
    VARP vae_encoder(VARP image);
    float computeEmpiricalMu(int imageSeqLen, int numSteps) const;
    std::vector<float> getSigmas(int numSteps, int imageSeqLen) const;

    void packLatentsToSeq(const float* src, float* dst, int B, int C, int H, int W) const;
    void unpackLatentsToPatchified(const float* src, float* dst,
                                   int B, int C, int H, int W, int seqLen) const;
    // FLUX.3 ids: (t=0, h*scale, w*scale, l); generated image l=0, references l=-1,-2,...
    void prepareImgIds(float* dst, int H, int W, int seqOffset, float lCoord) const;
    void prepareTxtIds(float* dst, int seqLen) const;

    // Scheduler params (constant shift; FLUX.3 recipe alpha=6.93, no dynamic shifting)
    float mBaseShift          = 0.5f;
    float mMaxShift           = 1.15f;
    int   mBaseImageSeqLen    = 256;
    int   mMaxImageSeqLen     = 4096;

    // Latent dims
    int mLatentC = 32;   // VAE latent channels (before patchify)
    int mPackedC = 128;  // C*4 after patchify
    int mLatentH = 32;   // H/8
    int mLatentW = 32;   // W/8
    // config inference.height/width (static per-size export is authoritative)
    int mConfigImageWidth  = 0;
    int mConfigImageHeight = 0;

    // Text encoder
    // chat template hardcoded: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    // window = ceil(real_tokens / mPadMultiple) * mPadMultiple, capped at mMaxTextLen;
    // mFixedTextLen > 0 (config text_len, static DiT export) pins the window instead
    int mPadMultiple  = 80;
    int mMaxTextLen   = 8192;
    int mFixedTextLen = 0;
    std::string mTokenizerPath;   // resolved from config text_encoder.directory/tokenizer
#ifdef MNN_BUILD_LLM
    std::unique_ptr<MNN::Transformer::Tokenizer> mTokenizer;
#endif

    // VAE BN params (patchify-space, 128 channels). Decode denormalizes
    // (x * bn_std + bn_mean) BEFORE vae.decode; encode normalizes after patchify.
    // bn_std is expected precomputed as sqrt(running_var + eps).
    std::vector<float> mVaeBnMean;
    std::vector<float> mVaeBnStd;

    static const int ID_DIM = 4;              // FLUX.3 uses 4D ids (t, h, w, l)
    float mImageRopeScale = 0.5f;             // image H/W id scale (config dit.image_rope_scale)

    std::unique_ptr<DiffusionConfig> mDiffConfig;
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_FLUX3_KLEIN_DIFFUSION_HPP
