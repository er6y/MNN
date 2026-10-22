//  moss_image_diffusion.hpp - MOSS-Image (OpenMOSS, 5B single-stream DiT,
//  8-step DMD turbo LoRA) diffusion model.
//  Contract (matches work/moss_export_dit.py --dynamic):
//   - DiT inputs: hidden_states [1,128,pH,pW] (packed latent, NCHW),
//     timestep [1] raw 0..1000, encoder_hidden_states [1,S,4096],
//     rope_img [1,imgSeq,128], rope_txt [1,S,128]; per-token rope layout
//     [sin_r16,sin_f16,sin_h16,sin_w16,cos_r16,cos_f16,cos_h16,cos_w16],
//     fp64 host build; img coords (0,txtLen,h-(pH-1)/2,w-(pW-1)/2),
//     txt coords (0,pos,0,0).
//   - Output out_sample [1, imgSeq+txtSeq, 128]: per-row ops only after the
//     stream concat, the host reads the first imgSeq rows as the velocity.
//   - Scheduler MOSSImageScheduler: shift-5 grid, nearest-snap to the 1000
//     training grid (training_shift=1), consecutive-dedup, NO second shift;
//     Euler dt = sigma[i+1] - sigma[i], terminal 0.
//   - TE = Qwen3.5-9B GDN via llmexport contract (float embeds in, causal
//     mask built host-side, EXACT length - GDN is padding-sensitive).
//   - No image editing (model does not support reference conditioning).
#ifndef MNN_MOSS_IMAGE_DIFFUSION_HPP
#define MNN_MOSS_IMAGE_DIFFUSION_HPP

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

class MNN_PUBLIC MossImageDiffusion : public Diffusion {
public:
    MossImageDiffusion(std::string modelPath, DiffusionModelType modelType,
                       MNNForwardType backendType, int memoryMode,
                       int imageWidth, int imageHeight,
                       bool textEncoderOnCPU, bool vaeOnCPU,
                       DiffusionGpuMemoryMode gpuMemoryMode,
                       DiffusionPrecisionMode precisionMode,
                       DiffusionCFGMode cfgMode, int numThreads = 4);
    virtual ~MossImageDiffusion();

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
    // TE: tokenize + bf16 embedding lookup + exact-length forward
    bool ensureTeLoaded();
    bool textEncode(const std::string& prompt, std::vector<float>& embeds, int& seqLen);
    bool ensureDiTLoaded();
    bool ensureVAELoaded();
    VARP denoiseLoop(const std::vector<float>& textEmbeds, int txtSeqLen,
                     int iterNum, int randomSeed,
                     std::function<void(int)> progressCallback);
    VARP vae_decoder(VARP latent);

    // MOSSImageScheduler: shift-5 grid snapped to the training 1000-grid.
    // Returns numSteps+1 entries, last = 0 (terminal).
    std::vector<float> getMossSigmas(int numSteps) const;

    // config
    double mSchedulerShift    = 5.0;
    double mTrainingShift     = 1.0;
    int    mTrainTimesteps    = 1000;
    int    mMaxTextLen        = 4096;

    // latent dims (packed 128ch on the pH=pH16 patch grid)
    int mLatentC = 32;   // VAE latent channels (before patchify)
    int mPackedC = 128;  // C*4 after patchify
    int mLatentH = 32;   // image/8 (VAE space)
    int mLatentW = 32;
    int mConfigImageWidth  = 0;
    int mConfigImageHeight = 0;

    // TE (llmexport contract, qwenimage21 style)
    std::string mTeDir;            // text_encoder directory (tokenizer/embeddings live here)
    std::string mTokenizerPath;
    std::string mEmbeddingsFile;   // embeddings_bf16.bin
    int mHidden = 4096;
#ifdef MNN_BUILD_LLM
    std::unique_ptr<MNN::Transformer::Tokenizer> mTokenizer;
#endif

    // VAE BN params (patchify-space, 128 channels); decode denormalizes
    // (x * bn_std + bn_mean) BEFORE vae.decode. bn_std precomputed sqrt(var+eps).
    std::vector<float> mVaeBnMean;
    std::vector<float> mVaeBnStd;

    std::unique_ptr<DiffusionConfig> mDiffConfig;
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_MOSS_IMAGE_DIFFUSION_HPP
