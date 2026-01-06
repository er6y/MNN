#include <random>
#include <fstream>
#include <chrono>
#include <array>
#include "diffusion/diffusion.hpp"
#include "tokenizer.hpp"
#include "scheduler.hpp"
#ifdef MNN_BUILD_LLM
#include "llm/tokenizer.hpp"
#include "llm/llm.hpp"
#include "../../../llm/engine/src/llmconfig.hpp"
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

using namespace CV;

namespace MNN {
namespace DIFFUSION {

// Philox RNG implementation (aligned with PyTorch)
// Based on: Salmon et al. 2011, "Parallel Random Numbers: As Easy as 1, 2, 3"
class PhiloxRNG {
public:
    using result_type = uint32_t;
    using key_type = std::array<uint32_t, 2>;
    using counter_type = std::array<uint32_t, 4>;
    
    // Philox 4x32-10 constants
    static constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53;
    static constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57;
    static constexpr uint32_t PHILOX_W32_0 = 0x9E3779B9;
    static constexpr uint32_t PHILOX_W32_1 = 0xBB67AE85;
    
    PhiloxRNG(uint64_t seed = 0, uint64_t offset = 0) {
        key_[0] = static_cast<uint32_t>(seed);
        key_[1] = static_cast<uint32_t>(seed >> 32);
        counter_[0] = static_cast<uint32_t>(offset);
        counter_[1] = static_cast<uint32_t>(offset >> 32);
        counter_[2] = 0;
        counter_[3] = 0;
        index_ = 0;
    }
    
    // Generate next random uint32
    uint32_t operator()() {
        if (index_ == 0) {
            counter_type result = philox4x32_10(counter_, key_);
            output_[0] = result[0];
            output_[1] = result[1];
            output_[2] = result[2];
            output_[3] = result[3];
            increment_counter();
        }
        uint32_t ret = output_[index_];
        index_ = (index_ + 1) & 3;
        return ret;
    }
    
    // Generate random float in [0, 1) - aligned with PyTorch
    float uniform() {
        uint32_t x = (*this)();
        // PyTorch uses full precision: x / 2^32
        return static_cast<float>(x) * (1.0f / 4294967296.0f);
    }
    
    // Generate random normal distribution using Box-Muller transform
    // Note: We only use cos() output to maintain better spatial distribution
    float randn() {
        float u1 = uniform();
        float u2 = uniform();
        
        // Avoid log(0)
        u1 = u1 < 1e-10f ? 1e-10f : u1;
        
        // Box-Muller transform
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;
        
        // Only return cos() output for better spatial correlation
        return r * std::cos(theta);
    }
    
private:
    key_type key_;
    counter_type counter_;
    uint32_t output_[4];
    int index_;
    
    void increment_counter() {
        counter_[0]++;
        if (counter_[0] == 0) {
            counter_[1]++;
            if (counter_[1] == 0) {
                counter_[2]++;
                if (counter_[2] == 0) {
                    counter_[3]++;
                }
            }
        }
    }
    
    // Multiply two 32-bit integers and return high and low 32 bits
    static inline uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* hi) {
        uint64_t product = static_cast<uint64_t>(a) * b;
        *hi = static_cast<uint32_t>(product >> 32);
        return static_cast<uint32_t>(product);
    }
    
    // Single round of Philox 4x32
    static counter_type philox4x32_round(counter_type ctr, key_type key) {
        uint32_t hi0, hi1;
        uint32_t lo0 = mulhilo32(PHILOX_M4x32_0, ctr[0], &hi0);
        uint32_t lo1 = mulhilo32(PHILOX_M4x32_1, ctr[2], &hi1);
        
        counter_type ret;
        ret[0] = hi1 ^ ctr[1] ^ key[0];
        ret[1] = lo1;
        ret[2] = hi0 ^ ctr[3] ^ key[1];
        ret[3] = lo0;
        return ret;
    }
    
    // Philox 4x32-10: 10 rounds of mixing
    static counter_type philox4x32_10(counter_type ctr, key_type key) {
        // Round 1
        ctr = philox4x32_round(ctr, key);
        key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1;
        
        // Round 2
        ctr = philox4x32_round(ctr, key);
        key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1;
        
        // Round 3
        ctr = philox4x32_round(ctr, key);
        key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1;
        
        // Round 4
        ctr = philox4x32_round(ctr, key);
        key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1;
        
        // Round 5
        ctr = philox4x32_round(ctr, key);
        key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1;
        
        // Round 6
        ctr = philox4x32_round(ctr, key);
        key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1;
        
        // Round 7
        ctr = philox4x32_round(ctr, key);
        key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1;
        
        // Round 8
        ctr = philox4x32_round(ctr, key);
        key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1;
        
        // Round 9
        ctr = philox4x32_round(ctr, key);
        key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1;
        
        // Round 10 (final)
        ctr = philox4x32_round(ctr, key);
        
        return ctr;
    }
};

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

// ===== Image Processing Utility Functions Implementation =====

VARP Diffusion::resizeAndCenterCrop(VARP image, int targetW, int targetH) {
    auto info = image->getInfo();
    if (!info || info->dim.size() != 3) {
        MNN_ERROR("resizeAndCenterCrop: Invalid input shape, expected [H, W, C]\n");
        return nullptr;
    }
    
    int origH = info->dim[0];
    int origW = info->dim[1];
    int origC = info->dim[2];
    
    // Calculate aspect ratios
    float aspectRatio = static_cast<float>(origW) / origH;
    float targetAspect = static_cast<float>(targetW) / targetH;
    
    // Calculate resize dimensions to preserve aspect ratio
    int resizeW, resizeH;
    if (aspectRatio > targetAspect) {
        // Image is wider than target, fit height
        resizeH = targetH;
        resizeW = static_cast<int>(targetH * aspectRatio);
    } else {
        // Image is taller than target, fit width
        resizeW = targetW;
        resizeH = static_cast<int>(targetW / aspectRatio);
    }
    
    // Resize image using MNN CV API
    Size resizeSize(resizeW, resizeH);
    auto resized = resize(image, resizeSize);
    
    // Center crop to target size (manual implementation)
    int cropX = (resizeW - targetW) / 2;
    int cropY = (resizeH - targetH) / 2;
    
    auto resizedPtr = resized->readMap<uint8_t>();
    VARP cropped = _Input({targetH, targetW, origC}, NHWC, halide_type_of<uint8_t>());
    auto croppedPtr = cropped->writeMap<uint8_t>();
    
    for (int h = 0; h < targetH; ++h) {
        int srcRowOffset = (cropY + h) * resizeW * origC;
        int dstRowOffset = h * targetW * origC;
        for (int w = 0; w < targetW; ++w) {
            int srcOffset = srcRowOffset + (cropX + w) * origC;
            int dstOffset = dstRowOffset + w * origC;
            for (int c = 0; c < origC; ++c) {
                croppedPtr[dstOffset + c] = resizedPtr[srcOffset + c];
            }
        }
    }
    
    return cropped;
}

VARP Diffusion::bgrToRgb(VARP bgrImage) {
    auto info = bgrImage->getInfo();
    if (!info || info->dim.size() != 3 || info->dim[2] != 3) {
        MNN_ERROR("bgrToRgb: Invalid input shape, expected [H, W, 3]\n");
        return nullptr;
    }
    
    int H = info->dim[0];
    int W = info->dim[1];
    int C = info->dim[2];
    
    // Create output tensor
    VARP rgbImage = _Input({H, W, C}, info->order, info->type);
    
    auto bgrPtr = bgrImage->readMap<uint8_t>();
    auto rgbPtr = rgbImage->writeMap<uint8_t>();
    
    // Swap channels: BGR -> RGB
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx = (h * W + w) * C;
            rgbPtr[idx + 0] = bgrPtr[idx + 2];  // R = B
            rgbPtr[idx + 1] = bgrPtr[idx + 1];  // G = G
            rgbPtr[idx + 2] = bgrPtr[idx + 0];  // B = R
        }
    }
    
    return rgbImage;
}

VARP Diffusion::rgbToBgr(VARP rgbImage) {
    // RGB to BGR is the same operation as BGR to RGB (symmetric)
    return bgrToRgb(rgbImage);
}

VARP Diffusion::hwcToNchw(VARP hwcImage, bool normalize) {
    auto info = hwcImage->getInfo();
    if (!info || info->dim.size() != 3) {
        MNN_ERROR("hwcToNchw: Invalid input shape, expected [H, W, C]\n");
        return nullptr;
    }
    
    int H = info->dim[0];
    int W = info->dim[1];
    int C = info->dim[2];
    
    // Convert to float if needed
    VARP floatImage = hwcImage;
    if (info->type.code != halide_type_float) {
        floatImage = _Cast<float>(hwcImage);
    }
    
    // Apply normalization if requested: (x / 127.5) - 1.0
    if (normalize) {
        floatImage = floatImage * _Const(1.0f / 127.5f) - _Const(1.0f);
    }
    
    // Convert HWC to NCHW
    auto srcPtr = floatImage->readMap<float>();
    VARP nchwImage = _Input({1, C, H, W}, NCHW, halide_type_of<float>());
    auto dstPtr = nchwImage->writeMap<float>();
    
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int srcIdx = h * W * C + w * C + c;
                int dstIdx = c * H * W + h * W + w;
                dstPtr[dstIdx] = srcPtr[srcIdx];
            }
        }
    }
    
    return nchwImage;
}

VARP Diffusion::nchwToHwc(VARP nchwImage, bool denormalize) {
    auto info = nchwImage->getInfo();
    if (!info || info->dim.size() != 4) {
        MNN_ERROR("nchwToHwc: Invalid input shape, expected [N, C, H, W]\n");
        return nullptr;
    }
    
    int N = info->dim[0];
    int C = info->dim[1];
    int H = info->dim[2];
    int W = info->dim[3];
    
    if (N != 1) {
        MNN_ERROR("nchwToHwc: Only batch size 1 is supported, got %d\n", N);
        return nullptr;
    }
    
    // Apply denormalization if requested: (x + 1.0) * 127.5
    VARP floatImage = nchwImage;
    if (denormalize) {
        floatImage = (nchwImage + _Const(1.0f)) * _Const(127.5f);
    }
    
    // Convert NCHW to HWC
    auto srcPtr = floatImage->readMap<float>();
    VARP hwcImage = _Input({H, W, C}, NHWC, halide_type_of<uint8_t>());
    auto dstPtr = hwcImage->writeMap<uint8_t>();
    
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                int srcIdx = c * H * W + h * W + w;
                int dstIdx = h * W * C + w * C + c;
                float val = srcPtr[srcIdx];
                // Clamp to [0, 255]
                val = std::max(0.0f, std::min(255.0f, val));
                dstPtr[dstIdx] = static_cast<uint8_t>(val);
            }
        }
    }
    
    return hwcImage;
}

void Diffusion::packLatents(const float* src, float* dst, int B, int C, int H, int W, int seqOffset) {
    // Pack latents: [B, C, H, W] -> [B, H/2*W/2, C*4]
    // This implements the patchify operation for Flux-like models
    int packedH = H / 2;
    int packedW = W / 2;
    int packedC = C * 4;
    
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < packedH; ++h) {
            for (int w = 0; w < packedW; ++w) {
                int seqIdx = seqOffset + h * packedW + w;
                for (int c = 0; c < C; ++c) {
                    for (int dh = 0; dh < 2; ++dh) {
                        for (int dw = 0; dw < 2; ++dw) {
                            int srcH = h * 2 + dh;
                            int srcW = w * 2 + dw;
                            int srcIdx = b * C * H * W + c * H * W + srcH * W + srcW;
                            int dstC = c * 4 + dh * 2 + dw;
                            int dstIdx = b * (packedH * packedW + seqOffset) * packedC + seqIdx * packedC + dstC;
                            dst[dstIdx] = src[srcIdx];
                        }
                    }
                }
            }
        }
    }
}

void Diffusion::unpackLatents(const float* src, float* dst, int B, int C, int H, int W, int seqLen) {
    // CPU version: Unpack latents [B, seq, 64] -> [B, C, H, W]
    // This reverses the patchify operation
    // Note: This is kept for backward compatibility and debugging
    //       For production, use GPU version (unpackLatentsGPU) for better performance
    int packedH = H / 2;
    int packedW = W / 2;
    int packedC = C * 4;
    int expectedSeq = packedH * packedW;
    
    if (seqLen < expectedSeq) {
        MNN_ERROR("unpackLatents: seqLen=%d < expectedSeq=%d\n", seqLen, expectedSeq);
        return;
    }
    
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < packedH; ++h) {
            for (int w = 0; w < packedW; ++w) {
                int seqIdx = h * packedW + w;
                for (int c = 0; c < C; ++c) {
                    for (int dh = 0; dh < 2; ++dh) {
                        for (int dw = 0; dw < 2; ++dw) {
                            int srcC = c * 4 + dh * 2 + dw;
                            int srcIdx = b * seqLen * packedC + seqIdx * packedC + srcC;
                            int dstH = h * 2 + dh;
                            int dstW = w * 2 + dw;
                            int dstIdx = b * C * H * W + c * H * W + dstH * W + dstW;
                            dst[dstIdx] = src[srcIdx];
                        }
                    }
                }
            }
        }
    }
}

// ========================================================================
// GPU-side Unpack Function (Pure GPU, No CPU Sync)
// ========================================================================
// This function unpacks latents entirely on GPU using MNN Express API
// Input:  [B, seq, 64] - packed sequence format
// Output: [B, C, H, W] - standard NCHW latent format
//
// Algorithm:
//   1. Reshape [B, seq, 64] -> [B, H/2, W/2, C*4]
//   2. Reshape [B, H/2, W/2, C*4] -> [B, H/2, W/2, C, 2, 2]
//   3. Permute [B, H/2, W/2, C, 2, 2] -> [B, C, H/2, 2, W/2, 2]
//   4. Reshape [B, C, H/2, 2, W/2, 2] -> [B, C, H, W]
//
// Performance:
//   - Zero-copy: All operations are view changes (Reshape) or GPU kernels (Transpose)
//   - No GPU↔CPU synchronization
//   - ~10x faster than CPU version for large tensors
// ========================================================================
VARP Diffusion::unpackLatentsGPU(VARP packed, int B, int C, int H, int W) {
    int packedH = H / 2;
    int packedW = W / 2;
    int packedC = C * 4;  // 64 for LongCat (C=16)
    
    // Step 1: Reshape [B, seq, 64] -> [B, H/2, W/2, C*4]
    auto reshaped = _Reshape(packed, {B, packedH, packedW, packedC});
    
    // Step 2: Reshape [B, H/2, W/2, C*4] -> [B, H/2, W/2, C, 2, 2]
    auto view6d = _Reshape(reshaped, {B, packedH, packedW, C, 2, 2});
    
    // Step 3: Permute [B, H/2, W/2, C, 2, 2] -> [B, C, H/2, 2, W/2, 2]
    //         dims:    0   1     2     3  4  5  ->  0  3  1     4  2     5
    auto permuted = _Transpose(view6d, {0, 3, 1, 4, 2, 5});
    
    // Step 4: Reshape [B, C, H/2, 2, W/2, 2] -> [B, C, H, W]
    auto unpacked = _Reshape(permuted, {B, C, H, W});
    
    return unpacked;
}

// ========================================================================
// Scheduler Update Functions
// ========================================================================

// PLMS Update (SD1.5): 4-order predictor-corrector with noise history
// Formula: sample += dt * (55/24*n0 - 59/24*n1 + 37/24*n2 - 9/24*n3)
// Requires: noise history from previous steps
VARP Diffusion::applyPLMSUpdate(VARP sample, VARP noise_pred, float dt, int step) {
    // Maintain noise history (max 4 steps)
    mEts.push_back(noise_pred);
    if (mEts.size() > 4) {
        mEts.erase(mEts.begin());
    }
    
    // First 3 steps: use simple Euler update
    if (step < 3) {
        return sample + _Scalar<float>(dt) * noise_pred;
    }
    
    // Step 4+: use PLMS 4-order update
    auto n0 = mEts[3];  // Current
    auto n1 = mEts[2];  // Previous
    auto n2 = mEts[1];  // 2 steps ago
    auto n3 = mEts[0];  // 3 steps ago
    
    // PLMS formula: weighted combination of historical noise
    auto combined = _Scalar(55.0f/24.0f) * n0 
                  - _Scalar(59.0f/24.0f) * n1 
                  + _Scalar(37.0f/24.0f) * n2 
                  - _Scalar(9.0f/24.0f) * n3;
    
    return sample + _Scalar<float>(dt) * combined;
}

// Euler Update (Z-Image, LongCat): Simple 1-order forward Euler
// Formula: sample += dt * noise_pred
// This is the standard FlowMatch Euler scheduler
VARP Diffusion::applyEulerUpdate(VARP sample, VARP noise_pred, float dt) {
    return sample + _Scalar<float>(dt) * noise_pred;
}

// ========================================================================
// Debug Utility Functions
// ========================================================================

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

// Factory method with separate width and height for non-square aspect ratios
Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads) {
    Diffusion* diffusion = new Diffusion(modelPath, modelType, backendType, memoryMode, imageWidth, imageHeight, textEncoderOnCPU, gpuMemoryMode, precisionMode, numThreads);
    return diffusion;
}

// Full factory method with all options including VAE on CPU and CFG mode
Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, DiffusionCFGMode cfgMode, int numThreads) {
    Diffusion* diffusion = new Diffusion(modelPath, modelType, backendType, memoryMode, imageWidth, imageHeight, textEncoderOnCPU, gpuMemoryMode, precisionMode, numThreads);
    diffusion->mVaeOnCPU = vaeOnCPU;
    diffusion->mCFGMode = cfgMode;
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

// Constructor with separate width and height for non-square aspect ratios
Diffusion::Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads) :
mModelPath(modelPath), mModelType(modelType), mBackendType(backendType), mMemoryMode(memoryMode), mImageSize(0), mImageWidth(imageWidth), mImageHeight(imageHeight), mTextEncoderOnCPU(textEncoderOnCPU), mGpuMemoryMode(gpuMemoryMode), mPrecisionMode(precisionMode), mNumThreads(numThreads) {
    // Initialize based on model type
    if (modelType == STABLE_DIFFUSION_1_5) {
        mMaxTextLen = 77;
    } else if (modelType == STABLE_DIFFUSION_TAIYI_CHINESE) {
        mMaxTextLen = 512;
    } else if (modelType == STABLE_DIFFUSION_ZIMAGE || modelType == LONGCAT_IMAGE_EDIT) {
        mMaxTextLen = 128;
    }
    
    if (modelType == STABLE_DIFFUSION_ZIMAGE || modelType == LONGCAT_IMAGE_EDIT) {
        // Default FlowMatch Euler config
        mTrainTimestepsNum = 1000;
        mFlowShift = 3.0f;
        mUseDynamicShifting = false;
        
        // Load scheduler config if available
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
            }
        }
        
        // Load main config.json for LongCat
        if (modelType == LONGCAT_IMAGE_EDIT) {
            // Initialize LLM encoder config with LongCat defaults
            mLlmEncoderConfig = LLMEncoderConfig::longcat();
            
            std::string configPath = mModelPath + "/config.json";
            std::ifstream cfile(configPath.c_str());
            if (cfile.good()) {
                std::ostringstream oss;
                oss << cfile.rdbuf();
                cfile.close();
                std::string jsonStr = oss.str();
                rapidjson::Document doc;
                doc.Parse(jsonStr.c_str());
                if (!doc.HasParseError() && doc.IsObject()) {
                    // Get text_encoder directory
                    if (doc.HasMember("text_encoder") && doc["text_encoder"].IsObject()) {
                        auto& te = doc["text_encoder"];
                        if (te.HasMember("directory") && te["directory"].IsString()) {
                            mTextEncoderDir = te["directory"].GetString();
                        }
                    }
                    // Get inference config
                    if (doc.HasMember("inference") && doc["inference"].IsObject()) {
                        auto& inf = doc["inference"];
                        if (inf.HasMember("guidance_scale") && (inf["guidance_scale"].IsFloat() || inf["guidance_scale"].IsDouble())) {
                            mDefaultCfgScale = static_cast<float>(inf["guidance_scale"].GetDouble());
                        }
                    }
                    // Get LLM encoder config (optional overrides)
                    if (doc.HasMember("llm_encoder") && doc["llm_encoder"].IsObject()) {
                        auto& enc = doc["llm_encoder"];
                        if (enc.HasMember("prefix_len") && enc["prefix_len"].IsInt()) {
                            mLlmEncoderConfig.prefixLen = enc["prefix_len"].GetInt();
                        }
                        if (enc.HasMember("suffix_len") && enc["suffix_len"].IsInt()) {
                            mLlmEncoderConfig.suffixLen = enc["suffix_len"].GetInt();
                        }
                        if (enc.HasMember("target_seq_len") && enc["target_seq_len"].IsInt()) {
                            mLlmEncoderConfig.targetSeqLen = enc["target_seq_len"].GetInt();
                        }
                        if (enc.HasMember("vision_resize_size") && enc["vision_resize_size"].IsInt()) {
                            mLlmEncoderConfig.visionResizeSize = enc["vision_resize_size"].GetInt();
                        }
                        if (enc.HasMember("hidden_size") && enc["hidden_size"].IsInt()) {
                            mLlmEncoderConfig.hiddenSize = enc["hidden_size"].GetInt();
                        }
                    }
                    MNN_PRINT("[LongCat] Config loaded: text_encoder_dir=%s, llm_encoder(prefix=%d, suffix=%d, target=%d, resize=%d)\n", 
                              mTextEncoderDir.c_str(), mLlmEncoderConfig.prefixLen, mLlmEncoderConfig.suffixLen, 
                              mLlmEncoderConfig.targetSeqLen, mLlmEncoderConfig.visionResizeSize);
                }
            }
        }
    } else {
        // compute timesteps alphas for SD1.5/Taiyi (PNDM + PLMS)
        std::unique_ptr<Scheduler> scheduler;
        scheduler.reset(new PNDMScheduler);
        mAlphas = scheduler->get_alphas();
    }

    // Set latent dimensions based on width/height (non-square support)
    if (modelType == STABLE_DIFFUSION_ZIMAGE || modelType == LONGCAT_IMAGE_EDIT) {
        mLatentC = 16;
        if (mImageWidth > 0 && mImageHeight > 0) {
            // Ensure dimensions are multiples of 8
            int w = (mImageWidth / 8) * 8;
            int h = (mImageHeight / 8) * 8;
            if (w < 256) w = 256;
            if (h < 256) h = 256;
            if (w > 1280) w = 1280;
            if (h > 1280) h = 1280;
            mImageWidth = w;
            mImageHeight = h;
            mLatentW = w / 8;
            mLatentH = h / 8;
            MNN_PRINT("[ZIMAGE] latentScale=8, imageSize(user=%dx%d), latent=(1,%d,%d,%d)\n", mImageWidth, mImageHeight, mLatentC, mLatentH, mLatentW);
        } else {
            // Default to 1024x1024
            mLatentH = 128;
            mLatentW = 128;
            mImageWidth = 1024;
            mImageHeight = 1024;
            MNN_PRINT("[ZIMAGE] latentScale=8, imageSize(default=1024x1024), latent=(1,%d,%d,%d)\n", mLatentC, mLatentH, mLatentW);
        }
    }
}

Diffusion::Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageSize, bool textEncoderOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, int numThreads) :
mModelPath(modelPath), mModelType(modelType), mBackendType(backendType), mMemoryMode(memoryMode), mImageSize(imageSize), mTextEncoderOnCPU(textEncoderOnCPU), mGpuMemoryMode(gpuMemoryMode), mPrecisionMode(precisionMode), mNumThreads(numThreads) {
    if (modelType == STABLE_DIFFUSION_1_5) {
        mMaxTextLen = 77;
    } else if (modelType == STABLE_DIFFUSION_TAIYI_CHINESE) {
        mMaxTextLen = 512;
    } else if (modelType == STABLE_DIFFUSION_ZIMAGE || modelType == LONGCAT_IMAGE_EDIT) {
        // ZImage pipeline typically uses a longer context; align with Python side (e.g. max_length=128).
        mMaxTextLen = 128;
    }
    if (modelType == STABLE_DIFFUSION_ZIMAGE || modelType == LONGCAT_IMAGE_EDIT) {
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

    if (modelType == STABLE_DIFFUSION_ZIMAGE || modelType == LONGCAT_IMAGE_EDIT) {
        // Keep legacy behavior when user does not override image size.
        // This must stay consistent with the previous implementation (no imageSize option).
        if (mImageSize <= 0) {
            mLatentC = 16;
            mLatentH = 128;
            mLatentW = 128;
            mImageWidth = 1024;
            mImageHeight = 1024;
            MNN_PRINT("[ZIMAGE] latentScale=8, imageSize(user=default), latent=(1,%d,%d,%d)\n", mLatentC, mLatentH, mLatentW);
        } else {
            int normalizedSize = mImageSize;
            if (normalizedSize != 512 && normalizedSize != 640 && normalizedSize != 768 && normalizedSize != 896 && normalizedSize != 1024) {
                normalizedSize = 1024;
            }
            mImageSize = normalizedSize;
            mImageWidth = normalizedSize;
            mImageHeight = normalizedSize;
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
#ifdef MNN_BUILD_LLM
    if (mLlm) {
        delete static_cast<MNN::Transformer::Llm*>(mLlm);
        mLlm = nullptr;
    }
#endif
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
            // User explicitly specified BUFFER mode
            gpuMode |= MNN_GPU_MEMORY_BUFFER;
            MNN_PRINT("[DIFFUSION] OpenCL GPU Memory Mode: BUFFER (user specified)\n");
        } else if (mGpuMemoryMode == GPU_MEMORY_IMAGE) {
            // User explicitly specified IMAGE mode
            gpuMode |= MNN_GPU_MEMORY_IMAGE;
            MNN_PRINT("[DIFFUSION] OpenCL GPU Memory Mode: IMAGE (user specified)\n");
        } else {
            // AUTO mode (GPU_MEMORY_AUTO=0): Use BUFFER for all models
            // LongCat: BUFFER mode (IMAGE mode causes CL_MEM_OBJECT_ALLOCATION_FAILURE for large tensors)
            // SD1.5: BUFFER mode (FmhaV2 operator only supported in BUFFER mode)
            // ZImage: BUFFER mode (works fine)
            gpuMode |= MNN_GPU_MEMORY_BUFFER;
            MNN_PRINT("[DIFFUSION] OpenCL GPU Memory Mode: BUFFER (auto-selected)\n");
        }
        config.mode = gpuMode;
    } else {
        config.numThread = 1;
    }
    // Always use Memory_Low to ensure memory is released immediately after use
    // This is critical for mobile devices and prevents memory accumulation
    // Model-level memory management (load/unload) is controlled by mMemoryMode separately
    backendConfig.memory = BackendConfig::Memory_Low;
    
    // Configure precision based on user setting or auto-detect
    if (mPrecisionMode == PRECISION_LOW) {
        backendConfig.precision = BackendConfig::Precision_Low;
    } else if (mPrecisionMode == PRECISION_NORMAL) {
        backendConfig.precision = BackendConfig::Precision_Normal;
    } else if (mPrecisionMode == PRECISION_HIGH) {
        backendConfig.precision = BackendConfig::Precision_High;
    } else {
        // AUTO: Select precision based on model type and backend
        // Strategy:
        // - ZImage + GPU (OpenCL/Vulkan): FP32 (PRECISION_HIGH) - requires FP32 for -inf handling in attention mask
        // - SD1.5/Taiyi + any backend: FP16 (PRECISION_LOW) - works fine with FP16
        if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
            // ZImage requires FP32 for GPU (Adreno GPU -inf handling)
            if (config.type == MNN_FORWARD_OPENCL || config.type == MNN_FORWARD_VULKAN) {
                backendConfig.precision = BackendConfig::Precision_High;  // FP32
            } else {
                backendConfig.precision = BackendConfig::Precision_Normal;  // FP32 for CPU
            }
        } else {
            // SD1.5/Taiyi: FP16 works fine on all backends
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
    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
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
    // NOTE: RuntimeManager holds OpenCL context, command queue, and kernel cache (~100-200MB)
    // In Memory_Low mode, we only unload Modules but keep RuntimeManager alive
    // This is acceptable for most devices, but on very low memory devices (<2GB RAM),
    // consider also resetting runtime_manager_ when unloading models
    // need to consider memory
    if(mMemoryMode == 0) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 0);
    } else if(mMemoryMode == 2) {
        runtime_manager_->setHint(Interpreter::WINOGRAD_MEMORY_LEVEL, 1);
    }
    if(config.type == MNN_FORWARD_CPU) {
        if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
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
        if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
            runtime_manager_cpu_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 0);
        } else {
            runtime_manager_cpu_->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 2);
        }
        MNN_PRINT("[DIFFUSION] text_encoder forced on CPU (mTextEncoderOnCPU=true)\n");
    } else {
        MNN_PRINT("[DIFFUSION] text_encoder on same backend as UNet (mTextEncoderOnCPU=false)\n");
    }
    mLatentVar = _Input({1, mLatentC, mLatentH, mLatentW}, NCHW, halide_type_of<float>());
    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
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
    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
        mAttentionMaskVar->writeMap<int>();
    }
    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
        mTimestepVar->writeMap<float>();
    } else {
        mTimestepVar->writeMap<int>();
    }

    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
        // ZImage uses single batch UNet without classifier-free guidance.
        // Create a separate buffer for mSampleVar to avoid sharing with mLatentVar
        // This allows in-place updates during UNet loop without affecting mLatentVar
        mSampleVar = _Input({1, mLatentC, mLatentH, mLatentW}, NCHW, halide_type_of<float>());
        mSampleVar->writeMap<float>();  // Allocate buffer
    } else {
        mSampleVar = _Concat({mLatentVar, mLatentVar}, 0);
    }

    if(mMemoryMode > 0) {
        MNN_PRINT("First time initilizing may cost a few seconds to create cachefile, please wait ...\n");
    }

    // Load model paths from config.json (with fallback to hardcoded defaults)
    DiffusionConfig diff_config(mModelPath);

    VARP text_embeddings;
    // LongCat needs 4 modules: text_encoder, unet, vae_decoder, vae_encoder
    if (mModelType == LONGCAT_IMAGE_EDIT) {
        mModules.resize(4);
    } else {
        mModules.resize(3);
    }
    // load text_encoder model
    // LongCat uses external text_encoder_demo tool, skip loading text_encoder.mnn
    if (mModelType != LONGCAT_IMAGE_EDIT) {
        std::string model_path = diff_config.text_encoder_model();
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
    } else {
        MNN_PRINT("[LongCat] Skipping text_encoder.mnn (uses integrated LLM text encoder)\n");
    }
    // load unet model
    {
        std::string model_path = diff_config.unet_model();
        MNN_PRINT("Load %s\n", model_path.c_str());
        if (mModelType == LONGCAT_IMAGE_EDIT) {
            // LongCat UNet requires additional txt_ids and img_ids inputs
            mModules[1].reset(Module::load(
                                           {"sample", "timestep", "encoder_hidden_states", "txt_ids", "img_ids"}, {"out_sample"}, model_path.c_str(), runtime_manager_, &module_config));
        } else {
            mModules[1].reset(Module::load(
                                           {"sample", "timestep", "encoder_hidden_states"}, {"out_sample"}, model_path.c_str(), runtime_manager_, &module_config));
        }
    }
    // load vae_decoder model
    {
        std::string model_path = diff_config.vae_decoder_model();
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[2].reset(Module::load(
                                       {"latent_sample"}, {"sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }
    // load vae_encoder model (for LongCat image-to-image)
    if (mModelType == LONGCAT_IMAGE_EDIT) {
        std::string model_path = diff_config.vae_encoder_model();
        MNN_PRINT("Load %s\n", model_path.c_str());
        mModules[3].reset(Module::load(
                                       {"sample"}, {"latent_sample"}, model_path.c_str(), runtime_manager_, &module_config));
    }

    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
        log_module_io_mnn("ZIMAGE_text_encoder", mModules[0]);
        log_module_io_mnn("ZIMAGE_unet", mModules[1]);
        log_module_io_mnn("ZIMAGE_vae_decoder", mModules[2]);
    }

    // tokenizer loading
    // LongCat uses external text_encoder_demo, no tokenizer needed here
    if(mModelType == STABLE_DIFFUSION_1_5) {
        mTokenizer.reset(new CLIPTokenizer);
    } else if(mModelType == STABLE_DIFFUSION_TAIYI_CHINESE) {
        mTokenizer.reset(new BertTokenizer);
    } else if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
#ifdef MNN_BUILD_LLM
        mTokenizer.reset(new LlmTokenizerWrapper);
#else
        MNN_PRINT("Error: STABLE_DIFFUSION_ZIMAGE requires MNN_BUILD_LLM enabled.\n");
        return false;
#endif
    } else if (mModelType == LONGCAT_IMAGE_EDIT) {
        MNN_PRINT("[LongCat] Skipping tokenizer (uses integrated LLM text encoder)\n");
    }
    if (!mTokenizer && mModelType != LONGCAT_IMAGE_EDIT) {
        MNN_PRINT("Error: tokenizer not initialized for model type %d.\n", (int)mModelType);
        return false;
    }
    // LongCat doesn't need tokenizer
    if (mTokenizer && !mTokenizer->load(mModelPath)) {
        MNN_PRINT("Error: failed to load tokenizer from path %s.\n", mModelPath.c_str());
        return false;
    }
    
    // Resize fix
    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
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
    // text encoder warmup (skip for ZIMAGE and LongCat)
    {
        if (mModelType != STABLE_DIFFUSION_ZIMAGE && mModelType != LONGCAT_IMAGE_EDIT && mModules[0]) {
            auto outputs = mModules[0]->onForward({mPromptVar});
            text_embeddings = _Convert(outputs[0], NCHW);
        }
    }
    
    if(mMemoryMode > 0) {
        // Skip UNet warmup for ZIMAGE and LongCat (dynamic shapes, no text_embeddings yet)
        if (mModelType != STABLE_DIFFUSION_ZIMAGE && mModelType != LONGCAT_IMAGE_EDIT) {
            // unet
            {
                std::vector<VARP> unet_inputs = {mSampleVar, mTimestepVar, text_embeddings};
                auto outputs = mModules[1]->onForward(unet_inputs);
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
    
    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
        int expected = mMaxTextLen * 2;
        // Warning only when debug enabled
        if (diffusion_debug_enabled() && (int)ids.size() < expected) {
            MNN_PRINT("Warning: ZIMAGE tokenizer ids size %d smaller than expected %d\n", (int)ids.size(), expected);
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
        auto outputs = mModules[0]->onForward({mPromptVar, mAttentionMaskVar});
        auto output  = outputs[0];  // keep original [B, L, D] layout for UNet
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

#ifdef MNN_BUILD_LLM
VARP Diffusion::text_encoder_llm(const std::string& prompt, VARP preprocessedImage) {
    AUTOTIME;
    using namespace MNN::Transformer;
    
    // Use model-specific LLM encoder config
    const auto& cfg = mLlmEncoderConfig;
    
    // Determine mode: T2I (no image) vs Image Edit (with image)
    bool isT2IMode = !preprocessedImage.get();
    
    // Lazy load LLM
    if (!mLlm) {
        std::string configPath = mModelPath + "/" + mTextEncoderDir + "/config.json";
        
        // Step 1: Create LLM (loads config.json)
        mLlm = Llm::createLLM(configPath);
        if (!mLlm) {
            MNN_PRINT("Error: Failed to create LLM from %s\n", configPath.c_str());
            return nullptr;
        }
        
        // Step 2: Override mllm backend_type using set_config
        auto llm = static_cast<Llm*>(mLlm);
        const char* targetBackend = mTextEncoderOnCPU ? "cpu" : 
                                   (mBackendType == MNN_FORWARD_OPENCL ? "opencl" :
                                    mBackendType == MNN_FORWARD_VULKAN ? "vulkan" : "cpu");
        
        // Use set_config to override mllm.backend_type
        std::string overrideConfig = "{\"mllm\": {\"backend_type\": \"" + std::string(targetBackend) + "\"}}";
        llm->set_config(overrideConfig);
        
        if (mTextEncoderOnCPU) {
            MNN_PRINT("[LongCat] LLM backend forced to CPU (te_on_cpu=1, ignoring global backend)\n");
        } else {
            MNN_PRINT("[LongCat] LLM backend set to %s (following global backend_type)\n", targetBackend);
        }
        
        llm->load();
        MNN_PRINT("[LongCat] LLM text encoder loaded\n");
    }
    
    auto llm = static_cast<Llm*>(mLlm);
    
    VARP image = nullptr;
    
    if (!isT2IMode && preprocessedImage.get()) {
        // Image Edit mode: Use VAE preprocessed image and resize to 512x512
        // preprocessedImage is already cropped and converted to RGB by VAE preprocessing
        auto imgInfo = preprocessedImage->getInfo();
        int imgH = imgInfo->dim[0];
        int imgW = imgInfo->dim[1];
        int imgC = imgInfo->dim[2];
        MNN_PRINT("[LongCat] TE: Using VAE preprocessed image: %dx%d, channels=%d (RGB)\n", imgW, imgH, imgC);
        
        // Resize to visual encoder input size (512x512)
        const int visionSize = cfg.visionResizeSize;
        if (imgW != visionSize || imgH != visionSize) {
            image = CV::resize(preprocessedImage, {visionSize, visionSize}, 0, 0, CV::INTER_LINEAR, -1, {}, {});
            MNN_PRINT("[LongCat] TE: Resized to vision size: %dx%d -> %dx%d\n", imgW, imgH, visionSize, visionSize);
        } else {
            image = preprocessedImage;
            MNN_PRINT("[LongCat] TE: Image already at vision size: %dx%d\n", visionSize, visionSize);
        }
        
        // No need to convert BGR to RGB - already RGB from VAE preprocessing
        MNN_PRINT("[LongCat] TE: Using RGB image (no conversion needed)\n");
    } else {
        MNN_PRINT("[LongCat] T2I mode: text-only encoding (no image)\n");
    }
    
    // Build prompt using LLM's apply_chat_template (object-oriented, model-agnostic)
    // The chat_template is loaded from llm_config.json automatically
    std::string systemPrompt;
    std::string userContent;
    
    if (isT2IMode) {
        // T2I mode: use image captioning expert system prompt (from pipeline_longcat_image.py)
        systemPrompt = "As an image captioning expert, generate a descriptive text prompt based on an image content, "
                       "suitable for input to a text-to-image model.";
        userContent = prompt;  // Pure text, no image tags
    } else {
        // Image Edit mode: use image editing expert system prompt
        systemPrompt = "As an image editing expert, first analyze the content and attributes of the input image(s). "
                       "Then, based on the user's editing instructions, clearly and precisely determine how to modify "
                       "the given image(s), ensuring that only the specified parts are altered and all other aspects "
                       "remain consistent with the original(s).";
        userContent = "<|vision_start|><img>input</img><|vision_end|>" + prompt;
    }
    
    ChatMessages chatMessages;
    chatMessages.push_back({"system", systemPrompt});
    chatMessages.push_back({"user", userContent});
    std::string promptTemplate = llm->apply_chat_template(chatMessages);
    
    MNN_PRINT("[LongCat] Prompt template length: %d\n", (int)promptTemplate.size());
    
    // Create input (multimodal for Image Edit, text-only for T2I)
    MultimodalPrompt multimodalInput;
    multimodalInput.prompt_template = promptTemplate;
    
    if (!isT2IMode && image.get()) {
        auto imgInfo = image->getInfo();
        PromptImagePart imagePart;
        imagePart.image_data = image;
        imagePart.height = imgInfo->dim[0];
        imagePart.width = imgInfo->dim[1];
        multimodalInput.images["input"] = imagePart;
    }
    
    // Tokenize
    auto inputIds = llm->tokenizer_encode(multimodalInput);
    MNN_PRINT("[LongCat] Tokenized: %d tokens\n", (int)inputIds.size());
    
    // Forward to get hidden states
    llm->generate_init();
    llm->forward(inputIds);
    
    // Get hidden states from outputs
    auto outputs = llm->getOutputs();
    MNN_PRINT("[LongCat] LLM outputs count: %d\n", (int)outputs.size());
    
    VARP hiddenStates = nullptr;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto info = outputs[i]->getInfo();
        // Hidden states has shape [1, seq_len, hidden_size]
        if (info->dim.size() == 3 && info->dim[2] == cfg.hiddenSize) {
            hiddenStates = outputs[i];
            MNN_PRINT("[LongCat] Found hidden_states at output[%d]\n", (int)i);
            break;
        }
    }
    
    if (!hiddenStates.get()) {
        MNN_PRINT("Error: Failed to find hidden_states in LLM outputs\n");
        return nullptr;
    }
    
    // Slice hidden_states: remove prefix and suffix tokens
    // For T2I mode, we need different prefix/suffix since there's no image tokens
    // T2I prefix: "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n" 
    // T2I suffix: "<|im_end|>\n<|im_start|>assistant\n"
    // Image Edit prefix includes vision tokens, so it's longer (67 tokens)
    auto hsInfo = hiddenStates->getInfo();
    int seqLen = hsInfo->dim[1];
    int hiddenSize = hsInfo->dim[2];
    
    int prefixLen, suffixLen;
    if (isT2IMode) {
        // T2I mode: dynamically calculate prefix/suffix based on template
        // The prefix is: system prompt template tokens
        // The suffix is: "<|im_end|>\n<|im_start|>assistant\n" = 5 tokens (same as Image Edit)
        // For T2I, we tokenize the prefix template to get exact length
        std::string t2iPrefixTemplate = "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n<|im_start|>user\n";
        auto prefixIds = llm->tokenizer_encode(t2iPrefixTemplate);
        prefixLen = (int)prefixIds.size();
        suffixLen = cfg.suffixLen;  // suffix is the same: 5 tokens
        MNN_PRINT("[LongCat] T2I mode: calculated prefix=%d tokens\n", prefixLen);
    } else {
        // Image Edit mode: use config values
        prefixLen = cfg.prefixLen;
        suffixLen = cfg.suffixLen;
    }
    
    int sliceStart = prefixLen;
    int sliceEnd = seqLen - suffixLen;
    int outputSeqLen = sliceEnd - sliceStart;
    
    MNN_PRINT("[LongCat] Slicing: [%d:%d] -> %d tokens (prefix=%d, suffix=%d)\n", 
              sliceStart, sliceEnd, outputSeqLen, prefixLen, suffixLen);
    
    if (outputSeqLen <= 0) {
        MNN_PRINT("Error: Invalid slice range for hidden_states\n");
        return nullptr;
    }
    
    // Slice using gather
    std::vector<int> indices;
    for (int i = sliceStart; i < sliceEnd; ++i) {
        indices.push_back(i);
    }
    auto indicesVar = _Const(indices.data(), {outputSeqLen}, NCHW, halide_type_of<int>());
    auto sliced = _GatherV2(hiddenStates, indicesVar, _Scalar<int>(1));
    
    // Pad to target sequence length
    // T2I mode: pad to tokenizerMaxLength (512)
    // Image Edit mode: pad to targetSeqLen (838)
    auto slicedInfo = sliced->getInfo();
    int currentLen = slicedInfo->dim[1];
    int targetLen = isT2IMode ? cfg.tokenizerMaxLength : cfg.targetSeqLen;
    
    if (currentLen < targetLen) {
        int padLen = targetLen - currentLen;
        MNN_PRINT("[LongCat] %s: Padding %d -> %d (+%d zeros)\n", 
                  isT2IMode ? "T2I" : "Image Edit", currentLen, targetLen, padLen);
        std::vector<int> padShape = {1, padLen, hiddenSize};
        auto padding = _Fill(_Const(padShape.data(), {3}, NCHW, halide_type_of<int>()), _Scalar<float>(0.0f));
        sliced = _Concat({sliced, padding}, 1);
    } else {
        MNN_PRINT("[LongCat] %s: Output tokens: %d (no padding needed)\n", 
                  isT2IMode ? "T2I" : "Image Edit", currentLen);
    }
    
    sliced.fix(VARP::CONSTANT);
    
    auto finalInfo = sliced->getInfo();
    MNN_PRINT("[LongCat] Hidden states shape: [%d, %d, %d]\n", 
              finalInfo->dim[0], finalInfo->dim[1], finalInfo->dim[2]);
    
    // Save text encoder output for debugging
    return sliced;
}
#else
VARP Diffusion::text_encoder_llm(const std::string& prompt, VARP preprocessedImage) {
    MNN_PRINT("Error: LongCat requires MNN_BUILD_LLM enabled\n");
    return nullptr;
}
#endif

void Diffusion::getCFGSigmaRange(float& sigmaLow, float& sigmaHigh) const {
    // CFG sigma range based on mode (for dual-UNet models like LongCat)
    switch (mCFGMode) {
        case CFG_MODE_WIDE:
            sigmaLow = 0.1f; sigmaHigh = 0.9f;
            break;
        case CFG_MODE_STANDARD:
            sigmaLow = 0.1f; sigmaHigh = 0.8f;
            break;
        case CFG_MODE_MEDIUM:
            sigmaLow = 0.15f; sigmaHigh = 0.7f;
            break;
        case CFG_MODE_NARROW:
            sigmaLow = 0.2f; sigmaHigh = 0.6f;
            break;
        case CFG_MODE_MINIMAL:
            sigmaLow = 0.25f; sigmaHigh = 0.5f;
            break;
        case CFG_MODE_AUTO:
        default:
            // Model-specific defaults
            if (mModelType == LONGCAT_IMAGE_EDIT) {
                sigmaLow = 0.1f; sigmaHigh = 0.8f;
            } else {
                sigmaLow = 0.0f; sigmaHigh = 1.0f;  // Full range for other models
            }
            break;
    }
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
    // Unload text_encoder after encoding (memory_mode != 1)
    // This applies to all pipelines: SD, ZImage, LongCat, etc.
    if (mMemoryMode != 1 && mModules[0]) {
        mModules[0].reset();
        MNN_PRINT("[DIFFUSION] text_encoder unloaded (memory_mode=%d)\n", mMemoryMode);
    }
    
    // For LongCat: explicitly unload LLM text encoder to free memory
    // LLM is managed by mLlm pointer, not mModules[0]
#ifdef MNN_BUILD_LLM
    if (mMemoryMode != 1 && mLlm && mModelType == LONGCAT_IMAGE_EDIT) {
        delete static_cast<MNN::Transformer::Llm*>(mLlm);
        mLlm = nullptr;
        MNN_PRINT("[LongCat] LLM text encoder unloaded (memory_mode=%d)\n", mMemoryMode);
    }
#endif
    // Use size_t to avoid uint32 overflow for large images (e.g., 2048x2048+)
    size_t latentSize = (size_t)mLatentC * (size_t)mLatentH * (size_t)mLatentW;
    if (mInitNoise.size() != latentSize) {
        mInitNoise.resize(latentSize);
    }
    
    // Generate random noise using Philox RNG (aligned with PyTorch)
    int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
    PhiloxRNG rng(seed);
    
    for (size_t i = 0; i < latentSize; i++) {
        mInitNoise[i] = rng.randn();
    }
    
    memcpy((void *)mLatentVar->writeMap<float>(), mInitNoise.data(), latentSize * sizeof(float));
    
    // CFG scale variable - use user-provided cfgScale instead of hardcoded 7.5
    VARP scalevar;
    if (mModelType != STABLE_DIFFUSION_ZIMAGE) {
        scalevar = _Input({1}, NCHW, halide_type_of<float>());
        auto scaleptr = scalevar->writeMap<float>();
        scaleptr[0] = cfgScale;  // Use user-provided CFG scale
    }
    
    auto floatVar = _Input({1}, NCHW, halide_type_of<float>());
    auto ptr = floatVar->writeMap<float>();
    
    // Create a separate buffer for plms to allow in-place updates
    // Copy initial noise from mLatentVar using MNN's input() method
    VARP plms;
    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
        plms = _Input({1, mLatentC, mLatentH, mLatentW}, NCHW, halide_type_of<float>());
        plms->input(mLatentVar);  // GPU-side copy, more efficient than readMap/writeMap
    } else {
        plms = mLatentVar;
    }
    
    // Initialize txt_ids and img_ids for models that need them (LongCat, future edit models)
    // txt_ids: [text_seq, 3] - (modality=0, idx, idx) for text
    // img_ids: [img_seq, 3] - (modality=1, row, col) for noise latents, (modality=2, row, col) for image latents
    // T2I mode: only noise latents (modality=1), no image latents
    // Image Edit mode: noise latents (modality=1) + image latents (modality=2)
    bool hasImageLatents = (mImageLatentsVar.get() != nullptr);
    bool isT2IMode = !hasImageLatents;  // T2I when no image latents provided
    
    if (mModelType == LONGCAT_IMAGE_EDIT) {
        // text_embeddings is already padded by text_encoder_llm:
        // - T2I mode: padded to tokenizerMaxLength (512)
        // - Image Edit mode: padded to targetSeqLen (838)
        int textSeqLen = text_embeddings->getInfo()->dim[1];
        int imgH = mLatentH / 2;  // Packed latent height (32 for 512x512)
        int imgW = mLatentW / 2;  // Packed latent width (32 for 512x512)
        int singleSeq = imgH * imgW;  // 1024 for 512x512
        
        // T2I mode: only noise latents; Image Edit mode: noise + image latents
        int imgSeqLen = isT2IMode ? singleSeq : (singleSeq * 2);
        
        MNN_PRINT("[LongCat] %s mode: textSeqLen=%d, imgSeqLen=%d\n", 
                  isT2IMode ? "T2I" : "Image Edit", textSeqLen, imgSeqLen);
        
        // txt_ids: [textSeqLen, 3] - (modality=0, idx, idx) for text tokens
        // Python: txt_ids[i] = [0, i, i] for i in range(seq_len)
        mTxtIdsVar = _Input({textSeqLen, 3}, NCHW, halide_type_of<float>());
        auto txtIdsPtr = mTxtIdsVar->writeMap<float>();
        for (int i = 0; i < textSeqLen; ++i) {
            txtIdsPtr[i * 3 + 0] = 0.0f;  // modality=0 for text
            txtIdsPtr[i * 3 + 1] = static_cast<float>(i);  // row index
            txtIdsPtr[i * 3 + 2] = static_cast<float>(i);  // col index
        }
        MNN_PRINT("[LongCat] txt_ids: first=[%.0f, %.0f, %.0f], last=[%.0f, %.0f, %.0f]\n",
                  txtIdsPtr[0], txtIdsPtr[1], txtIdsPtr[2],
                  txtIdsPtr[(textSeqLen-1)*3], txtIdsPtr[(textSeqLen-1)*3+1], txtIdsPtr[(textSeqLen-1)*3+2]);
        
        // img_ids: [imgSeqLen, 3] with (modality, row, col)
        // T2I mode: start=(tokenizer_max_length, tokenizer_max_length) per Python pipeline
        // Image Edit mode: start=(prompt_embeds_length, prompt_embeds_length) = (textSeqLen, textSeqLen)
        mImgIdsVar = _Input({imgSeqLen, 3}, NCHW, halide_type_of<float>());
        auto imgIdsPtr = mImgIdsVar->writeMap<float>();
        
        // T2I uses tokenizer_max_length, Image Edit uses text_seq_len
        int startOffset = isT2IMode ? mLlmEncoderConfig.tokenizerMaxLength : textSeqLen;
        MNN_PRINT("[LongCat] img_ids start offset: %d (%s)\n", startOffset, isT2IMode ? "T2I: tokenizer_max_length" : "Image Edit: text_seq_len");
        
        if (isT2IMode) {
            // T2I mode: only noise latents (modality=1)
            for (int h = 0; h < imgH; ++h) {
                for (int w = 0; w < imgW; ++w) {
                    int idx = h * imgW + w;
                    imgIdsPtr[idx * 3 + 0] = 1.0f;  // modality=1 for noise
                    imgIdsPtr[idx * 3 + 1] = static_cast<float>(startOffset + h);
                    imgIdsPtr[idx * 3 + 2] = static_cast<float>(startOffset + w);
                }
            }
        } else {
            // Image Edit mode: noise latents (modality=1) + image latents (modality=2)
            for (int half = 0; half < 2; ++half) {
                float modality = (half == 0) ? 1.0f : 2.0f;  // noise=1, image=2
                for (int h = 0; h < imgH; ++h) {
                    for (int w = 0; w < imgW; ++w) {
                        int idx = half * singleSeq + h * imgW + w;
                        imgIdsPtr[idx * 3 + 0] = modality;
                        imgIdsPtr[idx * 3 + 1] = static_cast<float>(startOffset + h);
                        imgIdsPtr[idx * 3 + 2] = static_cast<float>(startOffset + w);
                    }
                }
            }
        }
        
        MNN_PRINT("[LongCat] img_ids: first=[%.0f, %.0f, %.0f] (with offset=%d)\n", 
                  imgIdsPtr[0], imgIdsPtr[1], imgIdsPtr[2], startOffset);
        
        MNN_PRINT("[LongCat] txt_ids: [%d, 3], img_ids: [%d, 3]\n", textSeqLen, imgSeqLen);
    }
    
    // Pre-create zero_embeddings for CFG mode to avoid OpenCL kernel compilation issues
    // when dynamically creating tensors in the loop (NVIDIA OpenCL err:-9999)
    // Use text_embeddings * 0 to ensure same tensor properties for OpenCL kernel reuse
    // This optimization applies to all models that use CFG (SD1.5, Z-image, LongCat)
    VARP zero_embeddings;
    if (std::abs(cfgScale - 1.0f) > 0.001f) {
        zero_embeddings = text_embeddings * _Scalar<float>(0.0f);
        zero_embeddings.fix(VARP::CONSTANT);
        if (diffusion_debug_enabled()) {
            auto text_info = text_embeddings->getInfo();
            MNN_PRINT("[DIFFUSION] Pre-created zero_embeddings for CFG: [%d, %d, %d]\n", 
                      text_info->dim[0], text_info->dim[1], text_info->dim[2]);
        }
    }
    
    // ========================================================================
    // Unified Scheduler API Initialization
    // ========================================================================
    // This two-layer design allows easy extension for new models and schedulers:
    //   Layer 1: UNet Output Preprocessing (model-specific format conversion)
    //   Layer 2: Scheduler Update (scheduler-specific update logic)
    //
    // To add a new model (e.g., Flux):
    //   1. Define preprocessing in Layer 1 to convert UNet output to [B, C, H, W]
    //   2. Select appropriate scheduler in Layer 2 (PLMS or Euler)
    //   3. No changes needed in the main loop!
    // ========================================================================
    
    // ========== Layer 1: Initialize UNet Output Preprocessing ==========
    // Converts model-specific UNet output format to standard [B, C, H, W]
    
    if (mModelType == LONGCAT_IMAGE_EDIT) {
        // LongCat: UNet outputs [B, 2*seq, 64] (noise + image latents)
        // Need to: 1) slice first half (noise), 2) unpack to [B, C, H, W]
        
        int packedSeq = (mLatentH / 2) * (mLatentW / 2);  // 1024 for 64x64
        
        // Pre-create slice parameters (only once, reused in loop)
        std::vector<int> starts_data = {0, 0, 0};
        std::vector<int> sizes_data = {1, packedSeq, 64};
        auto slice_starts = _Const(starts_data.data(), {3}, NCHW, halide_type_of<int>());
        auto slice_sizes = _Const(sizes_data.data(), {3}, NCHW, halide_type_of<int>());
        slice_starts.fix(VARP::CONSTANT);
        slice_sizes.fix(VARP::CONSTANT);
        
        // Capture parameters for lambda
        int C = mLatentC, H = mLatentH, W = mLatentW;
        
        // Define preprocessing: GPU slice + unpack
        // Note: This creates new VARP each call, but unpack is only called once per step
        // The step time increase is caused by CPU pack (readMap), not unpack
        mUNetPreprocess = [this, slice_starts, slice_sizes, C, H, W](VARP unet_output) -> VARP {
            // Step 1: Extract noise part [B, 2*seq, 64] -> [B, seq, 64]
            auto noise_part = _Slice(unet_output, slice_starts, slice_sizes);
            
            // Step 2: Unpack [B, seq, 64] -> [B, C, H, W]
            return this->unpackLatentsGPU(noise_part, 1, C, H, W);
        };
        
        if (diffusion_debug_enabled()) {
            MNN_PRINT("[SCHEDULER] Layer 1: LongCat UNet preprocessor initialized (GPU slice+unpack)\n");
            MNN_PRINT("[SCHEDULER]   - Slice params: starts=[0,0,0], sizes=[1,%d,64]\n", packedSeq);
            MNN_PRINT("[SCHEDULER]   - Unpack params: C=%d, H=%d, W=%d\n", C, H, W);
        }
        
    } else {
        // SD1.5, Z-Image: UNet already outputs [B, C, H, W], no preprocessing needed
        mUNetPreprocess = [](VARP unet_output) -> VARP {
            return unet_output;  // Pass through
        };
        
        if (diffusion_debug_enabled()) {
            MNN_PRINT("[SCHEDULER] Layer 1: Standard UNet preprocessor (no preprocessing)\n");
        }
    }
    
    // ========== Layer 2: Initialize Scheduler Type ==========
    // Select scheduler based on model type
    
    if (mModelType == STABLE_DIFFUSION_1_5) {
        mSchedulerType = SCHEDULER_PLMS;
        if (diffusion_debug_enabled()) {
            MNN_PRINT("[SCHEDULER] Layer 2: PLMS scheduler selected (4-order with history)\n");
        }
    } else if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
        mSchedulerType = SCHEDULER_EULER;
        if (diffusion_debug_enabled()) {
            MNN_PRINT("[SCHEDULER] Layer 2: Euler scheduler selected (FlowMatch 1-order)\n");
        }
    }
    
    // ========== Extension Guide ==========
    // To add a new model (e.g., Flux):
    //
    // 1. Add model type to DiffusionModelType enum in diffusion.hpp:
    //    FLUX_SCHNELL = 4
    //
    // 2. Define Layer 1 preprocessing here:
    //    if (mModelType == FLUX_SCHNELL) {
    //        mUNetPreprocess = [](VARP unet_output) -> VARP {
    //            // Flux-specific transformation
    //            return flux_transform(unet_output);
    //        };
    //    }
    //
    // 3. Select Layer 2 scheduler:
    //    if (mModelType == FLUX_SCHNELL) {
    //        mSchedulerType = SCHEDULER_EULER;  // or SCHEDULER_PLMS
    //    }
    //
    // 4. Done! The main loop will automatically use your preprocessing and scheduler.
    // ========================================================================
    
    if (diffusion_debug_enabled()) {
        MNN_PRINT("[SCHEDULER] Unified API initialized successfully\n");
    }
    
    // Pre-allocate mSampleVar for models that need it
    // - LongCat: packed format [B, seq, 64]
    // - Z-Image: standard NCHW format [B, C, H, W] (required for stable UNet input)
    if (mModelType == LONGCAT_IMAGE_EDIT) {
        int B = 1;
        int C = mLatentC;  // 16
        int H = mLatentH;  // 64
        int W = mLatentW;  // 64
        int packedH = H / 2;  // 32
        int packedW = W / 2;  // 32
        int packedSeq = packedH * packedW;  // 1024
        int packedC = C * 4;  // 64
        int totalSeq = isT2IMode ? packedSeq : (2 * packedSeq);
        
        mSampleVar = _Input({B, totalSeq, packedC}, NCHW, halide_type_of<float>());
        if (diffusion_debug_enabled()) {
            MNN_PRINT("[DIFFUSION] Pre-allocated mSampleVar: [%d, %d, %d] (%.1f KB)\n", 
                      B, totalSeq, packedC, (B * totalSeq * packedC * 4) / 1024.0f);
        }
    } else if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
        // Z-Image: pre-allocate mSampleVar with standard NCHW shape
        // This is required for stable UNet input (direct plms assignment causes crash)
        mSampleVar = _Input({1, mLatentC, mLatentH, mLatentW}, NCHW, halide_type_of<float>());
        if (diffusion_debug_enabled()) {
            MNN_PRINT("[DIFFUSION] Pre-allocated mSampleVar: [%d, %d, %d, %d] (%.1f KB)\n", 
                      1, mLatentC, mLatentH, mLatentW, 
                      (mLatentC * mLatentH * mLatentW * 4) / 1024.0f);
        }
    }
    
    int step_index = 0;
    for (int i = 0; i < mTimeSteps.size(); i++) {
        AUTOTIME;
        //display_progress(i, mTimeSteps.size());
        
        if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
            // FlowMatch Euler discrete scheduler for ZImage.
            if (mSigmas.empty() || mSigmas.size() < mTimeSteps.size() + 1) {
                MNN_PRINT("Error: FlowMatch sigma schedule not initialized correctly.\n");
                return nullptr;
            }

            float sigma = mSigmas[i];
            float sigmaNext = mSigmas[i + 1];
            // For FlowMatch Euler, the UNet "t" input aligns with 1 - sigma in Python tests.
            // Z-image: t = 1 - sigma
            // LongCat: t = sigma (different convention)
            float t = (mModelType == LONGCAT_IMAGE_EDIT) ? sigma : (1.0f - sigma);
            ptr[0] = t;

            mTimestepVar->input(floatVar);
            step_index += 1;
            
            // Prepare sample_input for UNet
            // - LongCat: CPU pack plms [B,C,H,W] -> mSampleVar [B,seq,64]
            // - Z-Image: copy plms to mSampleVar [B,C,H,W] (GPU-side copy)
            VARP sample_input;
            
            if (mModelType == LONGCAT_IMAGE_EDIT) {
                // LongCat: CPU pack plms [B,C,H,W] -> mSampleVar [B,seq,64]
                // Note: GPU pack causes computation graph accumulation (MNN Express API limitation)
                // CPU pack with writeMap/readMap is slower but avoids step time increase
                int B = 1, C = mLatentC, H = mLatentH, W = mLatentW;
                int packedSeq = (H / 2) * (W / 2);
                
                auto samplePtr = mSampleVar->writeMap<float>();
                auto noisePtr = plms->readMap<float>();
                packLatents(noisePtr, samplePtr, B, C, H, W, 0);
                
                if (!isT2IMode && mImageLatentsVar.get()) {
                    auto imagePtr = mImageLatentsVar->readMap<float>();
                    packLatents(imagePtr, samplePtr, B, C, H, W, packedSeq);
                }
                sample_input = mSampleVar;
            } else {
                // Z-Image: copy plms to pre-allocated mSampleVar (GPU-side copy)
                mSampleVar->input(plms);
                sample_input = mSampleVar;
            }
            
            // Debug logging only when MNN_DIFFUSION_DEBUG is set
            if (diffusion_debug_enabled() && step_index == 1) {
                log_var_shape_mnn("ZIMAGE_UNet_in_sample", sample_input);
                log_var_shape_mnn("ZIMAGE_UNet_in_timestep", mTimestepVar);
                log_var_shape_mnn("ZIMAGE_UNet_in_encoder_hidden_states", text_embeddings);
            }
            
            // Limited Interval CFG for LongCat: apply CFG only in intermediate noise levels
            // CFG range is configurable via mCFGMode
            float sigma_low, sigma_high;
            getCFGSigmaRange(sigma_low, sigma_high);
            bool apply_cfg = (mModelType == LONGCAT_IMAGE_EDIT) && 
                            (sigma > sigma_low && sigma <= sigma_high) && 
                            (std::abs(cfgScale - 1.0f) > 0.001f);
            
            VARP noise_pred;
            
            // Print step info (unified format for all models)
            float dt = sigmaNext - sigma;
            if (diffusion_debug_enabled()) {
                MNN_PRINT("[UNet] step=%d, sigma=%f, sigmaNext=%f, t=%f, dt=%f\n", 
                          step_index, sigma, sigmaNext, t, dt);
            }
            
            if (apply_cfg) {
                // Apply CFG: run UNet twice (conditional + unconditional)
                
                // 1. Conditional (with prompt embeddings)
                std::vector<VARP> unet_inputs_cond;
                if (mModelType == LONGCAT_IMAGE_EDIT) {
                    unet_inputs_cond = {sample_input, mTimestepVar, text_embeddings, mTxtIdsVar, mImgIdsVar};
                } else {
                    unet_inputs_cond = {sample_input, mTimestepVar, text_embeddings};
                }
                auto outputs_cond = mModules[1]->onForward(unet_inputs_cond);
                
                if (outputs_cond.empty() || !outputs_cond[0].get()) {
                    MNN_PRINT("[UNet] ERROR: conditional outputs is empty or null!\n");
                    return nullptr;
                }
                auto output_cond = _Convert(outputs_cond[0], NCHW);
                
                // 2. Unconditional (with pre-created zero embeddings)
                std::vector<VARP> unet_inputs_uncond;
                if (mModelType == LONGCAT_IMAGE_EDIT) {
                    unet_inputs_uncond = {sample_input, mTimestepVar, zero_embeddings, mTxtIdsVar, mImgIdsVar};
                } else {
                    unet_inputs_uncond = {sample_input, mTimestepVar, zero_embeddings};
                }
                auto outputs_uncond = mModules[1]->onForward(unet_inputs_uncond);
                
                if (outputs_uncond.empty() || !outputs_uncond[0].get()) {
                    MNN_PRINT("[UNet] ERROR: unconditional outputs is empty or null!\n");
                    return nullptr;
                }
                auto output_uncond = _Convert(outputs_uncond[0], NCHW);
                
                // Apply CFG formula: uncond + cfg_scale * (cond - uncond)
                noise_pred = output_uncond + _Scalar(cfgScale) * (output_cond - output_uncond);
                
                // Z-image: Python side treats model_output as the negative of raw UNet output
                if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
                    noise_pred = _Scalar(-1.0f) * noise_pred;
                }
                
            } else {
                // No CFG: run UNet once (conditional only)
                std::vector<VARP> unet_inputs;
                if (mModelType == LONGCAT_IMAGE_EDIT) {
                    unet_inputs = {sample_input, mTimestepVar, text_embeddings, mTxtIdsVar, mImgIdsVar};
                } else {
                    unet_inputs = {sample_input, mTimestepVar, text_embeddings};
                }
                auto outputs = mModules[1]->onForward(unet_inputs);
                
                if (outputs.empty() || !outputs[0].get()) {
                    MNN_PRINT("[UNet] ERROR: outputs is empty or null!\n");
                    return nullptr;
                }
                
                auto output = _Convert(outputs[0], NCHW);
                
                // Z-image: Python side treats model_output as the negative of raw UNet output
                // noise_pred = -UNet_raw_output
                // LongCat: no negation needed
                if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
                    noise_pred = _Scalar(-1.0f) * output;
                } else {
                    noise_pred = output;
                }
                
                // For ZImage with CFG != 1.0, apply simple scaling
                if (mModelType == STABLE_DIFFUSION_ZIMAGE && std::abs(cfgScale - 1.0f) > 0.001f) {
                    noise_pred = _Scalar(cfgScale) * noise_pred;
                }
            }
            
            // ========================================================================
            // Unified Scheduler API: Two-Layer Update
            // ========================================================================
            // Layer 1: Preprocess UNet output (model-specific format conversion)
            // Layer 2: Apply scheduler update (scheduler-specific logic)
            //
            // This design allows all models to share the same update logic:
            //   - SD1.5: no preprocessing + PLMS update
            //   - Z-Image: no preprocessing + Euler update
            //   - LongCat: GPU slice+unpack + Euler update
            //   - Future models: custom preprocessing + appropriate scheduler
            // ========================================================================
            
            // Layer 1: Preprocess UNet output to standard [B, C, H, W] format
            auto noise_pred_standard = mUNetPreprocess(noise_pred);
            
            // Layer 2: Apply scheduler update
            // For Z-Image/LongCat: copy result back to pre-allocated plms buffer to break computation graph chain
            if (mSchedulerType == SCHEDULER_PLMS) {
                // SD1.5: PLMS 4-order update with noise history
                plms = applyPLMSUpdate(plms, noise_pred_standard, dt, i);
            } else {
                // Z-Image, LongCat: Euler 1-order update
                auto updated = applyEulerUpdate(plms, noise_pred_standard, dt);
                // Copy result back to pre-allocated plms buffer to break computation graph chain
                // This prevents VARP accumulation that causes step time to increase
                plms->input(updated);
            }
            
            // Explicitly release temporary VARPs to prevent GPU memory accumulation
            // OpenCL backend has async memory release, so we need to force cleanup
            noise_pred = nullptr;
            noise_pred_standard = nullptr;
            
            // Trigger garbage collection for OpenCL to prevent memory buildup
            // All models now use pure GPU operations, so GC every 2 steps is sufficient
            if (mBackendType == MNN_FORWARD_OPENCL && (i + 1) % 2 == 0) {
                MNN::Express::ExecutorScope::Current()->gc(MNN::Express::Executor::PART);
                if (diffusion_debug_enabled()) {
                    MNN_PRINT("[MEMORY] Triggered GC at step %d\n", i + 1);
                }
            }

#ifdef MNN_DUMP_DATA
            // Note: output/noise_pred already released, use plms for debug
            auto xx = plms->readMap<float>();
            auto yy = mSampleVar->readMap<float>();
            auto zz = text_embeddings->readMap<float>();
            const float* mmF = nullptr;
            const int* mmI = nullptr;
            if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
                mmF = mTimestepVar->readMap<float>();
            } else {
                mmI = mTimestepVar->readMap<int>();
            }

            for(int j=0; j<6; j+=2) {
                if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
                    MNN_PRINT("(0)%f (1)%f (2)%f (3)%f ", xx[j], yy[j], zz[j] ,mmF ? mmF[0] : 0.0f);
                } else {
                    MNN_PRINT("(0)%f (1)%f (2)%f (3)%d ", xx[j], yy[j], zz[j] ,mmI ? mmI[0] : 0);
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
            std::vector<VARP> unet_inputs;
            if (mModelType == LONGCAT_IMAGE_EDIT) {
                // LongCat needs txt_ids and img_ids
                unet_inputs = {mSampleVar, mTimestepVar, text_embeddings, mTxtIdsVar, mImgIdsVar};
            } else {
                unet_inputs = {mSampleVar, mTimestepVar, text_embeddings};
            }
            auto outputs = mModules[1]->onForward(unet_inputs);
            auto output = _Convert(outputs[0], NCHW);
            
            auto noise_pred = output;
            
            auto splitvar = _Split(noise_pred, {2}, 0);
            auto noise_pred_uncond = splitvar[0];
            auto noise_pred_text = splitvar[1];
            
            noise_pred = scalevar * (noise_pred_text - noise_pred_uncond) + noise_pred_uncond;
            
            plms = step_plms(plms, noise_pred, i);
            // Note: output/noise_pred/splitvar/outputs are local variables, they will be
            // automatically released when going out of scope at end of loop iteration.
            // MNN uses reference counting, so no need to explicitly set to nullptr.
            
#ifdef MNN_DUMP_DATA
            // Note: These readMap calls are only for debug builds
            auto xx = plms->readMap<float>();
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
    
    // Apply VAE scaling based on model type
    if (mModelType == LONGCAT_IMAGE_EDIT) {
        // LongCat uses Flux VAE with different scaling
        float scalingFactor = 0.3611f;
        float shiftFactor = 0.1159f;
        latent = latent * _Const(1.0f / scalingFactor) + _Const(shiftFactor);
    } else {
        // SD1.5 and Z-image use standard SD VAE scaling
        latent = latent * _Const(1 / 0.18215);
    }
    
    AUTOTIME;
    auto outputs = mModules[2]->onForward({latent});
    auto output = _Convert(outputs[0], NCHW);
    
#ifdef MNN_DUMP_DATA
    auto xx = output->readMap<float>();
    for(int i=0; i<320; i+=32) {
        MNN_PRINT("%f ", xx[i]);
    }
    MNN_PRINT("\n\n");
#endif
    
    // VAE decoder output post-processing
    // VAE outputs in range [-1, 1], need to normalize to [0, 1] then scale to [0, 255]
    auto image = output;
    image = _Relu6(image * _Const(0.5) + _Const(0.5), 0, 1);  // [-1,1] -> [0,1] with clamp
    image = _Squeeze(_Transpose(image, {0, 2, 3, 1}));         // NCHW -> HWC
    image = _Cast(_Round(image * _Const(255.0)), halide_type_of<uint8_t>());  // [0,1] -> [0,255]
    
    // Convert BGR to RGB for imwrite
    // VAE decoder outputs RGB, but OpenCV imwrite expects BGR
    image = cvtColor(image, COLOR_BGR2RGB);
    
    image.fix(VARP::CONSTANT);
    if (mModelType == LONGCAT_IMAGE_EDIT) {
        MNN_PRINT("[LongCat] VAE decoder output ready (converted RGB->BGR for imwrite)\n");
    }
    return image;
}

bool Diffusion::run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback, const std::string inputImagePath) {
    AUTOTIME;
    mEts.clear();
    
    // LongCat: Determine mode (Image Edit vs T2I) and prepare image latents
    bool isT2IMode = false;
    VARP vaePreprocessedImage = nullptr;  // Save VAE preprocessed image for text encoder
    
    if (mModelType == LONGCAT_IMAGE_EDIT) {
        if (inputImagePath.empty()) {
            // T2I mode: no input image, generate from pure text
            isT2IMode = true;
            MNN_PRINT("[LongCat] T2I mode: no input image, generating from text only\n");
            // mImageLatentsVar remains nullptr, will be handled in unet()
            mImageLatentsVar = nullptr;
        } else {
            // Image Edit mode: encode input image
            MNN_PRINT("[LongCat] Image Edit mode: loading input image: %s\n", inputImagePath.c_str());
            
            // Load and process image
            using namespace MNN::CV;
            VARP inputImage;
            
            auto rawImage = imread(inputImagePath);
            if (!rawImage.get()) {
                MNN_PRINT("Error: Failed to load input image\n");
                return false;
            }
            
            auto origInfo = rawImage->getInfo();
            int origH = origInfo->dim[0];
            int origW = origInfo->dim[1];
            int origC = origInfo->dim[2];
            MNN_PRINT("[LongCat] Original: %dx%d, channels=%d\n", origW, origH, origC);
            
            int targetW = mImageWidth;
            int targetH = mImageHeight;
            
            // Use common utility function: resize preserving aspect ratio + center crop
            auto processedImage = resizeAndCenterCrop(rawImage, targetW, targetH);
            MNN_PRINT("[LongCat] Resized and center cropped: %dx%d -> %dx%d\n", origW, origH, targetW, targetH);
            
            // Update rawImage to processed image
            rawImage = processedImage;
            
            // Convert BGR to RGB using common utility function
            VARP rgbImage = bgrToRgb(rawImage);
            MNN_PRINT("[LongCat] Converted BGR to RGB (matching Diffusers Pipeline)\n");
            
            // Save preprocessed image (uint8 RGB format) for text encoder (in memory)
            // IMPORTANT: RGB image is shared between VAE and Text Encoder
            vaePreprocessedImage = rgbImage;
            MNN_PRINT("[LongCat] Prepared RGB preprocessed image for text encoder (in memory)\n");
            
            // Convert HWC to NCHW with normalization using common utility function
            inputImage = hwcToNchw(rgbImage, true);  // normalize=true: (x/127.5 - 1.0)
            
            MNN_PRINT("[LongCat] Prepared for VAE: [1,%d,%d,%d] RGB format, normalized to [-1,1]\n", origC, targetH, targetW);
            
            // Encode with VAE encoder
            auto vaeOutputs = mModules[3]->onForward({inputImage});
            if (vaeOutputs.empty() || !vaeOutputs[0].get()) {
                MNN_PRINT("Error: VAE encoder failed\n");
                return false;
            }
            
            // Apply VAE scaling: (latent - shift) * scale
            float scalingFactor = 0.3611f;
            float shiftFactor = 0.1159f;
            mImageLatentsVar = (vaeOutputs[0] - _Const(shiftFactor)) * _Const(scalingFactor);
            
            auto info = mImageLatentsVar->getInfo();
            MNN_PRINT("[LongCat] Image latents shape: [%d, %d, %d, %d]\n", 
                      info->dim[0], info->dim[1], info->dim[2], info->dim[3]);
            
            // Unload VAE encoder after use (memory_mode != 1)
            if (mMemoryMode != 1) {
                mModules[3].reset();
                MNN_PRINT("[LongCat] VAE encoder unloaded (memory_mode=%d)\n", mMemoryMode);
            }
        }
    }
 
    if(iterNum > 50) {
        iterNum = 50;
        MNN_PRINT("too much number of iterations, iterations will be set to 50.\n");
    }
    if(iterNum < 1) {
        iterNum = 10;
        MNN_PRINT("illegal number of iterations, iterations will be set to 10.\n");
    }
    if (mModelType == STABLE_DIFFUSION_ZIMAGE || mModelType == LONGCAT_IMAGE_EDIT) {
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

    VARP text_embeddings;
    
    if (mModelType == LONGCAT_IMAGE_EDIT) {
        // LongCat: Use integrated LLM text encoder
        // Pass VAE preprocessed image (or nullptr for T2I mode)
        text_embeddings = text_encoder_llm(prompt, vaePreprocessedImage);
        if (!text_embeddings.get()) {
            MNN_PRINT("Error: LLM text encoder failed\n");
            return false;
        }
        // Note: LLM text encoder cleanup is handled internally by mLlm destructor
    } else {
        // ZImage and SD: use internal tokenizer and text_encoder
        std::string promptForTokenizer = prompt;
        if (mModelType == STABLE_DIFFUSION_ZIMAGE) {
            // Mirror Python tokenizer.apply_chat_template for a single user message
            promptForTokenizer = std::string("<|im_start|>user\n") + prompt +
                                std::string("<|im_end|>\n<|im_start|>assistant\n<think>\n");
        }

        auto ids = mTokenizer->encode(promptForTokenizer, mMaxTextLen);
        text_embeddings = text_encoder(ids);
    }
     
    if (progressCallback) {
        progressCallback(1 * 100 / (iterNum + 3)); // percent
    }
    auto latent = unet(text_embeddings, iterNum, randomSeed, cfgScale, progressCallback);
     
    auto image = vae_decoder(latent);
    bool res = imwrite(outputPath, image);
    if (res) {
        MNN_PRINT("SUCCESS! write generated image to %s\n", outputPath.c_str());
    }

    // Unload VAE decoder based on memory_mode
    // Note: text_encoder is already unloaded in unet(), VAE encoder in run() after encoding
    if (mMemoryMode != 1) {
        mModules[2].reset();
        MNN_PRINT("[DIFFUSION] vae_decoder unloaded (memory_mode=%d)\n", mMemoryMode);
    }
     
    if (progressCallback) {
        progressCallback(100); // percent
    }
    return res;
}
}
}
