//
//  SoftmaxScalar.hpp
//  MNN
//
//  Generic scalar softmax used as a fallback by the SIMD softmax kernels.
//
//  The AVX / AVX512 / SSE variants process `packUnit` (8 / 16 / 4) contiguous
//  floats per step, but the layout of a packed tensor is given by `pack`, which
//  can be smaller than the vector width (e.g. pack = 4 for NC4HW4). Reading a
//  full vector in that case walks past each packed group and corrupts memory.
//  Those kernels call this implementation whenever pack <= 1 or pack < packUnit.
//

#ifndef MNN_SOFTMAX_SCALAR_HPP
#define MNN_SOFTMAX_SCALAR_HPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "core/Macro.h"

namespace MNN {

// outside  : number of independent softmax rows
// reduceSize: length reduced by each softmax
// pack     : packed layout step (1 = contiguous, otherwise groups of `pack`
//            elements separated by outside * pack)
static inline void MNNSoftmaxScalar(float* softmaxDst, const float* softmaxSrc, float* runningMax, float* runningSum,
                                    float* updateScale, int outside, int reduceSize, int kvSeqOffset, int validOffset,
                                    int pack, bool mask) {
    const int inner  = pack > 1 ? pack : reduceSize;
    const int outer  = pack > 1 ? UP_DIV(reduceSize, pack) : 1;
    const int stride = pack > 1 ? outside * inner : inner;
    const float lowest = std::numeric_limits<float>::lowest();

    for (int k = 0; k < outside; ++k) {
        const int valid = mask ? ALIMIN(reduceSize, ALIMAX(k + (validOffset + 1) - kvSeqOffset, 0)) : reduceSize;
        if (valid <= 0) {
            for (int j = 0; j < outer; ++j) {
                memset(softmaxDst + j * stride + k * inner, 0, inner * sizeof(float));
            }
            if (runningMax && runningSum && updateScale) {
                updateScale[k] = 1.0f;
            }
            continue;
        }
        float mx = lowest;
        for (int j = 0; j < outer; ++j) {
            const float* src = softmaxSrc + j * stride + k * inner;
            const int limit  = ALIMIN(inner, valid - j * inner);
            for (int i = 0; i < limit; ++i) {
                mx = ALIMAX(mx, src[i]);
            }
        }
        const float oldMax   = runningMax ? runningMax[k] : lowest;
        const float finalMax = ALIMAX(oldMax, mx);

        float sum = 0.0f;
        for (int j = 0; j < outer; ++j) {
            const float* src = softmaxSrc + j * stride + k * inner;
            float* dst       = softmaxDst + j * stride + k * inner;
            const int limit  = ALIMIN(inner, valid - j * inner);
            for (int i = 0; i < limit; ++i) {
                const float v = expf(src[i] - finalMax);
                dst[i]        = v;
                sum += v;
            }
            for (int i = ALIMAX(limit, 0); i < inner; ++i) {
                dst[i] = 0.0f;
            }
        }

        if (runningMax && runningSum && updateScale) {
            const float scaleForSum = expf(oldMax - finalMax);
            runningSum[k]           = runningSum[k] * scaleForSum + sum;
            runningMax[k]           = finalMax;
            updateScale[k]          = scaleForSum;
        } else {
            if (runningMax && runningSum) {
                sum += runningSum[k] * expf(oldMax - finalMax);
            }
            const float scale = 1.0f / (sum + 1e-20f);
            for (int j = 0; j < outer; ++j) {
                float* dst      = softmaxDst + j * stride + k * inner;
                const int limit = ALIMIN(inner, valid - j * inner);
                for (int i = 0; i < limit; ++i) {
                    dst[i] *= scale;
                }
            }
        }
    }
}

} // namespace MNN

#endif // MNN_SOFTMAX_SCALAR_HPP
