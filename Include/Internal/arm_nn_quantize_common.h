/*
 * SPDX-FileCopyrightText: Copyright 2026 Google LLC
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_quantize_common.h
 * Description:  Shared scalar and MVE helpers for affine float32 quantization
 *
 * $Date:        24 November 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_QUANTIZE_COMMON_H
#define ARM_NN_QUANTIZE_COMMON_H

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

__STATIC_INLINE uint32_t arm_nn_quantize_f32_to_bits(const float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

__STATIC_INLINE int32_t arm_nn_quantize_f32_is_valid_scale(const float scale)
{
    const uint32_t bits = arm_nn_quantize_f32_to_bits(scale);
    /* Strictly positive and finite: sign bit 0, exponent in [0, 254], non-zero. */
    return ((bits & 0x80000000UL) == 0UL) && (bits > 0UL) && (bits < 0x7F800000UL);
}

__STATIC_INLINE int32_t arm_nn_quantize_round_clamp_f32(const float scaled,
                                                        const int32_t zero_point,
                                                        const int32_t min_val,
                                                        const int32_t max_val)
{
    const uint32_t bits = arm_nn_quantize_f32_to_bits(scaled);
    const uint32_t abs_bits = bits & 0x7FFFFFFFUL;

    /* Match Arm vcvta.s32.f32 on NaN (converts to 0) + zero_point, clamped. */
    if (abs_bits > 0x7F800000UL)
    {
        return CLAMP(zero_point, max_val, min_val);
    }

    /* |scaled| >= 2^33 (including +/-Inf): |scaled| + any int32_t zero_point
     * strictly exceeds the int32_t range. */
    if (abs_bits >= 0x50000000UL)
    {
        return ((bits & 0x80000000UL) != 0UL) ? min_val : max_val;
    }

    /* 2^23 <= |scaled| < 2^33: already an exact integer in binary32. */
    if (abs_bits >= 0x4B000000UL)
    {
        const int64_t unclamped = (int64_t)scaled + (int64_t)zero_point;
        if (unclamped <= (int64_t)min_val)
        {
            return min_val;
        }
        if (unclamped >= (int64_t)max_val)
        {
            return max_val;
        }
        return (int32_t)unclamped;
    }

    /* |scaled| < 2^23: exact round-half-away-from-zero without <math.h>,
     * avoiding the 0.49999997f + 0.5f == 1.0f rounding hazard. */
    int32_t rounded = (int32_t)scaled;
    const float frac = scaled - (float)rounded;
    if (frac >= 0.5f)
    {
        rounded += 1;
    }
    else if (frac <= -0.5f)
    {
        rounded -= 1;
    }

    const int64_t unclamped = (int64_t)rounded + (int64_t)zero_point;
    if (unclamped <= (int64_t)min_val)
    {
        return min_val;
    }
    if (unclamped >= (int64_t)max_val)
    {
        return max_val;
    }
    return (int32_t)unclamped;
}

__STATIC_INLINE int32_t arm_nn_quantize_scalar_elem_f32(const float val,
                                                        const float scale,
                                                        const int32_t zero_point,
                                                        const int32_t min_val,
                                                        const int32_t max_val)
{
    /* Inspect raw IEEE-754 bits before dividing so NaN, +/-Inf, and
     * guaranteed-saturating divisions never execute float division on
     * non-finite inputs or overflow to Inf (safe even under -ffinite-math-only). */
    const uint32_t val_bits = arm_nn_quantize_f32_to_bits(val);
    const uint32_t val_abs = val_bits & 0x7FFFFFFFUL;
    if (val_abs > 0x7F800000UL)
    {
        return CLAMP(zero_point, max_val, min_val);
    }
    if (val_abs == 0x7F800000UL)
    {
        return ((val_bits & 0x80000000UL) != 0UL) ? min_val : max_val;
    }
    if (val_abs == 0UL)
    {
        return CLAMP(zero_point, max_val, min_val);
    }

    const uint32_t scale_bits = arm_nn_quantize_f32_to_bits(scale);
    const int32_t val_exp = (int32_t)(val_abs >> 23);
    const int32_t scale_exp = (int32_t)(scale_bits >> 23);
    /* When val is normal and (val_exp - scale_exp) >= 35, |val / scale| > 2^33
     * even for subnormal scale (scale_exp == 0), so val / scale + zero_point
     * strictly exceeds int32_t range. Otherwise |val / scale| < 2^57 << FLT_MAX. */
    if ((val_exp >= 1) && ((val_exp - scale_exp) >= 35))
    {
        return ((val_bits & 0x80000000UL) != 0UL) ? min_val : max_val;
    }

    /* Volatile copy prevents -Ofast (-freciprocal-math) from rewriting
     * val / scale into val * (1.0f / scale), preserving bit-exactness on ties. */
    volatile float denom = scale;
    return arm_nn_quantize_round_clamp_f32(val / denom, zero_point, min_val, max_val);
}

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE int32_t arm_nn_quantize_can_use_mve_f32(const float scale, const int32_t zero_point)
{
    const uint32_t scale_bits = arm_nn_quantize_f32_to_bits(scale);
    /* Require 2^-125 <= scale <= 2^126 (exponent in [2, 253]) BEFORE computing
     * 1.0f / scale so:
     * - 1.0f / scale is guaranteed normal finite (full 24-bit mantissa precision,
     *   never overflows to Inf under -ffinite-math-only),
     * - Armv8.1-M MVE's architectural flush-to-zero (StandardFPSCRValue FZ=1) on
     *   subnormal inputs (|v| < 2^-126) remains bit-exact with IEEE-754 scalar
     *   division because |v / scale| < 0.5 always rounds to 0 when scale >= 2^-125,
     * - zero_point is bounded so vqaddq_n_s32 cannot pull a saturated
     *   INT32_MIN/INT32_MAX back into [min_val, max_val]. */
    return (scale_bits >= 0x01000000UL) && (scale_bits <= 0x7E800000UL) && (zero_point >= -1000000000) &&
        (zero_point <= 1000000000);
}
#endif

#endif /* ARM_NN_QUANTIZE_COMMON_H */
