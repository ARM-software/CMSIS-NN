/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_activation_flt.h
 * Description:  Internal floating-point activation helper utilities
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_ACTIVATION_FLT_H
#define ARM_NN_ACTIVATION_FLT_H

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

__STATIC_INLINE float32_t arm_nn_hardswish_scalar_f32(float32_t x)
{
    float32_t t = x * (1.0f / 6.0f) + 0.5f;
    t = CLAMP(t, 1.0f, 0.0f);
    return x * t;
}

/*
 * Tanh deliberately uses different scalar approximations by dtype.
 * Float32 keeps a LUT + linear interpolation path to preserve tighter
 * accuracy over a wider input range, while float16 uses a compact rational
 * approximation because half precision does not benefit as much from a
 * larger table and the lower-order form is sufficient for its target error.
 */
__STATIC_INLINE float32_t arm_nn_tanh_scalar_ref_f32(float32_t x)
{
    float32_t ax = (x < 0.0f) ? -x : x;
    const float32_t xmax = 4.0f;

    if (ax > xmax)
    {
        return (x < 0.0f) ? -1.0f : 1.0f;
    }

    const float32_t t = ax * (256.0f / xmax);
    int32_t idx = (int32_t)t;
    idx = CLAMP(idx, 255, 0);
    const float32_t frac = t - (float32_t)idx;
    const float32_t y0 = arm_nn_tanh_lut256_f32[idx];
    const float32_t y1 = arm_nn_tanh_lut256_f32[idx + 1];
    const float32_t y = y0 + (y1 - y0) * frac;
    return (x < 0.0f) ? -y : y;
}

/*
 * Keep the float32 sigmoid on the non-positive exp domain used by softmax.
 * This mirrors the float16 helper and avoids needlessly evaluating exp()
 * on large positive arguments.
 */
__STATIC_INLINE float32_t arm_nn_sigmoid_scalar_f32(float32_t x)
{
    if (x >= 0.0f)
    {
        const float32_t e = arm_nn_softmax_exp_scalar_f32(-x);
        return 1.0f / (1.0f + e);
    }

    const float32_t e = arm_nn_softmax_exp_scalar_f32(x);
    return e / (1.0f + e);
}

__STATIC_INLINE float32_t arm_nn_apply_activation_type_f32(float32_t x,
                                                           arm_nn_activation_type_flt type,
                                                           float32_t act_param)
{
    switch (type)
    {
    case ARM_NN_FLT_ACT_NONE:
        return x;
    case ARM_NN_FLT_ACT_SIGMOID:
        return arm_nn_sigmoid_scalar_f32(x);
    case ARM_NN_FLT_ACT_RELU:
        return (x < 0.0f) ? 0.0f : x;
    case ARM_NN_FLT_ACT_RELU6: {
        const float32_t y = (x < 0.0f) ? 0.0f : x;
        return (y > 6.0f) ? 6.0f : y;
    }
    case ARM_NN_FLT_ACT_TANH:
        return arm_nn_tanh_scalar_ref_f32(x);
    case ARM_NN_FLT_ACT_HARDSWISH:
        return arm_nn_hardswish_scalar_f32(x);
    case ARM_NN_FLT_ACT_LEAKY_RELU:
        return (x >= 0.0f) ? x : (act_param * x);
    default:
        return x;
    }
}

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE float32x4_t arm_nn_clamp_mve_f32(float32x4_t x, float32x4_t min_v, float32x4_t max_v)
{
    x = vmaxnmq(x, min_v);
    x = vminnmq(x, max_v);
    return x;
}

__STATIC_INLINE float32x4_t arm_nn_vtanh_lut_direct_mve_f32(float32x4_t x)
{
    float32x4_t ax = vabsq(x);
    const mve_pred16_t sat_p = vcmpgtq(ax, 4.0f);
    ax = vminnmq(ax, vdupq_n_f32(4.0f));
    const uint32_t xmax = 4U;
    const uint32_t lut_tbl_max_idx = 256U;
    const float32x4_t t = vmulq(ax, (float32_t)(lut_tbl_max_idx / xmax));
    uint32x4_t idx = vcvtmq_u32_f32(t);
    idx = vminq(idx, vdupq_n_u32(255U));
    const float32x4_t frac = vsubq(t, vcvtq_f32_u32(idx));
    const float32x4_t y0 = vldrwq_gather_shifted_offset((const float32_t *)arm_nn_tanh_lut256_f32, idx);
    const float32x4_t y1 = vldrwq_gather_shifted_offset((const float32_t *)arm_nn_tanh_lut256_f32, vaddq(idx, 1U));
    float32x4_t y = vfmaq(y0, vsubq(y1, y0), frac);
    y = vpselq(vdupq_n_f32(1.0f), y, sat_p);
    return vnegq_m(y, y, vcmpltq(x, 0.0f));
}

__STATIC_INLINE float32x4_t arm_nn_vhardswish_mve_f32(float32x4_t x)
{
    float32x4_t t = vfmaq(vdupq_n_f32(0.5f), x, (1.0f / 6.0f));
    t = vmaxnmq(t, vdupq_n_f32(0.0f));
    t = vminnmq(t, vdupq_n_f32(1.0f));
    return vmulq(x, t);
}
    #endif

__STATIC_INLINE void
arm_nn_vector_clamp_f32(float32_t *data, int32_t block_size, float32_t activation_min, float32_t activation_max)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t vmin = vdupq_n_f32(activation_min);
    const float32x4_t vmax = vdupq_n_f32(activation_max);
    for (int32_t i = 0; i < block_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(block_size - i));
        float32x4_t v = vld1q_z(data + i, p);
        v = arm_nn_clamp_mve_f32(v, vmin, vmax);
        vst1q_p(data + i, v, p);
    }
    #else
    for (int32_t i = 0; i < block_size; ++i)
    {
        data[i] = CLAMP(data[i], activation_max, activation_min);
    }
    #endif
}

#endif /* ARM_NN_ENABLE_F32 */

#if ARM_NN_ENABLE_F16

__STATIC_INLINE float16_t arm_nn_clamp_scalar_f16(float16_t x, float16_t min_v, float16_t max_v)
{
    const _Float16 xh = (_Float16)x;
    const _Float16 min_h = (_Float16)min_v;
    const _Float16 max_h = (_Float16)max_v;
    const _Float16 y = (xh < min_h) ? min_h : ((xh > max_h) ? max_h : xh);
    return (float16_t)y;
}

__STATIC_INLINE float16_t arm_nn_hardswish_scalar_f16(float16_t x)
{
    _Float16 t = (_Float16)x * (_Float16)(1.0f / 6.0f) + (_Float16)0.5f;
    t = CLAMP(t, (_Float16)1.0f, (_Float16)0.0f);
    return (float16_t)((_Float16)x * t);
}

__STATIC_INLINE float16_t arm_nn_tanh_scalar_ref_f16(float16_t x)
{
    const _Float16 ax = ((_Float16)x < (_Float16)0.0f) ? -(_Float16)x : (_Float16)x;
    if (ax > (_Float16)arm_nn_tanh_approx_coeffs_f16[0])
    {
        return ((_Float16)x < (_Float16)0.0f) ? (_Float16)-1.0f : (_Float16)1.0f;
    }

    const _Float16 x2 = (_Float16)x * (_Float16)x;
    const _Float16 num = (_Float16)x * ((_Float16)arm_nn_tanh_approx_coeffs_f16[1] + x2);
    const _Float16 den = (_Float16)arm_nn_tanh_approx_coeffs_f16[1] + (_Float16)arm_nn_tanh_approx_coeffs_f16[2] * x2;
    return (float16_t)(num / den);
}

/*
 * Keep the float16 sigmoid on the non-positive exp domain used by softmax.
 * This avoids relying on the positive-side exp
 */
__STATIC_INLINE float16_t arm_nn_sigmoid_scalar_f16(float16_t x)
{
    if ((_Float16)x >= (_Float16)0.0f)
    {
        const float32_t e = (float32_t)arm_nn_softmax_exp_scalar_f16((float16_t)(-(_Float16)x));
        return (float16_t)(1.0f / (1.0f + e));
    }

    const float32_t e = (float32_t)arm_nn_softmax_exp_scalar_f16((float16_t)(_Float16)x);
    return (float16_t)(e / (1.0f + e));
}

__STATIC_INLINE float16_t arm_nn_apply_activation_type_f16(float16_t x,
                                                           arm_nn_activation_type_flt type,
                                                           float16_t act_param)
{
    switch (type)
    {
    case ARM_NN_FLT_ACT_NONE:
        return x;
    case ARM_NN_FLT_ACT_SIGMOID:
        return arm_nn_sigmoid_scalar_f16(x);
    case ARM_NN_FLT_ACT_RELU:
        return (float16_t)((_Float16)x < (_Float16)0.0f ? (_Float16)0.0f : (_Float16)x);
    case ARM_NN_FLT_ACT_RELU6: {
        const float16_t y = ((_Float16)x < (_Float16)0.0f) ? (_Float16)0.0f : (_Float16)x;
        return (float16_t)((_Float16)y > (_Float16)6.0f ? (_Float16)6.0f : (_Float16)y);
    }
    case ARM_NN_FLT_ACT_TANH:
        return arm_nn_tanh_scalar_ref_f16(x);
    case ARM_NN_FLT_ACT_HARDSWISH:
        return arm_nn_hardswish_scalar_f16(x);
    case ARM_NN_FLT_ACT_LEAKY_RELU:
        return (float16_t)((_Float16)x >= (_Float16)0.0f ? (_Float16)x : (_Float16)((_Float16)act_param * (_Float16)x));
    default:
        return x;
    }
}

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE float16x8_t arm_nn_clamp_mve_f16(float16x8_t x, float16x8_t min_v, float16x8_t max_v)
{
    x = vmaxnmq(x, min_v);
    x = vminnmq(x, max_v);
    return x;
}

__STATIC_INLINE float16x8_t arm_nn_vtanh_lut_direct_mve_f16(float16x8_t x)
{
    float16x8_t ax = vabsq(x);
    const mve_pred16_t sat_p = vcmpgtq(ax, (float16_t)4.0f);
    ax = vminnmq(ax, vdupq_n_f16((float16_t)4.0f));
    const uint16_t xmax = 4U;
    const uint16_t lut_tbl_max_idx = 256U;
    const float16x8_t t = vmulq(ax, (float16_t)(lut_tbl_max_idx / xmax));
    uint16x8_t idx = vcvtmq_u16_f16(t);
    idx = vminq(idx, vdupq_n_u16(255U));
    const float16x8_t frac = vsubq(t, vcvtq_f16_u16(idx));
    const float16x8_t y0 = vldrhq_gather_shifted_offset((const float16_t *)arm_nn_tanh_lut256_f16, idx);
    const float16x8_t y1 = vldrhq_gather_shifted_offset((const float16_t *)arm_nn_tanh_lut256_f16, vaddq(idx, 1U));
    float16x8_t y = vfmaq(y0, vsubq(y1, y0), frac);
    y = vpselq(vdupq_n_f16((float16_t)1.0f), y, sat_p);
    return vnegq_m(y, y, vcmpltq(x, (float16_t)0.0f));
}

__STATIC_INLINE float16x8_t arm_nn_vhardswish_mve_f16(float16x8_t x)
{
    float16x8_t t = vfmaq(vdupq_n_f16((float16_t)0.5f), x, (float16_t)(1.0f / 6.0f));
    t = vmaxnmq(t, vdupq_n_f16((float16_t)0.0f));
    t = vminnmq(t, vdupq_n_f16((float16_t)1.0f));
    return vmulq(x, t);
}
    #endif

__STATIC_INLINE void
arm_nn_vector_clamp_f16(float16_t *data, int32_t block_size, float16_t activation_min, float16_t activation_max)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t vmin = vdupq_n_f16(activation_min);
    const float16x8_t vmax = vdupq_n_f16(activation_max);
    for (int32_t i = 0; i < block_size; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(block_size - i));
        float16x8_t v = vld1q_z(data + i, p);
        v = arm_nn_clamp_mve_f16(v, vmin, vmax);
        vst1q_p(data + i, v, p);
    }
    #else
    for (int32_t i = 0; i < block_size; ++i)
    {
        data[i] = arm_nn_clamp_scalar_f16(data[i], activation_min, activation_max);
    }
    #endif
}

#endif /* ARM_NN_ENABLE_F16 */

#endif /* ARM_NN_ACTIVATION_FLT_H */
