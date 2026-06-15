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
 * Title:        arm_nnsupportfunctions_flt.h
 * Description:  Floating-point support API extensions for CMSIS-NN
 *
 * $Date:        17 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NNSUPPORTFUNCTIONS_FLT_H
#define ARM_NNSUPPORTFUNCTIONS_FLT_H

#include "Internal/arm_nn_compiler.h"
#include "arm_nn_types_flt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Floating-point internal support APIs. */

/**
 * @addtogroup groupSupport
 * @{
 */

#if ARM_NN_ENABLE_F32

/**
 * @brief Polynomial coefficients used by the float32 MVE exp approximation.
 */
extern const float32_t arm_nn_exp_poly_coeffs_f32[8];

/**
 * @brief LUT for `2^(i/256)` used by the float32 LUT softmax approximation.
 *
 * Stores 257 samples for `i = 0..256` so interpolation can safely read
 * `lut[idx + 1]` while indexing the 256 fractional segments.
 */
extern const float32_t arm_nn_exp2_lut256_f32[257];

__STATIC_INLINE int32_t arm_nn_softmax_floor_to_int_f32(float32_t x)
{
    const int32_t n = (int32_t)x;
    return (x < (float32_t)n) ? (n - 1) : n;
}

__STATIC_INLINE float32_t arm_nn_softmax_fp32_from_bits(uint32_t bits)
{
    union
    {
        uint32_t u;
        float32_t f;
    } cvt;
    cvt.u = bits;
    return cvt.f;
}

__STATIC_INLINE float32_t arm_nn_softmax_exp2i_f32(int32_t n)
{
    const int32_t float32_min_normal_exponent = -126;
    const int32_t float32_max_finite_exponent = 127;
    const int32_t float32_exponent_bias = 127;
    const int32_t float32_mantissa_bits = 23;

    n = CLAMP(n, float32_max_finite_exponent, float32_min_normal_exponent);
    return arm_nn_softmax_fp32_from_bits((uint32_t)(n + float32_exponent_bias) << float32_mantissa_bits);
}

/*
 * Taylor/Estrin exp approximation on r in [-ln2/2, ln2/2].
 * Coefficients come from the Maclaurin series of exp(r):
 *   exp(r) ~= 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5! + r^6/6!
 * Grouped via Estrin to reduce dependency depth:
 *   p = (1 + r) + r^2*(1/2 + r/6) + r^4*(1/24 + r/120) + r^6*(1/720)
 *
 * Range reduction follows:
 *   x = n * ln(2) + r,  exp(x) = exp(r) * 2^n
 */
__STATIC_INLINE float32_t arm_nn_softmax_exp_taylor_f32(float32_t x)
{
    const float32_t max_value = 80.0f;
    const float32_t min_value = -80.0f;
    const float32_t log2e = 1.44269504088896341f;
    const float32_t ln2 = 0.69314718055994531f;

    x = CLAMP(x, max_value, min_value);

    const float32_t t = x * log2e;
    const int32_t n = (t >= 0.0f) ? (int32_t)(t + 0.5f) : (int32_t)(t - 0.5f);
    const float32_t r = x - (float32_t)n * ln2;

    const float32_t r2 = r * r;
    const float32_t r4 = r2 * r2;
    const float32_t r6 = r4 * r2;

    const float32_t t0 = 1.0f + r;
    const float32_t t1 = 0.5f + (1.0f / 6.0f) * r;
    const float32_t t2 = (1.0f / 24.0f) + (1.0f / 120.0f) * r;
    const float32_t t3 = (1.0f / 720.0f);

    return (t0 + t1 * r2 + t2 * r4 + t3 * r6) * arm_nn_softmax_exp2i_f32(n);
}

__STATIC_INLINE float32_t arm_nn_softmax_exp_lut_f32(float32_t x)
{
    const float32_t max_value = 80.0f;
    const float32_t min_value = -80.0f;
    const float32_t log2e = 1.44269504088896341f;
    const float32_t exp2_lut_segments = 256.0f;
    const int32_t exp2_lut_max_index = 255;

    x = CLAMP(x, max_value, min_value);

    const float32_t t = x * log2e;
    const int32_t n = arm_nn_softmax_floor_to_int_f32(t);
    const float32_t f = t - (float32_t)n;
    const float32_t idx_f = f * exp2_lut_segments;
    int32_t idx = (int32_t)idx_f;

    if (idx < 0)
    {
        idx = 0;
    }
    else if (idx > exp2_lut_max_index)
    {
        idx = exp2_lut_max_index;
    }

    const float32_t frac = idx_f - (float32_t)idx;
    const float32_t y0 = arm_nn_exp2_lut256_f32[idx];
    const float32_t y1 = arm_nn_exp2_lut256_f32[idx + 1];
    return (y0 + (y1 - y0) * frac) * arm_nn_softmax_exp2i_f32(n);
}

__STATIC_INLINE float32_t arm_nn_softmax_exp_scalar_f32(float32_t x)
{
    #if defined(ARM_NN_USE_EXP_TAYLOR)
    return arm_nn_softmax_exp_taylor_f32(x);
    #else
    return arm_nn_softmax_exp_lut_f32(x);
    #endif
}

#endif

#if ARM_NN_ENABLE_F32

/**
 * @brief LUT for tanh(x) sampled over `x in [0, 4]` for float32 helpers.
 *
 * Stores 257 samples so interpolation can safely read `lut[idx + 1]` while
 * indexing the 256 fractional segments across the interval.
 */
extern const float32_t arm_nn_tanh_lut256_f32[257];

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
/**
 * @brief Reduce a float32 MVE vector with addition.
 */
__STATIC_INLINE float32_t arm_nn_vec_reduce_add_f32(float32x4_t v)
{
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

/**
 * @brief MVE float32 exp approximation used by float softmax paths.
 */
__STATIC_INLINE float32x4_t arm_nn_vexpq_poly_mve_f32(float32x4_t x)
{
    const int32x4_t m = vcvtq_s32_f32(vmulq(x, 1.4426950408f));
    const float32x4_t val = vfmsq(x, vcvtq_f32_s32(m), vdupq_n_f32(0.6931471805f));

    const float32x4_t a = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f32[4]), val, arm_nn_exp_poly_coeffs_f32[0]);
    const float32x4_t b = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f32[6]), val, arm_nn_exp_poly_coeffs_f32[2]);
    const float32x4_t c = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f32[5]), val, arm_nn_exp_poly_coeffs_f32[1]);
    const float32x4_t d = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f32[7]), val, arm_nn_exp_poly_coeffs_f32[3]);
    const float32x4_t x2 = vmulq(val, val);
    const float32x4_t x4 = vmulq(x2, x2);
    float32x4_t poly = vfmaq(vfmaq(a, b, x2), vfmaq(c, d, x2), x4);

    poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vdupq_m(poly, 0.0f, vcmpltq(m, -126));
    return poly;
}
    #endif

/**
 * @brief Copy a float32 vector.
 * @param[out] dst        Destination buffer.
 * @param[in]  src        Source buffer.
 * @param[in]  block_size Number of elements to copy.
 */
__STATIC_FORCEINLINE void
arm_memcpy_f32(float32_t *__RESTRICT dst, const float32_t *__RESTRICT src, uint32_t block_size)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    __asm volatile("   wlstp.32                lr, %[cnt], 1f             \n"
                   "2:                                                    \n"
                   "   vldrw.32                q0, [%[in]], #16           \n"
                   "   vstrw.32                q0, [%[out]], #16          \n"
                   "   letp                    lr, 2b                     \n"
                   "1:                                                    \n"
                   : [in] "+r"(src), [out] "+r"(dst)
                   : [cnt] "r"(block_size)
                   : "q0", "memory", "r14");
    #else
    __builtin_memcpy(dst, src, (size_t)block_size * sizeof(float32_t));
    #endif
}

/**
 * @brief Set a float32 vector to a constant value.
 * @param[out] dst        Destination buffer.
 * @param[in]  val        Fill value.
 * @param[in]  block_size Number of elements to write.
 */
__STATIC_FORCEINLINE void arm_memset_f32(float32_t *__RESTRICT dst, const float32_t val, uint32_t block_size)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t vec = vdupq_n_f32(val);
    uint32_t i = 0;
    for (; i + 4U <= block_size; i += 4U)
    {
        vst1q(dst + i, vec);
    }
    if (i < block_size)
    {
        const mve_pred16_t p = vctp32q(block_size - i);
        vst1q_p(dst + i, vec, p);
    }
    #else
    for (uint32_t i = 0; i < block_size; ++i)
    {
        dst[i] = val;
    }
    #endif
}

/**
 * @brief Specialized NHWC depthwise 1D kernel for `k=3`, `ch_mult=1` (float32).
 */
void arm_nn_depthwise_conv1d_k3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                         int32_t in_c,
                                         int32_t in_w,
                                         const float32_t *__RESTRICT kernel,
                                         const float32_t *__RESTRICT b,
                                         float32_t *__RESTRICT out,
                                         int32_t out_w);

/**
 * @brief Specialized NHWC depthwise `3x3` kernel (float32, `ch_mult=1`).
 */
void arm_nn_depthwise_conv3x3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                       int32_t batches,
                                       int32_t in_c,
                                       int32_t in_h,
                                       int32_t in_w,
                                       const float32_t *__RESTRICT kernel,
                                       const float32_t *__RESTRICT b,
                                       float32_t *__RESTRICT out,
                                       int32_t stride_x,
                                       int32_t stride_y,
                                       int32_t pad_x,
                                       int32_t pad_y,
                                       int32_t out_h,
                                       int32_t out_w,
                                       float32_t act_min,
                                       float32_t act_max);

/**
 * @brief Generic depthwise helper with packed lhs tiles and transposed rhs layout (float32).
 */
arm_cmsis_nn_status arm_nn_depthwise_conv_nt_t_f32(const float32_t *__RESTRICT lhs,
                                                   const float32_t *__RESTRICT rhs,
                                                   const float32_t *__RESTRICT bias,
                                                   float32_t *__RESTRICT out,
                                                   int32_t lhs_rows,
                                                   int32_t total_ch,
                                                   int32_t row_x_col,
                                                   int32_t out_row_stride,
                                                   float32_t activation_min,
                                                   float32_t activation_max);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=5` (float32).
 */
void arm_nn_conv1d_k5_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float32_t *__RESTRICT kernel,
                               const float32_t *__RESTRICT b,
                               float32_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=5` (float32, packed weights).
 *
 * The packed kernel uses the same `NTxN` RHS layout as
 * `arm_nn_mat_mult_nt_n_packed_f32`, i.e. `[(5 * in_c)][out_c_block_of_4]`.
 */
void arm_nn_conv1d_k5_packed_f32(const float32_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float32_t *__RESTRICT kernel_packed,
                                 const float32_t *__RESTRICT b,
                                 float32_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=3` (float32).
 */
void arm_nn_conv1d_k3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float32_t *__RESTRICT kernel,
                               const float32_t *__RESTRICT b,
                               float32_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=3` (float32, packed weights).
 *
 * The packed kernel uses the same `NTxN` RHS layout as
 * `arm_nn_mat_mult_nt_n_packed_f32`, i.e. `[(3 * in_c)][out_c_block_of_4]`.
 */
void arm_nn_conv1d_k3_packed_f32(const float32_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float32_t *__RESTRICT kernel_packed,
                                 const float32_t *__RESTRICT b,
                                 float32_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=3`, `s=3` (float32).
 */
void arm_nn_maxpool1d_k3s3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float32_t *__RESTRICT out,
                                    int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=2`, `s=2` without output clamp (float32).
 */
void arm_nn_maxpool1d_k2s2_nhwc_noclip_f32(const float32_t *__RESTRICT x_nhwc,
                                           int32_t in_c,
                                           int32_t in_w,
                                           float32_t *__RESTRICT out,
                                           int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=2`, `s=2` with clamp (float32).
 */
void arm_nn_maxpool1d_k2s2_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float32_t *__RESTRICT out,
                                    int32_t out_w,
                                    float32_t act_min,
                                    float32_t act_max);

/**
 * @brief Matrix multiply with non-transposed lhs and transposed rhs rows (float32).
 *
 * @param[in]  lhs                Left-hand matrix stored row-major.
 * @param[in]  rhs                Right-hand matrix stored row-major, one row per output channel.
 * @param[in]  bias               Optional bias vector.
 * @param[out] dst                Output matrix.
 * @param[in]  lhs_rows           Number of rows in @p lhs.
 * @param[in]  rhs_rows           Number of rows in @p rhs.
 * @param[in]  rhs_cols           Number of columns in @p rhs.
 * @param[in]  row_address_offset Output row stride, expressed in elements.
 * @param[in]  activation_min     Lower clamp bound.
 * @param[in]  activation_max     Upper clamp bound.
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_f32(const float32_t *__RESTRICT lhs,
                                             const float32_t *__RESTRICT rhs,
                                             const float32_t *__RESTRICT bias,
                                             float32_t *__RESTRICT dst,
                                             int32_t lhs_rows,
                                             int32_t rhs_rows,
                                             int32_t rhs_cols,
                                             int32_t row_address_offset,
                                             float32_t activation_min,
                                             float32_t activation_max);

/**
 * @brief Matrix multiply with non-transposed lhs and packed non-transposed rhs (float32).
 *
 * @param[in]  lhs                Left-hand matrix stored row-major with logical shape `[lhs_rows, rhs_cols]`.
 * @param[in]  rhs_packed         Right-hand matrix with logical shape `[rhs_cols, rhs_rows]`, packed in column blocks
 *                                of 4. The final block uses the same packed stride and inactive tail lanes are ignored.
 * @param[in]  bias               Optional bias vector.
 * @param[out] dst                Output matrix.
 * @param[in]  lhs_rows           Number of rows in @p lhs.
 * @param[in]  rhs_rows           Number of logical output columns in the unpacked rhs matrix.
 * @param[in]  rhs_cols           Shared reduction dimension `K`.
 * @param[in]  row_address_offset Output row stride, expressed in elements.
 * @param[in]  activation_min     Lower clamp bound.
 * @param[in]  activation_max     Upper clamp bound.
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_n_packed_f32(const float32_t *__RESTRICT lhs,
                                                    const float32_t *__RESTRICT rhs_packed,
                                                    const float32_t *__RESTRICT bias,
                                                    float32_t *__RESTRICT dst,
                                                    int32_t lhs_rows,
                                                    int32_t rhs_rows,
                                                    int32_t rhs_cols,
                                                    int32_t row_address_offset,
                                                    float32_t activation_min,
                                                    float32_t activation_max);

/**
 * @brief Pack a single convolution patch into one row of a contiguous float32 patch matrix.
 *
 * Developers familiar with im2row/im2col terminology can think of this as packing one output patch into one row.
 */
void arm_nn_pack_conv_patch_f32(const float32_t *__RESTRICT input,
                                int32_t in_h,
                                int32_t in_w,
                                int32_t in_c,
                                int32_t kernel_h,
                                int32_t kernel_w,
                                int32_t stride_h,
                                int32_t stride_w,
                                int32_t pad_h,
                                int32_t pad_w,
                                int32_t dilation_h,
                                int32_t dilation_w,
                                int32_t out_y,
                                int32_t out_x,
                                float32_t pad_value,
                                float32_t *__RESTRICT patch_row);

/**
 * @brief Specialized softmax helper for a single float32 row of length 2.
 */
void arm_nn_softmax_1x2_f32(const float32_t in[2], float32_t out[2]);

#endif /* ARM_NN_ENABLE_F32 */

#if ARM_NN_ENABLE_F16

/**
 * @brief Polynomial coefficients used by the float16 MVE exp approximation.
 *
 * The float16 MVE helper evaluates the polynomial in widened float32 lanes,
 * but it uses a dedicated coefficient table to keep the float16 path isolated
 * from the float32 feature gate and softmax support stack.
 */
extern const float32_t arm_nn_exp_poly_coeffs_f16[8];

/**
 * @brief Coefficients used by the float16 tanh rational approximation.
 */
extern const float32_t arm_nn_tanh_approx_coeffs_f16[3];

/**
 * @brief Quantized binary16 LUT for `2^(i/256)` used by float16 helpers.
 *
 * Stores 257 samples for `i = 0..256` so interpolation can safely read
 * `lut[idx + 1]` while indexing the 256 fractional segments.
 */
extern const uint16_t arm_nn_exp2_lut256_f16[257];

/**
 * @brief Quantized binary16 LUT for tanh(x) with `x in [0, 4]`.
 *
 * Stores 257 samples so interpolation can safely read `lut[idx + 1]` while
 * indexing the 256 fractional segments across the interval.
 */
extern const uint16_t arm_nn_tanh_lut256_f16[257];

__STATIC_INLINE float16_t arm_nn_softmax_fp16_from_bits(uint16_t bits)
{
    union
    {
        uint16_t u;
        float16_t f;
    } cvt;
    cvt.u = bits;
    return cvt.f;
}

__STATIC_INLINE int32_t arm_nn_softmax_floor_to_int_f16(float16_t x)
{
    const float32_t x_f32 = (float32_t)x;
    const int32_t n = (int32_t)x_f32;
    return (x_f32 < (float32_t)n) ? (n - 1) : n;
}

__STATIC_INLINE float16_t arm_nn_softmax_exp2i_f16(int32_t n)
{
    const int32_t float16_min_normal_exponent = -14;
    const int32_t float16_max_finite_exponent = 15;
    const int32_t float16_exponent_bias = 15;
    const int32_t float16_mantissa_bits = 10;

    n = CLAMP(n, float16_max_finite_exponent, float16_min_normal_exponent);
    return arm_nn_softmax_fp16_from_bits((uint16_t)((n + float16_exponent_bias) << float16_mantissa_bits));
}

/*
 * Taylor/Estrin exp approximation for float16 softmax helpers.
 * The evaluation uses float32 intermediates to keep the approximation stable,
 * but it is fully independent from the float32 softmax support tables.
 */
__STATIC_INLINE float16_t arm_nn_softmax_exp_taylor_f16(float16_t x)
{
    const float32_t max_value = 80.0f;
    const float32_t min_value = -80.0f;
    const float32_t log2e = 1.44269504088896341f;
    const float32_t ln2 = 0.69314718055994531f;

    float32_t x_f32 = (float32_t)x;
    x_f32 = CLAMP(x_f32, max_value, min_value);

    const float32_t t = x_f32 * log2e;
    const int32_t n = (t >= 0.0f) ? (int32_t)(t + 0.5f) : (int32_t)(t - 0.5f);
    const float32_t r = x_f32 - (float32_t)n * ln2;

    const float32_t r2 = r * r;
    const float32_t r4 = r2 * r2;
    const float32_t r6 = r4 * r2;

    const float32_t t0 = 1.0f + r;
    const float32_t t1 = 0.5f + (1.0f / 6.0f) * r;
    const float32_t t2 = (1.0f / 24.0f) + (1.0f / 120.0f) * r;
    const float32_t t3 = (1.0f / 720.0f);

    const float32_t poly = t0 + t1 * r2 + t2 * r4 + t3 * r6;
    return (float16_t)(poly * (float32_t)arm_nn_softmax_exp2i_f16(n));
}

__STATIC_INLINE float16_t arm_nn_softmax_exp_lut_f16(float16_t x)
{
    const float32_t max_value = 80.0f;
    const float32_t min_value = -80.0f;
    const float32_t log2e = 1.44269504088896341f;
    const float32_t exp2_lut_segments = 256.0f;
    const int32_t exp2_lut_max_index = 255;

    float32_t x_f32 = (float32_t)x;
    x_f32 = CLAMP(x_f32, max_value, min_value);

    const float32_t t = x_f32 * log2e;
    const int32_t n = arm_nn_softmax_floor_to_int_f16((float16_t)t);
    const float32_t f = t - (float32_t)n;
    const float32_t idx_f = f * exp2_lut_segments;
    int32_t idx = (int32_t)idx_f;

    if (idx < 0)
    {
        idx = 0;
    }
    else if (idx > exp2_lut_max_index)
    {
        idx = exp2_lut_max_index;
    }

    const float32_t frac = idx_f - (float32_t)idx;
    const float32_t y0 = (float32_t)arm_nn_softmax_fp16_from_bits(arm_nn_exp2_lut256_f16[idx]);
    const float32_t y1 = (float32_t)arm_nn_softmax_fp16_from_bits(arm_nn_exp2_lut256_f16[idx + 1]);
    return (float16_t)((y0 + (y1 - y0) * frac) * (float32_t)arm_nn_softmax_exp2i_f16(n));
}

__STATIC_INLINE float16_t arm_nn_softmax_exp_scalar_f16(float16_t x)
{
    #if defined(ARM_NN_USE_EXP_TAYLOR)
    return arm_nn_softmax_exp_taylor_f16(x);
    #else
    return arm_nn_softmax_exp_lut_f16(x);
    #endif
}

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
/**
 * @brief Reduce a float16 MVE vector with addition.
 */
__STATIC_INLINE float16_t arm_nn_vec_reduce_add_f16(float16x8_t in)
{
    float16x8_t tmp_vec = (float16x8_t)vrev32q_s16((int16x8_t)in);
    in = vaddq(tmp_vec, in);
    tmp_vec = (float16x8_t)vrev64q_s32((int32x4_t)in);
    in = vaddq(tmp_vec, in);
    return (float16_t)((_Float16)vgetq_lane_f16(in, 0) + (_Float16)vgetq_lane_f16(in, 4));
}

/**
 * @brief MVE float16 exp approximation used by float softmax paths.
 */
__STATIC_INLINE float16x8_t arm_nn_vexpq_poly_mve_f16(float16x8_t x)
{
    const float32x4_t x_lo = vcvtbq_f32_f16(x);
    const float32x4_t x_hi = vcvttq_f32_f16(x);
    const int32x4_t m_lo = vcvtq_s32_f32(vmulq(x_lo, 1.4426950408f));
    const int32x4_t m_hi = vcvtq_s32_f32(vmulq(x_hi, 1.4426950408f));
    const float32x4_t val_lo = vfmsq(x_lo, vcvtq_f32_s32(m_lo), vdupq_n_f32(0.6931471805f));
    const float32x4_t val_hi = vfmsq(x_hi, vcvtq_f32_s32(m_hi), vdupq_n_f32(0.6931471805f));

    const float32x4_t a_lo = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[4]), val_lo, arm_nn_exp_poly_coeffs_f16[0]);
    const float32x4_t b_lo = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[6]), val_lo, arm_nn_exp_poly_coeffs_f16[2]);
    const float32x4_t c_lo = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[5]), val_lo, arm_nn_exp_poly_coeffs_f16[1]);
    const float32x4_t d_lo = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[7]), val_lo, arm_nn_exp_poly_coeffs_f16[3]);
    const float32x4_t a_hi = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[4]), val_hi, arm_nn_exp_poly_coeffs_f16[0]);
    const float32x4_t b_hi = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[6]), val_hi, arm_nn_exp_poly_coeffs_f16[2]);
    const float32x4_t c_hi = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[5]), val_hi, arm_nn_exp_poly_coeffs_f16[1]);
    const float32x4_t d_hi = vfmasq(vdupq_n_f32(arm_nn_exp_poly_coeffs_f16[7]), val_hi, arm_nn_exp_poly_coeffs_f16[3]);
    const float32x4_t x2_lo = vmulq(val_lo, val_lo);
    const float32x4_t x2_hi = vmulq(val_hi, val_hi);
    const float32x4_t x4_lo = vmulq(x2_lo, x2_lo);
    const float32x4_t x4_hi = vmulq(x2_hi, x2_hi);
    float32x4_t y_lo = vfmaq(vfmaq(a_lo, b_lo, x2_lo), vfmaq(c_lo, d_lo, x2_lo), x4_lo);
    float32x4_t y_hi = vfmaq(vfmaq(a_hi, b_hi, x2_hi), vfmaq(c_hi, d_hi, x2_hi), x4_hi);

    y_lo = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(y_lo), vqshlq_n_s32(m_lo, 23)));
    y_hi = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(y_hi), vqshlq_n_s32(m_hi, 23)));
    y_lo = vdupq_m(y_lo, 0.0f, vcmpltq(m_lo, -126));
    y_hi = vdupq_m(y_hi, 0.0f, vcmpltq(m_hi, -126));

    float16x8_t y = vdupq_n_f16((float16_t)0.0f);
    y = vcvtbq_f16_f32(y, y_lo);
    y = vcvttq_f16_f32(y, y_hi);
    return y;
}
    #endif

/**
 * @copydoc arm_memcpy_f32
 */
__STATIC_FORCEINLINE void
arm_memcpy_f16(float16_t *__RESTRICT dst, const float16_t *__RESTRICT src, uint32_t block_size)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    __asm volatile("   wlstp.16                lr, %[cnt], 1f             \n"
                   "2:                                                    \n"
                   "   vldrh.16                q0, [%[in]], #16           \n"
                   "   vstrh.16                q0, [%[out]], #16          \n"
                   "   letp                    lr, 2b                     \n"
                   "1:                                                    \n"
                   : [in] "+r"(src), [out] "+r"(dst)
                   : [cnt] "r"(block_size)
                   : "q0", "memory", "r14");
    #else
    __builtin_memcpy(dst, src, (size_t)block_size * sizeof(float16_t));
    #endif
}

/**
 * @copydoc arm_memset_f32
 */
__STATIC_FORCEINLINE void arm_memset_f16(float16_t *__RESTRICT dst, const float16_t val, uint32_t block_size)
{
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t vec = vdupq_n_f16(val);
    uint32_t i = 0;
    for (; i + 8U <= block_size; i += 8U)
    {
        vst1q(dst + i, vec);
    }
    if (i < block_size)
    {
        const mve_pred16_t p = vctp16q(block_size - i);
        vst1q_p(dst + i, vec, p);
    }
    #else
    for (uint32_t i = 0; i < block_size; ++i)
    {
        dst[i] = val;
    }
    #endif
}

/**
 * @brief Specialized NHWC depthwise `2x5` kernel (float16).
 */
void arm_nn_depthwise_conv2x5_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                       int32_t batches,
                                       int32_t in_c,
                                       int32_t in_w,
                                       int32_t ch_mult,
                                       const float16_t *__RESTRICT kernel,
                                       const float16_t *__RESTRICT b,
                                       float16_t *__RESTRICT out,
                                       int32_t out_w,
                                       float16_t act_min,
                                       float16_t act_max);

/**
 * @copydoc arm_nn_depthwise_conv1d_k3_nhwc_f32
 */
void arm_nn_depthwise_conv1d_k3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                         int32_t in_c,
                                         int32_t in_w,
                                         const float16_t *__RESTRICT kernel,
                                         const float16_t *__RESTRICT b,
                                         float16_t *__RESTRICT out,
                                         int32_t out_w);

/**
 * @copydoc arm_nn_depthwise_conv3x3_nhwc_f32
 */
void arm_nn_depthwise_conv3x3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                       int32_t batches,
                                       int32_t in_c,
                                       int32_t in_h,
                                       int32_t in_w,
                                       const float16_t *__RESTRICT kernel,
                                       const float16_t *__RESTRICT b,
                                       float16_t *__RESTRICT out,
                                       int32_t stride_x,
                                       int32_t stride_y,
                                       int32_t pad_x,
                                       int32_t pad_y,
                                       int32_t out_h,
                                       int32_t out_w,
                                       float16_t act_min,
                                       float16_t act_max);

/**
 * @copydoc arm_nn_depthwise_conv_nt_t_f32
 */
arm_cmsis_nn_status arm_nn_depthwise_conv_nt_t_f16(const float16_t *__RESTRICT lhs,
                                                   const float16_t *__RESTRICT rhs,
                                                   const float16_t *__RESTRICT bias,
                                                   float16_t *__RESTRICT out,
                                                   int32_t lhs_rows,
                                                   int32_t total_ch,
                                                   int32_t row_x_col,
                                                   int32_t out_row_stride,
                                                   float16_t activation_min,
                                                   float16_t activation_max);

/**
 * @copydoc arm_nn_conv1d_k5_nhwc_f32
 */
void arm_nn_conv1d_k5_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float16_t *__RESTRICT kernel,
                               const float16_t *__RESTRICT b,
                               float16_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=5` (float16, packed weights).
 *
 * The packed kernel uses the same `NTxN` RHS layout as
 * `arm_nn_mat_mult_nt_n_packed_f16`, i.e. `[(5 * in_c)][out_c_block_of_8]`.
 */
void arm_nn_conv1d_k5_packed_f16(const float16_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float16_t *__RESTRICT kernel_packed,
                                 const float16_t *__RESTRICT b,
                                 float16_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w);

/**
 * @copydoc arm_nn_conv1d_k3_nhwc_f32
 */
void arm_nn_conv1d_k3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float16_t *__RESTRICT kernel,
                               const float16_t *__RESTRICT b,
                               float16_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w);

/**
 * @brief Specialized NHWC 1D convolution kernel for `k=3` (float16, packed weights).
 *
 * The packed kernel uses the same `NTxN` RHS layout as
 * `arm_nn_mat_mult_nt_n_packed_f16`, i.e. `[(3 * in_c)][out_c_block_of_8]`.
 */
void arm_nn_conv1d_k3_packed_f16(const float16_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float16_t *__RESTRICT kernel_packed,
                                 const float16_t *__RESTRICT b,
                                 float16_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=3`, `s=3` (float16).
 */
void arm_nn_maxpool1d_k3s3_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float16_t *__RESTRICT out,
                                    int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=2`, `s=2` without output clamp (float16).
 */
void arm_nn_maxpool1d_k2s2_nhwc_noclip_f16(const float16_t *__RESTRICT x_nhwc,
                                           int32_t in_c,
                                           int32_t in_w,
                                           float16_t *__RESTRICT out,
                                           int32_t out_w);

/**
 * @brief Specialized NHWC max-pool 1D kernel for `k=2`, `s=2` with clamp (float16).
 */
void arm_nn_maxpool1d_k2s2_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                    int32_t in_c,
                                    int32_t in_w,
                                    float16_t *__RESTRICT out,
                                    int32_t out_w,
                                    float16_t act_min,
                                    float16_t act_max);

/**
 * @copydoc arm_nn_mat_mult_nt_t_f32
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_f16(const float16_t *__RESTRICT lhs,
                                             const float16_t *__RESTRICT rhs,
                                             const float16_t *__RESTRICT bias,
                                             float16_t *__RESTRICT dst,
                                             int32_t lhs_rows,
                                             int32_t rhs_rows,
                                             int32_t rhs_cols,
                                             int32_t row_address_offset,
                                             float16_t activation_min,
                                             float16_t activation_max);

/**
 * @brief Matrix multiply with non-transposed lhs and packed non-transposed rhs (float16).
 *
 * @param[in]  lhs                Left-hand matrix stored row-major with logical shape `[lhs_rows, rhs_cols]`.
 * @param[in]  rhs_packed         Right-hand matrix with logical shape `[rhs_cols, rhs_rows]`, packed in column blocks
 *                                of 8. The final block uses the same packed stride and inactive tail lanes are ignored.
 * @param[in]  bias               Optional bias vector.
 * @param[out] dst                Output matrix.
 * @param[in]  lhs_rows           Number of rows in @p lhs.
 * @param[in]  rhs_rows           Number of logical output columns in the unpacked rhs matrix.
 * @param[in]  rhs_cols           Shared reduction dimension `K`.
 * @param[in]  row_address_offset Output row stride, expressed in elements.
 * @param[in]  activation_min     Lower clamp bound.
 * @param[in]  activation_max     Upper clamp bound.
 * @return `ARM_CMSIS_NN_SUCCESS` on success or `ARM_CMSIS_NN_ARG_ERROR` on invalid arguments.
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_n_packed_f16(const float16_t *__RESTRICT lhs,
                                                    const float16_t *__RESTRICT rhs_packed,
                                                    const float16_t *__RESTRICT bias,
                                                    float16_t *__RESTRICT dst,
                                                    int32_t lhs_rows,
                                                    int32_t rhs_rows,
                                                    int32_t rhs_cols,
                                                    int32_t row_address_offset,
                                                    float16_t activation_min,
                                                    float16_t activation_max);

/**
 * @brief Update LSTM function for an iteration step using float16 input, output and state.
 *
 * @param[in]   data_in                         Data input pointer.
 * @param[in]   hidden_in                       Hidden state / recurrent input pointer. May be NULL for the first step.
 * @param[out]  hidden_out                      Hidden state / recurrent output pointer.
 * @param[in]   params                          Struct containing all information about the LSTM operator.
 * @param[in]   buffers                         Struct containing pointers to mutable cell-state storage.
 * @param[in]   batch_offset                    Number of timesteps between consecutive batches.
 * @return                                      The function returns ARM_CMSIS_NN_SUCCESS.
 */
arm_cmsis_nn_status arm_nn_lstm_step_f16(const float16_t *data_in,
                                         const float16_t *hidden_in,
                                         float16_t *hidden_out,
                                         const cmsis_nn_lstm_params_f16 *params,
                                         cmsis_nn_lstm_context_f16 *buffers,
                                         const int32_t batch_offset);

/**
 * @copydoc arm_nn_pack_conv_patch_f32
 */
void arm_nn_pack_conv_patch_f16(const float16_t *__RESTRICT input,
                                int32_t in_h,
                                int32_t in_w,
                                int32_t in_c,
                                int32_t kernel_h,
                                int32_t kernel_w,
                                int32_t stride_h,
                                int32_t stride_w,
                                int32_t pad_h,
                                int32_t pad_w,
                                int32_t dilation_h,
                                int32_t dilation_w,
                                int32_t out_y,
                                int32_t out_x,
                                float16_t pad_value,
                                float16_t *__RESTRICT patch_row);

/**
 * @copydoc arm_nn_softmax_1x2_f32
 */
void arm_nn_softmax_1x2_f16(const float16_t in[2], float16_t out[2]);

#endif /* ARM_NN_ENABLE_F16 */

#if ARM_NN_ENABLE_F32

/**
 * @brief Update LSTM function for an iteration step using float32 input, output and state.
 *
 * @param[in]   data_in                         Data input pointer.
 * @param[in]   hidden_in                       Hidden state / recurrent input pointer. May be NULL for the first step.
 * @param[out]  hidden_out                      Hidden state / recurrent output pointer.
 * @param[in]   params                          Struct containing all information about the LSTM operator.
 * @param[in]   buffers                         Struct containing pointers to mutable cell-state storage.
 * @param[in]   batch_offset                    Number of timesteps between consecutive batches.
 * @return                                      The function returns ARM_CMSIS_NN_SUCCESS.
 */
arm_cmsis_nn_status arm_nn_lstm_step_f32(const float32_t *data_in,
                                         const float32_t *hidden_in,
                                         float32_t *hidden_out,
                                         const cmsis_nn_lstm_params_f32 *params,
                                         cmsis_nn_lstm_context_f32 *buffers,
                                         const int32_t batch_offset);

#endif /* ARM_NN_ENABLE_F32 */

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif /* ARM_NNSUPPORTFUNCTIONS_FLT_H */
