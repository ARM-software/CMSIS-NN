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
 * Title:        arm_nn_depthwise_conv3x3_f32.c
 * Description:  Support: depthwise conv 3x3 kernels for f32
 *
 * $Date:        30 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

__STATIC_INLINE void depthwise_init_vec_f32(float32_t *dst, const float32_t *bias, int32_t channels)
{
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (bias)
    {
        for (int32_t c = 0; c < channels; c += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(channels - c));
            vst1q_p(dst + c, vld1q_z(bias + c, p), p);
        }
    }
    else
    {
        const float32x4_t vzero = vdupq_n_f32(0.0f);
        for (int32_t c = 0; c < channels; c += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(channels - c));
            vst1q_p(dst + c, vzero, p);
        }
    }
#else
    if (bias)
    {
        for (int32_t c = 0; c < channels; ++c)
        {
            dst[c] = bias[c];
        }
    }
    else
    {
        for (int32_t c = 0; c < channels; ++c)
        {
            dst[c] = 0.0f;
        }
    }
#endif
}

__STATIC_INLINE void
depthwise_accumulate_vec_f32(float32_t *acc, const float32_t *input, const float32_t *kernel, int32_t channels)
{
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t c = 0; c < channels; c += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(channels - c));
        float32x4_t vacc = vld1q_z(acc + c, p);
        vacc = vfmaq(vacc, vld1q_z(input + c, p), vld1q_z(kernel + c, p));
        vst1q_p(acc + c, vacc, p);
    }
#else
    for (int32_t c = 0; c < channels; ++c)
    {
        acc[c] += input[c] * kernel[c];
    }
#endif
}

__STATIC_INLINE void depthwise_clamp_vec_f32(float32_t *data, int32_t channels, float32_t act_min, float32_t act_max)
{
    arm_nn_vector_clamp_f32(data, channels, act_min, act_max);
}

static void depthwise_conv3x3_nhwc_valid_f32(const float32_t *x_nhwc,
                                             int32_t batches,
                                             int32_t in_c,
                                             int32_t in_h,
                                             int32_t in_w,
                                             const float32_t *kernel,
                                             const float32_t *b,
                                             float32_t *out,
                                             int32_t out_h,
                                             int32_t out_w,
                                             float32_t act_min,
                                             float32_t act_max)
{
    const size_t in_row_stride = (size_t)in_w * (size_t)in_c;
    const size_t in_batch_stride = (size_t)in_h * in_row_stride;
    const size_t out_row_stride = (size_t)out_w * (size_t)in_c;
    const size_t out_batch_stride = (size_t)out_h * out_row_stride;
    const float32_t *w0 = kernel;
    const float32_t *w1 = w0 + in_c;
    const float32_t *w2 = w1 + in_c;
    const float32_t *w3 = w2 + in_c;
    const float32_t *w4 = w3 + in_c;
    const float32_t *w5 = w4 + in_c;
    const float32_t *w6 = w5 + in_c;
    const float32_t *w7 = w6 + in_c;
    const float32_t *w8 = w7 + in_c;

    for (int32_t batch = 0; batch < batches; ++batch)
    {
        const float32_t *input_batch = x_nhwc + (size_t)batch * in_batch_stride;
        float32_t *output_batch = out + (size_t)batch * out_batch_stride;

        for (int32_t oy = 0; oy < out_h; ++oy)
        {
            const float32_t *x0 = input_batch + (size_t)oy * in_row_stride;
            const float32_t *x1 = x0 + in_row_stride;
            const float32_t *x2 = x1 + in_row_stride;
            float32_t *y = output_batch + (size_t)oy * out_row_stride;

            for (int32_t ox = 0; ox < out_w; ++ox)
            {
                float32_t *out_px = y + (size_t)ox * (size_t)in_c;
                depthwise_init_vec_f32(out_px, b, in_c);
                depthwise_accumulate_vec_f32(out_px, x0 + (size_t)(ox + 0) * (size_t)in_c, w0, in_c);
                depthwise_accumulate_vec_f32(out_px, x0 + (size_t)(ox + 1) * (size_t)in_c, w1, in_c);
                depthwise_accumulate_vec_f32(out_px, x0 + (size_t)(ox + 2) * (size_t)in_c, w2, in_c);
                depthwise_accumulate_vec_f32(out_px, x1 + (size_t)(ox + 0) * (size_t)in_c, w3, in_c);
                depthwise_accumulate_vec_f32(out_px, x1 + (size_t)(ox + 1) * (size_t)in_c, w4, in_c);
                depthwise_accumulate_vec_f32(out_px, x1 + (size_t)(ox + 2) * (size_t)in_c, w5, in_c);
                depthwise_accumulate_vec_f32(out_px, x2 + (size_t)(ox + 0) * (size_t)in_c, w6, in_c);
                depthwise_accumulate_vec_f32(out_px, x2 + (size_t)(ox + 1) * (size_t)in_c, w7, in_c);
                depthwise_accumulate_vec_f32(out_px, x2 + (size_t)(ox + 2) * (size_t)in_c, w8, in_c);
                depthwise_clamp_vec_f32(out_px, in_c, act_min, act_max);
            }
        }
    }
}

static void depthwise_conv3x3_nhwc_generic_f32(const float32_t *x_nhwc,
                                               int32_t batches,
                                               int32_t in_c,
                                               int32_t in_h,
                                               int32_t in_w,
                                               const float32_t *kernel,
                                               const float32_t *b,
                                               float32_t *out,
                                               int32_t stride_x,
                                               int32_t stride_y,
                                               int32_t pad_x,
                                               int32_t pad_y,
                                               int32_t out_h,
                                               int32_t out_w,
                                               float32_t act_min,
                                               float32_t act_max)
{
    const size_t in_row_stride = (size_t)in_w * (size_t)in_c;
    const size_t in_batch_stride = (size_t)in_h * in_row_stride;
    const size_t out_row_stride = (size_t)out_w * (size_t)in_c;
    const size_t out_batch_stride = (size_t)out_h * out_row_stride;
    const float32_t *w0 = kernel;
    const float32_t *w1 = w0 + in_c;
    const float32_t *w2 = w1 + in_c;
    const float32_t *w3 = w2 + in_c;
    const float32_t *w4 = w3 + in_c;
    const float32_t *w5 = w4 + in_c;
    const float32_t *w6 = w5 + in_c;
    const float32_t *w7 = w6 + in_c;
    const float32_t *w8 = w7 + in_c;

    for (int32_t batch = 0; batch < batches; ++batch)
    {
        const float32_t *input_batch = x_nhwc + (size_t)batch * in_batch_stride;
        float32_t *output_batch = out + (size_t)batch * out_batch_stride;

        for (int32_t oy = 0; oy < out_h; ++oy)
        {
            const int32_t base_y = oy * stride_y - pad_y;
            const int32_t y0_idx = base_y;
            const int32_t y1_idx = base_y + 1;
            const int32_t y2_idx = base_y + 2;
            const bool y0_valid = (uint32_t)y0_idx < (uint32_t)in_h;
            const bool y1_valid = (uint32_t)y1_idx < (uint32_t)in_h;
            const bool y2_valid = (uint32_t)y2_idx < (uint32_t)in_h;
            const float32_t *x0 = y0_valid ? (input_batch + (size_t)y0_idx * in_row_stride) : NULL;
            const float32_t *x1 = y1_valid ? (input_batch + (size_t)y1_idx * in_row_stride) : NULL;
            const float32_t *x2 = y2_valid ? (input_batch + (size_t)y2_idx * in_row_stride) : NULL;
            float32_t *y = output_batch + (size_t)oy * out_row_stride;

            for (int32_t ox = 0; ox < out_w; ++ox)
            {
                const int32_t base_x = ox * stride_x - pad_x;
                const int32_t x0_idx = base_x;
                const int32_t x1_idx = base_x + 1;
                const int32_t x2_idx = base_x + 2;
                float32_t *out_px = y + (size_t)ox * (size_t)in_c;

                depthwise_init_vec_f32(out_px, b, in_c);

                if (y0_valid)
                {
                    if ((uint32_t)x0_idx < (uint32_t)in_w)
                    {
                        depthwise_accumulate_vec_f32(out_px, x0 + (size_t)x0_idx * (size_t)in_c, w0, in_c);
                    }
                    if ((uint32_t)x1_idx < (uint32_t)in_w)
                    {
                        depthwise_accumulate_vec_f32(out_px, x0 + (size_t)x1_idx * (size_t)in_c, w1, in_c);
                    }
                    if ((uint32_t)x2_idx < (uint32_t)in_w)
                    {
                        depthwise_accumulate_vec_f32(out_px, x0 + (size_t)x2_idx * (size_t)in_c, w2, in_c);
                    }
                }

                if (y1_valid)
                {
                    if ((uint32_t)x0_idx < (uint32_t)in_w)
                    {
                        depthwise_accumulate_vec_f32(out_px, x1 + (size_t)x0_idx * (size_t)in_c, w3, in_c);
                    }
                    if ((uint32_t)x1_idx < (uint32_t)in_w)
                    {
                        depthwise_accumulate_vec_f32(out_px, x1 + (size_t)x1_idx * (size_t)in_c, w4, in_c);
                    }
                    if ((uint32_t)x2_idx < (uint32_t)in_w)
                    {
                        depthwise_accumulate_vec_f32(out_px, x1 + (size_t)x2_idx * (size_t)in_c, w5, in_c);
                    }
                }

                if (y2_valid)
                {
                    if ((uint32_t)x0_idx < (uint32_t)in_w)
                    {
                        depthwise_accumulate_vec_f32(out_px, x2 + (size_t)x0_idx * (size_t)in_c, w6, in_c);
                    }
                    if ((uint32_t)x1_idx < (uint32_t)in_w)
                    {
                        depthwise_accumulate_vec_f32(out_px, x2 + (size_t)x1_idx * (size_t)in_c, w7, in_c);
                    }
                    if ((uint32_t)x2_idx < (uint32_t)in_w)
                    {
                        depthwise_accumulate_vec_f32(out_px, x2 + (size_t)x2_idx * (size_t)in_c, w8, in_c);
                    }
                }

                depthwise_clamp_vec_f32(out_px, in_c, act_min, act_max);
            }
        }
    }
}

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
                                       float32_t act_max)
{
    if (stride_x == 1 && stride_y == 1 && pad_x == 0 && pad_y == 0)
    {
        depthwise_conv3x3_nhwc_valid_f32(
            x_nhwc, batches, in_c, in_h, in_w, kernel, b, out, out_h, out_w, act_min, act_max);
        return;
    }

    depthwise_conv3x3_nhwc_generic_f32(x_nhwc,
                                       batches,
                                       in_c,
                                       in_h,
                                       in_w,
                                       kernel,
                                       b,
                                       out,
                                       stride_x,
                                       stride_y,
                                       pad_x,
                                       pad_y,
                                       out_h,
                                       out_w,
                                       act_min,
                                       act_max);
}

/**
 * @} end of supportConvolution group
 */
