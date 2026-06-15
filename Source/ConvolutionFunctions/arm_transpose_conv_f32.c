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
 * Title:        arm_transpose_conv_f32.c
 * Description:  Transpose convolution (float32, NHWC)
 *
 * $Date:        23 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

arm_cmsis_nn_status arm_transpose_conv_nhwc_f32(const cmsis_nn_context *ctx,
                                                const cmsis_nn_context *output_ctx,
                                                const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float32_t *input_data,
                                                const cmsis_nn_dims *filter_dims,
                                                const float32_t *filter_data,
                                                const cmsis_nn_dims *bias_dims,
                                                const float32_t *bias_data,
                                                const cmsis_nn_dims *output_dims,
                                                float32_t *output_data)
{
    (void)ctx;
    (void)output_ctx;
    (void)bias_dims;

    if (!transpose_conv_params || !input_dims || !filter_dims || !output_dims || !input_data || !filter_data ||
        !output_data)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;

    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;

    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;

    const int32_t stride_h = transpose_conv_params->stride.h;
    const int32_t stride_w = transpose_conv_params->stride.w;
    const int32_t pad_h = transpose_conv_params->padding.h;
    const int32_t pad_w = transpose_conv_params->padding.w;
    const int32_t pad_off_h = transpose_conv_params->padding_offsets.h;
    const int32_t pad_off_w = transpose_conv_params->padding_offsets.w;
    const int32_t dil_h = transpose_conv_params->dilation.h;
    const int32_t dil_w = transpose_conv_params->dilation.w;
    const size_t kernel_oc_stride = (size_t)kernel_h * (size_t)kernel_w * (size_t)input_c;
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const uint32_t kernel_oc_stride_u32 = (uint32_t)kernel_oc_stride;
    const uint32x4_t oc_offsets = vmulq(vidupq_u32((uint32_t)0, 1), kernel_oc_stride_u32);
#endif

    for (int32_t b = 0; b < batch; ++b)
    {
        float32_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;
        const float32_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;

        for (int32_t out_y = 0; out_y < output_h; ++out_y)
        {
            for (int32_t out_x = 0; out_x < output_w; ++out_x)
            {
                float32_t *out_px = output_b + ((size_t)out_y * output_w + (size_t)out_x) * output_c;
                if (bias_data)
                {
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                    for (int32_t oc = 0; oc < output_c; oc += 4)
                    {
                        const mve_pred16_t p = vctp32q((uint32_t)(output_c - oc));
                        vst1q_p(out_px + oc, vld1q_z(bias_data + oc, p), p);
                    }
#else
                    for (int32_t oc = 0; oc < output_c; ++oc)
                    {
                        out_px[oc] = bias_data[oc];
                    }
#endif
                }
                else
                {
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                    const float32x4_t vzero = vdupq_n_f32(0.0f);
                    for (int32_t oc = 0; oc < output_c; oc += 4)
                    {
                        const mve_pred16_t p = vctp32q((uint32_t)(output_c - oc));
                        vst1q_p(out_px + oc, vzero, p);
                    }
#else
                    for (int32_t oc = 0; oc < output_c; ++oc)
                    {
                        out_px[oc] = 0.0f;
                    }
#endif
                }
            }
        }

        for (int32_t in_y = 0; in_y < input_h; ++in_y)
        {
            for (int32_t in_x = 0; in_x < input_w; ++in_x)
            {
                const float32_t *input_px = input_b + ((size_t)in_y * input_w + (size_t)in_x) * input_c;
                for (int32_t ic = 0; ic < input_c; ++ic)
                {
                    const float32_t val = input_px[ic];
                    if (val == 0.0f)
                    {
                        continue;
                    }
                    for (int32_t ky = 0; ky < kernel_h; ++ky)
                    {
                        const int32_t out_y = in_y * stride_h - pad_h + ky * dil_h + pad_off_h;
                        if ((uint32_t)out_y >= (uint32_t)output_h)
                        {
                            continue;
                        }
                        for (int32_t kx = 0; kx < kernel_w; ++kx)
                        {
                            const int32_t out_x = in_x * stride_w - pad_w + kx * dil_w + pad_off_w;
                            if ((uint32_t)out_x >= (uint32_t)output_w)
                            {
                                continue;
                            }
                            float32_t *out_px = output_b + ((size_t)out_y * output_w + (size_t)out_x) * output_c;
                            const float32_t *w_k_ic =
                                filter_data + ((size_t)ky * kernel_w + (size_t)kx) * input_c + (size_t)ic;
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                            int32_t oc = 0;
                            for (; oc < output_c; oc += 4)
                            {
                                const mve_pred16_t p = vctp32q((uint32_t)(output_c - oc));
                                const float32_t *w_oc = w_k_ic + (size_t)oc * kernel_oc_stride;
                                const float32x4_t vw =
                                    vldrwq_gather_shifted_offset_z((const float32_t *)w_oc, oc_offsets, p);
                                float32x4_t vout = vld1q_z(out_px + oc, p);
                                vout = vfmaq_m(vout, vw, val, p);
                                vst1q_p(out_px + oc, vout, p);
                            }
#else
                            int32_t oc = 0;
#endif
                            for (; oc < output_c; ++oc)
                            {
                                out_px[oc] += val * w_k_ic[(size_t)oc * kernel_oc_stride];
                            }
                        }
                    }
                }
            }
        }

        for (int32_t out_y = 0; out_y < output_h; ++out_y)
        {
            for (int32_t out_x = 0; out_x < output_w; ++out_x)
            {
                float32_t *out_px = output_b + ((size_t)out_y * output_w + (size_t)out_x) * output_c;
                arm_nn_vector_clamp_f32(
                    out_px, output_c, transpose_conv_params->activation.min, transpose_conv_params->activation.max);
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_transpose_conv_f32(const cmsis_nn_context *ctx,
                                           const cmsis_nn_context *output_ctx,
                                           const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float32_t *input_data,
                                           const cmsis_nn_dims *filter_dims,
                                           const float32_t *filter_data,
                                           const cmsis_nn_dims *bias_dims,
                                           const float32_t *bias_data,
                                           const cmsis_nn_dims *output_dims,
                                           float32_t *output_data,
                                           arm_nn_tensor_layout layout)
{
    if (layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_transpose_conv_nhwc_f32(ctx,
                                       output_ctx,
                                       transpose_conv_params,
                                       input_dims,
                                       input_data,
                                       filter_dims,
                                       filter_data,
                                       bias_dims,
                                       bias_data,
                                       output_dims,
                                       output_data);
}

arm_cmsis_nn_status arm_transpose_conv_wrapper_f32(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_context *output_ctx,
                                                   const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float32_t *input_data,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float32_t *filter_data,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float32_t *bias_data,
                                                   const cmsis_nn_dims *output_dims,
                                                   float32_t *output_data,
                                                   arm_nn_tensor_layout layout)
{
    return arm_transpose_conv_f32(ctx,
                                  output_ctx,
                                  transpose_conv_params,
                                  input_dims,
                                  input_data,
                                  filter_dims,
                                  filter_data,
                                  bias_dims,
                                  bias_data,
                                  output_dims,
                                  output_data,
                                  layout);
}

int32_t arm_transpose_conv_f32_get_buffer_size(const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *out_dims)
{
    /* Placeholder: current float32 transpose conv is bufferless. */
    /* Future optimized kernels may require scratch buffers; keep API for compatibility. */
    (void)transpose_conv_params;
    (void)input_dims;
    (void)filter_dims;
    (void)out_dims;
    return 0;
}

int32_t
arm_transpose_conv_f32_get_reverse_conv_buffer_size(const cmsis_nn_transpose_conv_params_f32 *transpose_conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims)
{
    /* Placeholder: no reverse-convolution workspace needed today. */
    /* Reserved for future implementations that may require temporary buffers. */
    (void)transpose_conv_params;
    (void)input_dims;
    (void)filter_dims;
    return 0;
}
/**
 * @} end of NNConv group
 */
