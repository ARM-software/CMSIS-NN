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
 * Title:        arm_convolve_f32.c
 * Description:  Generic float32 convolution
 *
 * $Date:        31 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

/* Generic float32 convolution. */

#include "Internal/arm_conv_opt_common.h"
#include "Internal/arm_conv_opt_f32.h"
#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

__STATIC_INLINE float32_t arm_conv_dot_f32(const float32_t *lhs, const float32_t *rhs, int32_t len)
{
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (int32_t i = 0; i < len; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(len - i));
        vacc = vfmaq_m(vacc, vld1q_z(lhs + i, p), vld1q_z(rhs + i, p), p);
    }
    return arm_nn_vec_reduce_add_f32(vacc);
#else
    float32_t acc = 0.0f;
    for (int32_t i = 0; i < len; ++i)
    {
        acc += lhs[i] * rhs[i];
    }
    return acc;
#endif
}

__STATIC_INLINE bool arm_conv_nhwc_use_patch_gemm_f32(const cmsis_nn_context *ctx,
                                                      int32_t patch_len,
                                                      int32_t output_c,
                                                      int32_t output_positions)
{
    if (!ctx || !ctx->buf || ctx->size <= 0 || patch_len <= 0)
    {
        return false;
    }

    if (patch_len < ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_K || output_c < ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_OC ||
        output_positions < ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_POS)
    {
        return false;
    }

    const size_t row_bytes = (size_t)patch_len * sizeof(float32_t);
    return row_bytes > 0U && (size_t)ctx->size >= row_bytes;
}

__STATIC_INLINE bool arm_conv_nhwc_use_1x1_f32(const cmsis_nn_conv_params_f32 *conv_params,
                                               const cmsis_nn_dims *filter_dims)
{
    return conv_params && filter_dims && filter_dims->h == 1 && filter_dims->w == 1 && conv_params->padding.h == 0 &&
        conv_params->padding.w == 0;
}

__STATIC_INLINE arm_cmsis_nn_status arm_convolve_patch_mat_mul_f32(const float32_t *lhs,
                                                                   const float32_t *rhs,
                                                                   const float32_t *bias,
                                                                   float32_t *dst,
                                                                   int32_t lhs_rows,
                                                                   int32_t rhs_rows,
                                                                   int32_t rhs_cols,
                                                                   int32_t row_address_offset,
                                                                   const cmsis_nn_conv_params_f32 *conv_params)
{
    if (conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED)
    {
        return arm_nn_mat_mult_nt_n_packed_f32(lhs,
                                               rhs,
                                               bias,
                                               dst,
                                               lhs_rows,
                                               rhs_rows,
                                               rhs_cols,
                                               row_address_offset,
                                               conv_params->activation.min,
                                               conv_params->activation.max);
    }

    return arm_nn_mat_mult_nt_t_f32(lhs,
                                    rhs,
                                    bias,
                                    dst,
                                    lhs_rows,
                                    rhs_rows,
                                    rhs_cols,
                                    row_address_offset,
                                    conv_params->activation.min,
                                    conv_params->activation.max);
}

__STATIC_INLINE bool arm_conv_nhwc_use_1xn_f32(const cmsis_nn_context *ctx,
                                               const cmsis_nn_conv_params_f32 *conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims)
{
    if (!ctx || !ctx->buf || ctx->size <= 0 || !conv_params || !input_dims || !filter_dims || !output_dims)
    {
        return false;
    }

    /* This helper only selects the generic 1xN NHWC kernel family. */
    if (input_dims->h != 1 || output_dims->h != 1 || filter_dims->h != 1 || filter_dims->w <= 1 ||
        conv_params->stride.h != 1 || conv_params->stride.w <= 0 || conv_params->padding.h != 0 ||
        conv_params->dilation.h != 1 || conv_params->dilation.w != 1)
    {
        return false;
    }

#ifndef NN_DISABLE_SPECIALIZATION
    /*
     * If a direct specialization already claims the shape, let the normal
     * specialization dispatcher handle it instead of forcing the generic 1xN
     * implementation to know kernel-specific details.
     */
    if (arm_conv_spec_nhwc_f32_matches_any(ctx, conv_params, input_dims, filter_dims, output_dims))
    {
        return false;
    }
#endif

    /* Remaining 1xN shapes use the generic packed-input helper when workspace is available. */
    const int32_t buf_size =
        arm_convolve_1_x_n_f32_get_buffer_size(conv_params, input_dims, filter_dims, output_dims, ARM_NN_LAYOUT_NHWC);
    return buf_size > 0 && ctx->size >= buf_size;
}

static arm_cmsis_nn_status arm_convolve_nhwc_patch_gemm_f32(const cmsis_nn_context *ctx,
                                                            const cmsis_nn_conv_params_f32 *conv_params,
                                                            const cmsis_nn_dims *input_dims,
                                                            const float32_t *input_data,
                                                            const cmsis_nn_dims *filter_dims,
                                                            const float32_t *filter_data,
                                                            const float32_t *bias_data,
                                                            const cmsis_nn_dims *output_dims,
                                                            float32_t *output_data)
{
    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const int32_t stride_h = conv_params->stride.h;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t pad_h = conv_params->padding.h;
    const int32_t pad_w = conv_params->padding.w;
    const int32_t dil_h = conv_params->dilation.h;
    const int32_t dil_w = conv_params->dilation.w;
    const int32_t patch_len = kernel_h * kernel_w * input_c;
    const int32_t output_positions = output_h * output_w;

    const size_t row_bytes = (size_t)patch_len * sizeof(float32_t);
    if (row_bytes == 0U)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t tile_rows = ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MAX_TILE_ROWS;
    const int32_t max_rows_by_ctx = (int32_t)((size_t)ctx->size / row_bytes);
    if (max_rows_by_ctx < tile_rows)
    {
        tile_rows = max_rows_by_ctx;
    }
    if (tile_rows <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    /* Developers familiar with im2row terminology: each output position becomes one packed row here. */
    float32_t *patch_matrix = (float32_t *)ctx->buf;
    for (int32_t b = 0; b < batch; ++b)
    {
        const float32_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float32_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t pos = 0; pos < output_positions; pos += tile_rows)
        {
            const int32_t rows = ((output_positions - pos) < tile_rows) ? (output_positions - pos) : tile_rows;
            for (int32_t r = 0; r < rows; ++r)
            {
                const int32_t out_pos = pos + r;
                const int32_t out_y = out_pos / output_w;
                const int32_t out_x = out_pos - out_y * output_w;
                float32_t *patch_row = patch_matrix + (size_t)r * patch_len;
                arm_nn_pack_conv_patch_f32(input_b,
                                           input_h,
                                           input_w,
                                           input_c,
                                           kernel_h,
                                           kernel_w,
                                           stride_h,
                                           stride_w,
                                           pad_h,
                                           pad_w,
                                           dil_h,
                                           dil_w,
                                           out_y,
                                           out_x,
                                           0.0f,
                                           patch_row);
            }

            arm_cmsis_nn_status st = arm_convolve_patch_mat_mul_f32(patch_matrix,
                                                                    filter_data,
                                                                    bias_data,
                                                                    output_b + (size_t)pos * output_c,
                                                                    rows,
                                                                    output_c,
                                                                    patch_len,
                                                                    output_c,
                                                                    conv_params);
            if (st != ARM_CMSIS_NN_SUCCESS)
            {
                return st;
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_convolve_nhwc_f32(const cmsis_nn_context *ctx,
                                          const cmsis_nn_conv_params_f32 *conv_params,
                                          const cmsis_nn_dims *input_dims,
                                          const float32_t *input_data,
                                          const cmsis_nn_dims *filter_dims,
                                          const float32_t *filter_data,
                                          const cmsis_nn_dims *bias_dims,
                                          const float32_t *bias_data,
                                          const cmsis_nn_dims *output_dims,
                                          float32_t *output_data)
{
    (void)bias_dims;

    if (!conv_params || !input_dims || !filter_dims || !output_dims || !input_data || !filter_data || !output_data)
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
    const int32_t stride_h = conv_params->stride.h;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t pad_h = conv_params->padding.h;
    const int32_t pad_w = conv_params->padding.w;
    const int32_t dil_h = conv_params->dilation.h;
    const int32_t dil_w = conv_params->dilation.w;
    const int32_t patch_len = kernel_h * kernel_w * input_c;
    const int32_t output_positions = output_h * output_w;

    if (arm_conv_nhwc_use_1x1_f32(conv_params, filter_dims))
    {
        return arm_convolve_1x1_nhwc_f32(ctx,
                                         conv_params,
                                         input_dims,
                                         input_data,
                                         filter_dims,
                                         filter_data,
                                         bias_dims,
                                         bias_data,
                                         output_dims,
                                         output_data);
    }

    if (arm_conv_nhwc_use_1xn_f32(ctx, conv_params, input_dims, filter_dims, output_dims))
    {
        return arm_convolve_1_x_n_nhwc_f32(ctx,
                                           conv_params,
                                           input_dims,
                                           input_data,
                                           filter_dims,
                                           filter_data,
                                           bias_dims,
                                           bias_data,
                                           output_dims,
                                           output_data);
    }

#ifndef NN_DISABLE_SPECIALIZATION
    /*
     * Let direct specializations claim their shapes first. Packed-patch GEMM
     * remains the generic fallback for shapes that are not handled by a tuned
     * direct kernel.
     */
    ARM_CONV_DISPATCH(arm_conv_spec_nhwc_f32,
                      ARM_CONV_ARRAY_SIZE(arm_conv_spec_nhwc_f32),
                      ctx,
                      conv_params,
                      input_dims,
                      input_data,
                      filter_dims,
                      filter_data,
                      bias_dims,
                      bias_data,
                      output_dims,
                      output_data);
#endif

    const bool use_patch_gemm = arm_conv_nhwc_use_patch_gemm_f32(ctx, patch_len, output_c, output_positions);

    if (use_patch_gemm)
    {
        arm_cmsis_nn_status st = arm_convolve_nhwc_patch_gemm_f32(
            ctx, conv_params, input_dims, input_data, filter_dims, filter_data, bias_data, output_dims, output_data);
        if (st == ARM_CMSIS_NN_SUCCESS)
        {
            return st;
        }
    }

    for (int32_t b = 0; b < batch; ++b)
    {
        const float32_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float32_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t out_y = 0; out_y < output_h; ++out_y)
        {
            const int32_t in_y0 = out_y * stride_h - pad_h;
            for (int32_t out_x = 0; out_x < output_w; ++out_x)
            {
                const int32_t in_x0 = out_x * stride_w - pad_w;
                for (int32_t oc = 0; oc < output_c; ++oc)
                {
                    float32_t acc = bias_data ? bias_data[oc] : 0.0f;
                    const float32_t *w_oc = filter_data + (size_t)oc * kernel_h * kernel_w * input_c;

                    for (int32_t ky = 0; ky < kernel_h; ++ky)
                    {
                        const int32_t in_y = in_y0 + ky * dil_h;
                        if (in_y < 0 || in_y >= input_h)
                        {
                            continue;
                        }
                        for (int32_t kx = 0; kx < kernel_w; ++kx)
                        {
                            const int32_t in_x = in_x0 + kx * dil_w;
                            if (in_x < 0 || in_x >= input_w)
                            {
                                continue;
                            }
                            const float32_t *w_k = w_oc + ((size_t)ky * kernel_w + (size_t)kx) * input_c;
                            const float32_t *x = input_b + ((size_t)in_y * input_w + (size_t)in_x) * input_c;
                            acc += arm_conv_dot_f32(x, w_k, input_c);
                        }
                    }

                    acc = CLAMP(acc, conv_params->activation.max, conv_params->activation.min);
                    output_b[((size_t)out_y * output_w + (size_t)out_x) * output_c + (size_t)oc] = acc;
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_convolve_f32(const cmsis_nn_context *ctx,
                                     const cmsis_nn_conv_params_f32 *conv_params,
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

    return arm_convolve_nhwc_f32(ctx,
                                 conv_params,
                                 input_dims,
                                 input_data,
                                 filter_dims,
                                 filter_data,
                                 bias_dims,
                                 bias_data,
                                 output_dims,
                                 output_data);
}

arm_cmsis_nn_status arm_convolve_wrapper_f32(const cmsis_nn_context *ctx,
                                             const cmsis_nn_conv_params_f32 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const float32_t *input_data,
                                             const cmsis_nn_dims *filter_dims,
                                             const float32_t *filter_data,
                                             const cmsis_nn_dims *bias_dims,
                                             const float32_t *bias_data,
                                             const cmsis_nn_dims *output_dims,
                                             float32_t *output_data)
{
    return arm_convolve_f32(ctx,
                            conv_params,
                            input_dims,
                            input_data,
                            filter_dims,
                            filter_data,
                            bias_dims,
                            bias_data,
                            output_dims,
                            output_data,
                            ARM_NN_LAYOUT_NHWC);
}
/**
 * @} end of NNConv group
 */
