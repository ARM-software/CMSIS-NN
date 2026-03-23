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
 * Title:        arm_convolve_1x1_f16.c
 * Description:  Generic float16 1x1 convolution
 *
 * $Date:        24 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

/* Generic float16 1x1 convolution. */

#include "Internal/arm_conv1x1_opt_common.h"
#include "Internal/arm_conv1x1_opt_f16.h"
#include "Internal/arm_conv_opt_common.h"
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

__STATIC_INLINE bool arm_conv_1x1_nhwc_use_patch_gemm_f16(const cmsis_nn_context *ctx,
                                                          int32_t input_c,
                                                          int32_t output_c,
                                                          int32_t output_positions)
{
    if (!ctx || !ctx->buf || ctx->size <= 0 || input_c <= 0)
    {
        return false;
    }

    if (input_c < ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_K || output_c < ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_OC ||
        output_positions < ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_POS)
    {
        return false;
    }

    const size_t row_bytes = (size_t)input_c * sizeof(float16_t);
    return row_bytes > 0U && (size_t)ctx->size >= row_bytes;
}

__STATIC_INLINE arm_cmsis_nn_status arm_convolve_1x1_mat_mul_f16(const float16_t *lhs,
                                                                 const float16_t *rhs,
                                                                 const float16_t *bias,
                                                                 float16_t *dst,
                                                                 int32_t lhs_rows,
                                                                 int32_t rhs_rows,
                                                                 int32_t rhs_cols,
                                                                 int32_t row_address_offset,
                                                                 const cmsis_nn_conv_params_f16 *conv_params)
{
    if (conv_params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED)
    {
        return arm_nn_mat_mult_nt_n_packed_f16(lhs,
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

    return arm_nn_mat_mult_nt_t_f16(lhs,
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

static arm_cmsis_nn_status arm_convolve_1x1_nhwc_patch_gemm_f16(const cmsis_nn_context *ctx,
                                                                const cmsis_nn_conv_params_f16 *conv_params,
                                                                const cmsis_nn_dims *input_dims,
                                                                const float16_t *input_data,
                                                                const float16_t *filter_data,
                                                                const float16_t *bias_data,
                                                                const cmsis_nn_dims *output_dims,
                                                                float16_t *output_data)
{
    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t stride_h = conv_params->stride.h;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t output_positions = output_h * output_w;

    const size_t row_bytes = (size_t)input_c * sizeof(float16_t);
    if (row_bytes == 0U)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t tile_rows = ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MAX_TILE_ROWS;
    const int32_t max_rows_by_ctx = (int32_t)((size_t)ctx->size / row_bytes);
    if (max_rows_by_ctx < tile_rows)
    {
        tile_rows = max_rows_by_ctx;
    }
    if (tile_rows <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    float16_t *patch_matrix = (float16_t *)ctx->buf;
    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t pos = 0; pos < output_positions; pos += tile_rows)
        {
            const int32_t rows = ((output_positions - pos) < tile_rows) ? (output_positions - pos) : tile_rows;
            for (int32_t r = 0; r < rows; ++r)
            {
                const int32_t out_pos = pos + r;
                const int32_t out_y = out_pos / output_w;
                const int32_t out_x = out_pos - out_y * output_w;
                const int32_t in_y = out_y * stride_h;
                const int32_t in_x = out_x * stride_w;
                const int32_t src = ((in_y * input_w) + in_x) * input_c;
                arm_memcpy_f16(patch_matrix + (size_t)r * input_c, input_b + src, (uint32_t)input_c);
            }

            arm_cmsis_nn_status st = arm_convolve_1x1_mat_mul_f16(patch_matrix,
                                                                  filter_data,
                                                                  bias_data,
                                                                  output_b + (size_t)pos * output_c,
                                                                  rows,
                                                                  output_c,
                                                                  input_c,
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

arm_cmsis_nn_status arm_convolve_1x1_nhwc_f16(const cmsis_nn_context *ctx,
                                              const cmsis_nn_conv_params_f16 *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const float16_t *input_data,
                                              const cmsis_nn_dims *filter_dims,
                                              const float16_t *filter_data,
                                              const cmsis_nn_dims *bias_dims,
                                              const float16_t *bias_data,
                                              const cmsis_nn_dims *output_dims,
                                              float16_t *output_data)
{
    (void)bias_dims;

    if (!conv_params || !input_dims || !filter_dims || !output_dims || !input_data || !filter_data || !output_data)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (conv_params->padding.w != 0 || conv_params->padding.h != 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (filter_dims->w != 1 || filter_dims->h != 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t input_c = input_dims->c;
    const int32_t output_c = output_dims->c;
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t stride_h = conv_params->stride.h;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t input_hw = input_h * input_w;
    const int32_t output_positions = output_h * output_w;

    if (stride_h == 1 && stride_w == 1 && output_h == input_h && output_w == input_w)
    {
        for (int32_t b = 0; b < input_dims->n; ++b)
        {
            const float16_t *input_b = input_data + (size_t)b * input_hw * input_c;
            float16_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;
            arm_cmsis_nn_status status = arm_convolve_1x1_mat_mul_f16(
                input_b, filter_data, bias_data, output_b, input_hw, output_c, input_c, output_c, conv_params);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
        }
        return ARM_CMSIS_NN_SUCCESS;
    }

    if (arm_conv_1x1_nhwc_use_patch_gemm_f16(ctx, input_c, output_c, output_positions))
    {
        return arm_convolve_1x1_nhwc_patch_gemm_f16(
            ctx, conv_params, input_dims, input_data, filter_data, bias_data, output_dims, output_data);
    }

    for (int32_t b = 0; b < input_dims->n; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t out_y = 0; out_y < output_h; ++out_y)
        {
            const int32_t in_y = out_y * stride_h;
            for (int32_t out_x = 0; out_x < output_w; ++out_x)
            {
                const int32_t in_x = out_x * stride_w;
                const int32_t in_base = (in_y * input_w + in_x) * input_c;
                float16_t *output_pixel = output_b + ((size_t)out_y * output_w + (size_t)out_x) * output_c;

                arm_cmsis_nn_status status = arm_convolve_1x1_mat_mul_f16(input_b + in_base,
                                                                          filter_data,
                                                                          bias_data,
                                                                          output_pixel,
                                                                          1,
                                                                          output_c,
                                                                          input_c,
                                                                          output_c,
                                                                          conv_params);
                if (status != ARM_CMSIS_NN_SUCCESS)
                {
                    return status;
                }
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}
arm_cmsis_nn_status arm_convolve_1x1_f16(const cmsis_nn_context *ctx,
                                         const cmsis_nn_conv_params_f16 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const float16_t *input_data,
                                         const cmsis_nn_dims *filter_dims,
                                         const float16_t *filter_data,
                                         const cmsis_nn_dims *bias_dims,
                                         const float16_t *bias_data,
                                         const cmsis_nn_dims *output_dims,
                                         float16_t *output_data,
                                         arm_nn_tensor_layout layout)
{
    if (layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_convolve_1x1_nhwc_f16(ctx,
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
/**
 * @} end of NNConv group
 */
