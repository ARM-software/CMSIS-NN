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
 * Title:        arm_get_buffer_size_f16.c
 * Description:  Buffer size helpers (float16)
 *
 * $Date:        31 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_conv_opt_common.h"
#include "arm_get_buffer_size_common.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup NNConv
 */

/**
 * @addtogroup GetBufferSizeNNConv
 * @{
 */

/* Keep in sync with arm_depthwise_conv_f16.c fast NT_T packing tile. */
#define ARM_NN_DW_NT_T_F16_TILE_ROWS (4)

int32_t arm_depthwise_conv_f16_get_buffer_size(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout)
{
    if (!dw_conv_params || !input_dims || !filter_dims || !output_dims)
    {
        return 0;
    }

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (layout == ARM_NN_LAYOUT_NHWC && input_dims->c == 1 &&
        output_dims->c >= CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD)
    {
        const cmsis_nn_conv_params_f16 conv_params = {.stride = dw_conv_params->stride,
                                                      .padding = dw_conv_params->padding,
                                                      .dilation = dw_conv_params->dilation,
                                                      .activation = dw_conv_params->activation};
        const cmsis_nn_dims conv_filter_dims = {
            .n = output_dims->c, .h = filter_dims->h, .w = filter_dims->w, .c = input_dims->c};
        const int32_t conv_buf_size =
            arm_convolve_wrapper_f16_get_buffer_size(&conv_params, input_dims, &conv_filter_dims, output_dims);
        size_t filter_bytes = (size_t)ARM_NN_ROUND_UP(output_dims->c, 8);

        if (!arm_nn_checked_size_mul(filter_bytes, (size_t)filter_dims->h, &filter_bytes) ||
            !arm_nn_checked_size_mul(filter_bytes, (size_t)filter_dims->w, &filter_bytes) ||
            !arm_nn_checked_size_mul(filter_bytes, sizeof(float16_t), &filter_bytes))
        {
            return 0;
        }

        if (conv_buf_size > 0)
        {
            if (filter_bytes > (((size_t)-1) - (size_t)conv_buf_size))
            {
                return 0;
            }
            filter_bytes += (size_t)conv_buf_size;
        }

        return arm_nn_size_to_i32_or_zero(filter_bytes);
    }
#endif

    /* Scratch is used only by the NHWC ch_mult=1, dilation=1 fast NT_T kernel. */
    if (layout != ARM_NN_LAYOUT_NHWC || dw_conv_params->ch_mult != 1 || dw_conv_params->dilation.w != 1 ||
        dw_conv_params->dilation.h != 1)
    {
        return 0;
    }

    if (input_dims->c <= 0 || filter_dims->w <= 0 || filter_dims->h <= 0)
    {
        return 0;
    }

    const size_t kernel_size = (size_t)filter_dims->w * (size_t)filter_dims->h;
    const size_t channels = (size_t)input_dims->c;
    size_t lhs_bytes = (size_t)ARM_NN_DW_NT_T_F16_TILE_ROWS;

    if (!arm_nn_checked_size_mul(lhs_bytes, kernel_size, &lhs_bytes) ||
        !arm_nn_checked_size_mul(lhs_bytes, channels, &lhs_bytes) ||
        !arm_nn_checked_size_mul(lhs_bytes, sizeof(float16_t), &lhs_bytes))
    {
        return 0;
    }

    return arm_nn_size_to_i32_or_zero(lhs_bytes);
}

int32_t arm_depthwise_conv_wrapper_f16_get_buffer_size(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                       const cmsis_nn_dims *input_dims,
                                                       const cmsis_nn_dims *filter_dims,
                                                       const cmsis_nn_dims *output_dims)
{
    return arm_depthwise_conv_f16_get_buffer_size(
        dw_conv_params, input_dims, filter_dims, output_dims, ARM_NN_LAYOUT_NHWC);
}

int32_t arm_convolve_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                         const cmsis_nn_dims *input_dims,
                                         const cmsis_nn_dims *filter_dims,
                                         const cmsis_nn_dims *output_dims,
                                         arm_nn_tensor_layout layout)
{
    if (!conv_params || !input_dims || !filter_dims || !output_dims)
    {
        return 0;
    }

    if (input_dims->c <= 0 || filter_dims->h <= 0 || filter_dims->w <= 0 || output_dims->c <= 0)
    {
        return 0;
    }

    if (layout != ARM_NN_LAYOUT_NHWC)
    {
        return 0;
    }

    if (filter_dims->h == 1 && filter_dims->w == 1 && conv_params->padding.h == 0 && conv_params->padding.w == 0)
    {
        return arm_convolve_1x1_f16_get_buffer_size(conv_params, input_dims, filter_dims, output_dims, layout);
    }

    if (input_dims->h == 1 && output_dims->h == 1 && filter_dims->h == 1 && filter_dims->w > 1 &&
        conv_params->stride.h == 1 && conv_params->stride.w > 0 && conv_params->padding.h == 0 &&
        conv_params->dilation.h == 1 && conv_params->dilation.w == 1 &&
        !((input_dims->n == 1 && filter_dims->w == 3 && conv_params->stride.w == 1 && conv_params->padding.w == 0) ||
          (input_dims->n == 1 && filter_dims->w == 5 && conv_params->stride.w == 1 && conv_params->padding.w == 0)))
    {
        return arm_convolve_1_x_n_f16_get_buffer_size(conv_params, input_dims, filter_dims, output_dims, layout);
    }

    size_t tile_bytes = (size_t)ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MAX_TILE_ROWS;

    if (!arm_nn_checked_size_mul(tile_bytes, (size_t)filter_dims->h, &tile_bytes) ||
        !arm_nn_checked_size_mul(tile_bytes, (size_t)filter_dims->w, &tile_bytes) ||
        !arm_nn_checked_size_mul(tile_bytes, (size_t)input_dims->c, &tile_bytes) ||
        !arm_nn_checked_size_mul(tile_bytes, sizeof(float16_t), &tile_bytes))
    {
        return 0;
    }

    return arm_nn_size_to_i32_or_zero(tile_bytes);
}

int32_t arm_convolve_wrapper_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const cmsis_nn_dims *output_dims)
{
    return arm_convolve_f16_get_buffer_size(conv_params, input_dims, filter_dims, output_dims, ARM_NN_LAYOUT_NHWC);
}

int32_t arm_convolve_1x1_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                             const cmsis_nn_dims *input_dims,
                                             const cmsis_nn_dims *filter_dims,
                                             const cmsis_nn_dims *output_dims,
                                             arm_nn_tensor_layout layout)
{
    if (!conv_params || !input_dims || !filter_dims || !output_dims)
    {
        return 0;
    }

    if (layout != ARM_NN_LAYOUT_NHWC || filter_dims->h != 1 || filter_dims->w != 1 || conv_params->padding.h != 0 ||
        conv_params->padding.w != 0 || input_dims->c <= 0)
    {
        return 0;
    }

    if (conv_params->stride.h == 1 && conv_params->stride.w == 1 && output_dims->h == input_dims->h &&
        output_dims->w == input_dims->w)
    {
        return 0;
    }

    size_t tile_bytes = (size_t)ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MAX_TILE_ROWS;
    if (!arm_nn_checked_size_mul(tile_bytes, (size_t)input_dims->c, &tile_bytes) ||
        !arm_nn_checked_size_mul(tile_bytes, sizeof(float16_t), &tile_bytes))
    {
        return 0;
    }

    return arm_nn_size_to_i32_or_zero(tile_bytes);
}

int32_t arm_convolve_1_x_n_f16_get_buffer_size(const cmsis_nn_conv_params_f16 *conv_params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_tensor_layout layout)
{
    if (!conv_params || !input_dims || !filter_dims || !output_dims)
    {
        return 0;
    }

    if (layout != ARM_NN_LAYOUT_NHWC || input_dims->h != 1 || output_dims->h != 1 || filter_dims->h != 1 ||
        filter_dims->w <= 1 || conv_params->stride.h != 1 || conv_params->stride.w <= 0 ||
        conv_params->padding.h != 0 || conv_params->dilation.h != 1 || conv_params->dilation.w != 1 ||
        input_dims->c <= 0 || input_dims->c != filter_dims->c)
    {
        return 0;
    }

    size_t row_bytes = (size_t)filter_dims->w;
    if (!arm_nn_checked_size_mul(row_bytes, (size_t)input_dims->c, &row_bytes) ||
        !arm_nn_checked_size_mul(row_bytes, sizeof(float16_t), &row_bytes))
    {
        return 0;
    }

    return arm_nn_size_to_i32_or_zero(row_bytes);
}

int32_t arm_fully_connected_f16_get_buffer_size(const cmsis_nn_fc_params_f16 *fc_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims,
                                                arm_nn_tensor_layout layout)
{
    /* Current float16 fully-connected implementations are bufferless. */
    (void)fc_params;
    (void)input_dims;
    (void)filter_dims;
    (void)output_dims;
    (void)layout;
    return 0;
}

int32_t arm_batch_matmul_f16_get_buffer_size(const cmsis_nn_bmm_params_f16 *bmm_params,
                                             const cmsis_nn_dims *input_lhs_dims,
                                             const cmsis_nn_dims *input_rhs_dims,
                                             const cmsis_nn_dims *output_dims)
{
    /* Current float16 batch-matmul implementations are bufferless. */
    (void)bmm_params;
    (void)input_lhs_dims;
    (void)input_rhs_dims;
    (void)output_dims;
    return 0;
}
/**
 * @} end of GetBufferSizeNNConv group
 */
