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
 * Title:        arm_convolve_1_x_n_f16.c
 * Description:  Dedicated float16 1xN convolution
 *
 * $Date:        31 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

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

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    #define ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS (8)
    #define ARM_NN_CONV_1XN_F16_MVE_DUAL_BLOCK_ROWS (16)
    #define ARM_NN_CONV_1XN_F16_MVE_SUB_BLOCK_ROWS (4)
    #define ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS ((int32_t)ARM_NN_MVE_F16_MAX_GATHER_STRIDE_8)
    #define ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS_DUAL ((int32_t)ARM_NN_MVE_F16_MAX_GATHER_STRIDE_16)
    #define ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS_SUB ((int32_t)ARM_NN_MVE_F16_MAX_GATHER_STRIDE_4)
#endif

__STATIC_INLINE float16_t arm_convolve_1_x_n_dot_f16(const float16_t *lhs, const float16_t *rhs, int32_t len)
{
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);
    for (int32_t i = 0; i < len; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(len - i));
        vacc = vfmaq_m(vacc, vld1q_z(lhs + i, p), vld1q_z(rhs + i, p), p);
    }
    return arm_nn_vec_reduce_add_f16(vacc);
#else
    _Float16 acc = (_Float16)0.0f;
    for (int32_t i = 0; i < len; ++i)
    {
        acc += (_Float16)lhs[i] * (_Float16)rhs[i];
    }
    return (float16_t)acc;
#endif
}

__STATIC_INLINE arm_cmsis_nn_status arm_convolve_1_x_n_mat_mult_nt_t_strided_f16(const float16_t *__RESTRICT lhs,
                                                                                 const float16_t *__RESTRICT rhs,
                                                                                 const float16_t *__RESTRICT bias,
                                                                                 float16_t *__RESTRICT dst,
                                                                                 int32_t lhs_rows,
                                                                                 int32_t rhs_rows,
                                                                                 int32_t rhs_cols,
                                                                                 int32_t lhs_cols_offset,
                                                                                 int32_t row_address_offset,
                                                                                 float16_t activation_min,
                                                                                 float16_t activation_max)
{
    if (!lhs || !rhs || !dst || lhs_rows <= 0 || rhs_rows <= 0 || rhs_cols <= 0 || lhs_cols_offset <= 0 ||
        row_address_offset <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (int32_t r = 0; r < lhs_rows; ++r)
    {
        const float16_t *lhs_row = lhs + (size_t)r * lhs_cols_offset;
        float16_t *dst_row = dst + (size_t)r * row_address_offset;

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        int32_t c = 0;
        if (rhs_cols <= ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS)
        {
            const uint16_t rhs_cols_u16 = (uint16_t)rhs_cols;
            const uint16x8_t offsets = vmulq(vidupq_u16((uint32_t)0, 1), rhs_cols_u16);
            const float16x8_t vmin = vdupq_n_f16(activation_min);
            const float16x8_t vmax = vdupq_n_f16(activation_max);

            if (rhs_cols <= ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS_DUAL)
            {
                const uint16x8_t offsets_hi =
                    vaddq(offsets, (uint16_t)(ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS * rhs_cols_u16));

                for (; c + ARM_NN_CONV_1XN_F16_MVE_DUAL_BLOCK_ROWS <= rhs_rows;
                     c += ARM_NN_CONV_1XN_F16_MVE_DUAL_BLOCK_ROWS)
                {
                    const float16_t *rhs_block = rhs + (size_t)c * rhs_cols;
                    float16x8_t vacc_lo = bias ? vld1q(bias + c) : vdupq_n_f16((float16_t)0.0f);
                    float16x8_t vacc_hi =
                        bias ? vld1q(bias + c + ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS) : vdupq_n_f16((float16_t)0.0f);

                    for (int32_t k = 0; k < rhs_cols; ++k)
                    {
                        const float16_t lhs_v = lhs_row[k];
                        const float16x8_t vrhs_lo = vldrhq_gather_shifted_offset(rhs_block + k, offsets);
                        const float16x8_t vrhs_hi = vldrhq_gather_shifted_offset(rhs_block + k, offsets_hi);
                        vacc_lo = vfmaq(vacc_lo, vrhs_lo, lhs_v);
                        vacc_hi = vfmaq(vacc_hi, vrhs_hi, lhs_v);
                    }

                    vacc_lo = arm_nn_clamp_mve_f16(vacc_lo, vmin, vmax);
                    vacc_hi = arm_nn_clamp_mve_f16(vacc_hi, vmin, vmax);
                    vst1q(dst_row + c, vacc_lo);
                    vst1q(dst_row + c + ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS, vacc_hi);
                }
            }

            for (; c + ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS <= rhs_rows; c += ARM_NN_CONV_1XN_F16_MVE_BLOCK_ROWS)
            {
                const float16_t *rhs_block = rhs + (size_t)c * rhs_cols;
                float16x8_t vacc = bias ? vld1q(bias + c) : vdupq_n_f16((float16_t)0.0f);

                for (int32_t k = 0; k < rhs_cols; ++k)
                {
                    const float16x8_t vrhs = vldrhq_gather_shifted_offset(rhs_block + k, offsets);
                    vacc = vfmaq(vacc, vrhs, lhs_row[k]);
                }

                vacc = arm_nn_clamp_mve_f16(vacc, vmin, vmax);
                vst1q(dst_row + c, vacc);
            }

            if (rhs_cols <= ARM_NN_CONV_1XN_F16_MVE_MAX_RHS_COLS_SUB)
            {
                const mve_pred16_t p = vctp16q((uint32_t)ARM_NN_CONV_1XN_F16_MVE_SUB_BLOCK_ROWS);

                for (; c + ARM_NN_CONV_1XN_F16_MVE_SUB_BLOCK_ROWS <= rhs_rows;
                     c += ARM_NN_CONV_1XN_F16_MVE_SUB_BLOCK_ROWS)
                {
                    const float16_t *rhs_block = rhs + (size_t)c * rhs_cols;
                    float16x8_t vacc = bias ? vld1q_z(bias + c, p) : vdupq_n_f16((float16_t)0.0f);

                    for (int32_t k = 0; k < rhs_cols; ++k)
                    {
                        const float16x8_t vrhs = vldrhq_gather_shifted_offset_z(rhs_block + k, offsets, p);
                        vacc = vfmaq(vacc, vrhs, lhs_row[k]);
                    }

                    vacc = arm_nn_clamp_mve_f16(vacc, vmin, vmax);
                    vst1q_p(dst_row + c, vacc, p);
                }
            }
        }
#else
        int32_t c = 0;
#endif

        for (; c < rhs_rows; ++c)
        {
            const float16_t *rhs_row = rhs + (size_t)c * rhs_cols;
            _Float16 acc = bias ? (_Float16)bias[c] : (_Float16)0.0f;
            acc += (_Float16)arm_convolve_1_x_n_dot_f16(lhs_row, rhs_row, rhs_cols);
            dst_row[c] = (float16_t)(_Float16)CLAMP(acc, (_Float16)activation_max, (_Float16)activation_min);
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

__STATIC_INLINE void arm_convolve_1_x_n_find_regions(const cmsis_nn_conv_params_f16 *conv_params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims,
                                                     int32_t *left_pad_num,
                                                     int32_t *no_pad_num,
                                                     int32_t *right_pad_num)
{
    int32_t first_valid = 0;
    while (first_valid < output_dims->w)
    {
        const int32_t base_x = first_valid * conv_params->stride.w - conv_params->padding.w;
        if (base_x >= 0 && (base_x + filter_dims->w) <= input_dims->w)
        {
            break;
        }
        ++first_valid;
    }

    int32_t last_valid = output_dims->w - 1;
    while (last_valid >= first_valid)
    {
        const int32_t base_x = last_valid * conv_params->stride.w - conv_params->padding.w;
        if (base_x >= 0 && (base_x + filter_dims->w) <= input_dims->w)
        {
            break;
        }
        --last_valid;
    }

    *left_pad_num = first_valid;
    *no_pad_num = MAX(last_valid - first_valid + 1, 0);
    *right_pad_num = output_dims->w - *left_pad_num - *no_pad_num;
}

__STATIC_INLINE void arm_convolve_1_x_n_pack_rows_f16(float16_t *scratch,
                                                      const float16_t *input_b,
                                                      const cmsis_nn_conv_params_f16 *conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      int32_t start_out_x,
                                                      int32_t rows)
{
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;
    const int32_t kernel_w = filter_dims->w;
    const int32_t rhs_cols = kernel_w * input_c;

    for (int32_t r = 0; r < rows; ++r)
    {
        const int32_t out_x = start_out_x + r;
        const int32_t base_x = out_x * conv_params->stride.w - conv_params->padding.w;
        const int32_t left_pad_cols = MAX(0, -base_x);
        const int32_t valid_x0 = MAX(base_x, 0);
        const int32_t valid_x1 = MIN(base_x + kernel_w, input_w);
        const int32_t valid_cols = MAX(valid_x1 - valid_x0, 0);
        const int32_t right_pad_cols = kernel_w - left_pad_cols - valid_cols;
        float16_t *patch_row = scratch + (size_t)r * rhs_cols;

        if (left_pad_cols > 0)
        {
            arm_memset_f16(patch_row, (float16_t)0.0f, (uint32_t)((size_t)left_pad_cols * input_c));
        }
        if (valid_cols > 0)
        {
            arm_memcpy_f16(patch_row + (size_t)left_pad_cols * input_c,
                           input_b + (size_t)valid_x0 * input_c,
                           (uint32_t)((size_t)valid_cols * input_c));
        }
        if (right_pad_cols > 0)
        {
            arm_memset_f16(patch_row + (size_t)(left_pad_cols + valid_cols) * input_c,
                           (float16_t)0.0f,
                           (uint32_t)((size_t)right_pad_cols * input_c));
        }
    }
}

arm_cmsis_nn_status arm_convolve_1_x_n_nhwc_f16(const cmsis_nn_context *ctx,
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

    if (!ctx || !ctx->buf || !conv_params || !input_dims || !input_data || !filter_dims || !filter_data ||
        !output_dims || !output_data)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (input_dims->h != 1 || output_dims->h != 1 || filter_dims->h != 1 || filter_dims->w <= 1 ||
        conv_params->stride.h != 1 || conv_params->stride.w <= 0 || conv_params->padding.h != 0 ||
        conv_params->dilation.h != 1 || conv_params->dilation.w != 1 || input_dims->c != filter_dims->c)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t buf_size =
        arm_convolve_1_x_n_f16_get_buffer_size(conv_params, input_dims, filter_dims, output_dims, ARM_NN_LAYOUT_NHWC);
    if (buf_size <= 0 || ctx->size < buf_size)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = input_dims->n;
    const int32_t input_w = input_dims->w;
    const int32_t input_c = input_dims->c;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t kernel_w = filter_dims->w;
    const int32_t rhs_cols = kernel_w * input_c;
    const int32_t lhs_cols_offset = input_c * conv_params->stride.w;
    const int32_t tile_rows = (int32_t)((size_t)ctx->size / ((size_t)rhs_cols * sizeof(float16_t)));
    float16_t *scratch = (float16_t *)ctx->buf;
    int32_t left_pad_num = 0;
    int32_t no_pad_num = 0;
    int32_t right_pad_num = 0;

    if (tile_rows <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    arm_convolve_1_x_n_find_regions(
        conv_params, input_dims, filter_dims, output_dims, &left_pad_num, &no_pad_num, &right_pad_num);

    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_w * output_c;

        for (int32_t row = 0; row < left_pad_num; row += tile_rows)
        {
            const int32_t rows = MIN(tile_rows, left_pad_num - row);
            arm_convolve_1_x_n_pack_rows_f16(scratch, input_b, conv_params, input_dims, filter_dims, row, rows);

            arm_cmsis_nn_status st = arm_nn_mat_mult_nt_t_f16(scratch,
                                                              filter_data,
                                                              bias_data,
                                                              output_b,
                                                              rows,
                                                              output_c,
                                                              rhs_cols,
                                                              output_c,
                                                              conv_params->activation.min,
                                                              conv_params->activation.max);
            if (st != ARM_CMSIS_NN_SUCCESS)
            {
                return st;
            }
            output_b += (size_t)rows * output_c;
        }

        if (no_pad_num > 0)
        {
            const int32_t input_start = (conv_params->stride.w * left_pad_num - conv_params->padding.w) * input_c;
            arm_cmsis_nn_status st = arm_convolve_1_x_n_mat_mult_nt_t_strided_f16(input_b + input_start,
                                                                                  filter_data,
                                                                                  bias_data,
                                                                                  output_b,
                                                                                  no_pad_num,
                                                                                  output_c,
                                                                                  rhs_cols,
                                                                                  lhs_cols_offset,
                                                                                  output_c,
                                                                                  conv_params->activation.min,
                                                                                  conv_params->activation.max);
            if (st != ARM_CMSIS_NN_SUCCESS)
            {
                return st;
            }
            output_b += (size_t)no_pad_num * output_c;
        }

        for (int32_t row = 0; row < right_pad_num; row += tile_rows)
        {
            const int32_t rows = MIN(tile_rows, right_pad_num - row);
            const int32_t start_out_x = left_pad_num + no_pad_num + row;
            arm_convolve_1_x_n_pack_rows_f16(scratch, input_b, conv_params, input_dims, filter_dims, start_out_x, rows);

            arm_cmsis_nn_status st = arm_nn_mat_mult_nt_t_f16(scratch,
                                                              filter_data,
                                                              bias_data,
                                                              output_b,
                                                              rows,
                                                              output_c,
                                                              rhs_cols,
                                                              output_c,
                                                              conv_params->activation.min,
                                                              conv_params->activation.max);
            if (st != ARM_CMSIS_NN_SUCCESS)
            {
                return st;
            }
            output_b += (size_t)rows * output_c;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_convolve_1_x_n_f16(const cmsis_nn_context *ctx,
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

    return arm_convolve_1_x_n_nhwc_f16(ctx,
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

/** @} end of NNConv group */
