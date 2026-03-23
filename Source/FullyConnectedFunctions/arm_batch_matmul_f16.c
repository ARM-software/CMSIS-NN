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
 * Title:        arm_batch_matmul_f16.c
 * Description:  Batch matrix multiplication (float16)
 *
 * $Date:        31 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_BATCH_MATMUL_F16_MVE_BLOCK (8)
        #define ARM_BATCH_MATMUL_F16_MVE_MAX_STRIDE ((int32_t)ARM_NN_MVE_F16_MAX_GATHER_STRIDE_8)
    #endif

/**
 * @ingroup Public
 */

/**
 * @addtogroup FC
 * @{
 */

__STATIC_INLINE float16_t dot_contig_strided_f16_scalar(const float16_t *__RESTRICT contig_row,
                                                        const float16_t *__RESTRICT strided_base,
                                                        int32_t len,
                                                        int32_t stride)
{
    _Float16 acc = (_Float16)0.0f;
    for (int32_t i = 0; i < len; ++i)
    {
        acc += (_Float16)contig_row[i] * (_Float16)strided_base[(size_t)i * stride];
    }
    return (float16_t)acc;
}

__STATIC_INLINE float16_t dot_strided_strided_f16_scalar(const float16_t *__RESTRICT lhs_base,
                                                         int32_t lhs_stride,
                                                         const float16_t *__RESTRICT rhs_base,
                                                         int32_t rhs_stride,
                                                         int32_t len)
{
    _Float16 acc = (_Float16)0.0f;
    for (int32_t i = 0; i < len; ++i)
    {
        acc += (_Float16)lhs_base[(size_t)i * lhs_stride] * (_Float16)rhs_base[(size_t)i * rhs_stride];
    }
    return (float16_t)acc;
}

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE float16_t dot_contig_strided_f16_mve(const float16_t *__RESTRICT contig_row,
                                                     const float16_t *__RESTRICT strided_base,
                                                     int32_t len,
                                                     int32_t stride,
                                                     uint16x8_t offsets)
{
    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);

    for (int32_t i = 0; i < len; i += ARM_BATCH_MATMUL_F16_MVE_BLOCK)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(len - i));
        const float16x8_t vcontig = vld1q_z(contig_row + i, p);
        const float16x8_t vstrided = vldrhq_gather_shifted_offset_z(strided_base + (size_t)i * stride, offsets, p);
        vacc = vfmaq(vacc, vcontig, vstrided);
    }

    return arm_nn_vec_reduce_add_f16(vacc);
}

__STATIC_INLINE float16_t dot_strided_strided_f16_mve(const float16_t *__RESTRICT lhs_base,
                                                      int32_t lhs_stride,
                                                      uint16x8_t lhs_offsets,
                                                      const float16_t *__RESTRICT rhs_base,
                                                      int32_t rhs_stride,
                                                      uint16x8_t rhs_offsets,
                                                      int32_t len)
{
    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);

    for (int32_t i = 0; i < len; i += ARM_BATCH_MATMUL_F16_MVE_BLOCK)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(len - i));
        const float16x8_t vlhs = vldrhq_gather_shifted_offset_z(lhs_base + (size_t)i * lhs_stride, lhs_offsets, p);
        const float16x8_t vrhs = vldrhq_gather_shifted_offset_z(rhs_base + (size_t)i * rhs_stride, rhs_offsets, p);
        vacc = vfmaq(vacc, vlhs, vrhs);
    }

    return arm_nn_vec_reduce_add_f16(vacc);
}
    #endif

/* Refer header file for details. */
arm_cmsis_nn_status arm_batch_matmul_f16(const cmsis_nn_context *ctx,
                                         const cmsis_nn_bmm_params_f16 *bmm_params,
                                         const cmsis_nn_dims *input_lhs_dims,
                                         const float16_t *input_lhs,
                                         const cmsis_nn_dims *input_rhs_dims,
                                         const float16_t *input_rhs,
                                         const cmsis_nn_dims *output_dims,
                                         float16_t *output)
{
    (void)ctx;
    if (!bmm_params || !input_lhs_dims || !input_rhs_dims || !output_dims || !input_lhs || !input_rhs || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t output_batch = output_dims->n;
    const int32_t output_height = output_dims->h;
    const int32_t lhs_rows = input_lhs_dims->c;
    const int32_t rhs_rows = input_rhs_dims->c;
    const int32_t rhs_cols = input_rhs_dims->w;
    const bool adj_x = bmm_params->adj_x;
    const bool adj_y = bmm_params->adj_y;

    const int32_t inner_lhs_diff = input_lhs_dims->h >= input_rhs_dims->h ? 0 : lhs_rows * rhs_cols;
    const int32_t inner_rhs_diff = input_rhs_dims->h >= input_lhs_dims->h ? rhs_rows * rhs_cols : 0;
    const int32_t outer_lhs_diff = input_lhs_dims->n >= input_rhs_dims->n
        ? inner_lhs_diff
        : -((lhs_rows * rhs_cols) - inner_lhs_diff) * input_lhs_dims->h;
    const int32_t outer_rhs_diff = input_rhs_dims->n >= input_lhs_dims->n ? (rhs_rows * rhs_cols) - inner_rhs_diff
                                                                          : -inner_rhs_diff * input_rhs_dims->h;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const bool lhs_stride_mve = lhs_rows <= ARM_BATCH_MATMUL_F16_MVE_MAX_STRIDE;
    const bool rhs_stride_mve = rhs_rows <= ARM_BATCH_MATMUL_F16_MVE_MAX_STRIDE;
    uint16x8_t lhs_offsets = vdupq_n_u16(0);
    uint16x8_t rhs_offsets = vdupq_n_u16(0);
    if (adj_x && lhs_stride_mve)
    {
        lhs_offsets = vmulq(vidupq_u16((uint32_t)0, 1), (uint16_t)lhs_rows);
    }
    if (adj_y && rhs_stride_mve)
    {
        rhs_offsets = vmulq(vidupq_u16((uint32_t)0, 1), (uint16_t)rhs_rows);
    }
    #endif

    for (int32_t i_out_batch = 0; i_out_batch < output_batch; i_out_batch++)
    {
        for (int32_t i_out_height = 0; i_out_height < output_height; i_out_height++)
        {
            const float16_t *lhs_mat = input_lhs;
            const float16_t *rhs_mat = input_rhs;
            if (!adj_x && !adj_y)
            {
                arm_cmsis_nn_status status = (bmm_params->rhs_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED)
                    ? arm_nn_mat_mult_nt_n_packed_f16(lhs_mat,
                                                      rhs_mat,
                                                      NULL,
                                                      output,
                                                      lhs_rows,
                                                      rhs_rows,
                                                      rhs_cols,
                                                      rhs_rows,
                                                      bmm_params->activation.min,
                                                      bmm_params->activation.max)
                    : arm_nn_mat_mult_nt_t_f16(lhs_mat,
                                               rhs_mat,
                                               NULL,
                                               output,
                                               lhs_rows,
                                               rhs_rows,
                                               rhs_cols,
                                               rhs_rows,
                                               bmm_params->activation.min,
                                               bmm_params->activation.max);
                if (status != ARM_CMSIS_NN_SUCCESS)
                {
                    return status;
                }
                output += lhs_rows * rhs_rows;
            }
            else if (bmm_params->rhs_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED)
            {
                return ARM_CMSIS_NN_ARG_ERROR;
            }
            else if (!adj_x && adj_y)
            {
                for (int32_t i_lhs_rows = 0; i_lhs_rows < lhs_rows; i_lhs_rows++)
                {
                    const float16_t *lhs_ptr = lhs_mat + (size_t)i_lhs_rows * rhs_cols;
                    for (int32_t i_rhs_rows = 0; i_rhs_rows < rhs_rows; i_rhs_rows++)
                    {
                        float16_t acc;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                        if (rhs_stride_mve)
                        {
                            acc = dot_contig_strided_f16_mve(
                                lhs_ptr, rhs_mat + i_rhs_rows, rhs_cols, rhs_rows, rhs_offsets);
                        }
                        else
    #endif
                        {
                            acc = dot_contig_strided_f16_scalar(lhs_ptr, rhs_mat + i_rhs_rows, rhs_cols, rhs_rows);
                        }
                        output[i_rhs_rows] = (float16_t)(_Float16)CLAMP(
                            (_Float16)acc, (_Float16)bmm_params->activation.max, (_Float16)bmm_params->activation.min);
                    }
                    output += rhs_rows;
                }
            }
            else if (adj_x && !adj_y)
            {
                for (int32_t i_lhs_rows = 0; i_lhs_rows < lhs_rows; i_lhs_rows++)
                {
                    for (int32_t i_rhs_rows = 0; i_rhs_rows < rhs_rows; i_rhs_rows++)
                    {
                        const float16_t *rhs_ptr = rhs_mat + (size_t)i_rhs_rows * rhs_cols;
                        float16_t acc;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                        if (lhs_stride_mve)
                        {
                            acc = dot_contig_strided_f16_mve(
                                rhs_ptr, lhs_mat + i_lhs_rows, rhs_cols, lhs_rows, lhs_offsets);
                        }
                        else
    #endif
                        {
                            acc = dot_contig_strided_f16_scalar(rhs_ptr, lhs_mat + i_lhs_rows, rhs_cols, lhs_rows);
                        }
                        output[i_rhs_rows] = (float16_t)(_Float16)CLAMP(
                            (_Float16)acc, (_Float16)bmm_params->activation.max, (_Float16)bmm_params->activation.min);
                    }
                    output += rhs_rows;
                }
            }
            else
            {
                for (int32_t i_lhs_rows = 0; i_lhs_rows < lhs_rows; i_lhs_rows++)
                {
                    for (int32_t i_rhs_rows = 0; i_rhs_rows < rhs_rows; i_rhs_rows++)
                    {
                        float16_t acc;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                        if (lhs_stride_mve && rhs_stride_mve)
                        {
                            acc = dot_strided_strided_f16_mve(lhs_mat + i_lhs_rows,
                                                              lhs_rows,
                                                              lhs_offsets,
                                                              rhs_mat + i_rhs_rows,
                                                              rhs_rows,
                                                              rhs_offsets,
                                                              rhs_cols);
                        }
                        else
    #endif
                        {
                            acc = dot_strided_strided_f16_scalar(
                                lhs_mat + i_lhs_rows, lhs_rows, rhs_mat + i_rhs_rows, rhs_rows, rhs_cols);
                        }
                        output[i_rhs_rows] = (float16_t)(_Float16)CLAMP(
                            (_Float16)acc, (_Float16)bmm_params->activation.max, (_Float16)bmm_params->activation.min);
                    }
                    output += rhs_rows;
                }
            }
            input_lhs += lhs_rows * rhs_cols;
            input_lhs -= inner_lhs_diff;
            input_rhs += inner_rhs_diff;
        }
        input_lhs += outer_lhs_diff;
        input_rhs += outer_rhs_diff;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of FC group
 */

#endif /* ARM_NN_ENABLE_F16 */
