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
 * Title:        arm_nn_mat_mult_nt_n_packed_f16.c
 * Description:  Support: matrix multiply (lhs non-transposed, rhs packed non-transposed)
 *
 * $Date:        16 April 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Private
 */

/**
 * @addtogroup groupSupport
 * @{
 */

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_NN_MAT_MULT_NT_N_PACKED_F16_BLOCK_COLS (8)
        #define ARM_NN_MAT_MULT_NT_N_PACKED_F16_BLOCK_ROWS (4)
    #endif

/* Refer header file for details. */
arm_cmsis_nn_status arm_nn_mat_mult_nt_n_packed_f16(const float16_t *__RESTRICT lhs,
                                                    const float16_t *__RESTRICT rhs_packed,
                                                    const float16_t *__RESTRICT bias,
                                                    float16_t *__RESTRICT dst,
                                                    int32_t lhs_rows,
                                                    int32_t rhs_rows,
                                                    int32_t rhs_cols,
                                                    int32_t row_address_offset,
                                                    float16_t activation_min,
                                                    float16_t activation_max)
{
    if (!lhs || !rhs_packed || !dst || lhs_rows <= 0 || rhs_rows <= 0 || rhs_cols <= 0 || row_address_offset <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t block_cols = 8;
    const int32_t rhs_blocks = (rhs_rows + block_cols - 1) / block_cols;
    int32_t r = 0;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t vmin = vdupq_n_f16(activation_min);
    const float16x8_t vmax = vdupq_n_f16(activation_max);

    for (; r + ARM_NN_MAT_MULT_NT_N_PACKED_F16_BLOCK_ROWS <= lhs_rows; r += ARM_NN_MAT_MULT_NT_N_PACKED_F16_BLOCK_ROWS)
    {
        const float16_t *lhs_row0 = lhs + (size_t)(r + 0) * rhs_cols;
        const float16_t *lhs_row1 = lhs + (size_t)(r + 1) * rhs_cols;
        const float16_t *lhs_row2 = lhs + (size_t)(r + 2) * rhs_cols;
        const float16_t *lhs_row3 = lhs + (size_t)(r + 3) * rhs_cols;
        float16_t *dst_row0 = dst + (size_t)(r + 0) * row_address_offset;
        float16_t *dst_row1 = dst + (size_t)(r + 1) * row_address_offset;
        float16_t *dst_row2 = dst + (size_t)(r + 2) * row_address_offset;
        float16_t *dst_row3 = dst + (size_t)(r + 3) * row_address_offset;

        for (int32_t b = 0; b < rhs_blocks; ++b)
        {
            const int32_t c = b * block_cols;
            const int32_t remaining = rhs_rows - c;
            const int32_t valid_cols = remaining < block_cols ? remaining : block_cols;
            const float16_t *rhs_block = rhs_packed + (size_t)b * rhs_cols * block_cols;
            const mve_pred16_t p = vctp16q((uint32_t)valid_cols);

            float16x8_t vacc0 = bias ? vld1q_z(bias + c, p) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = bias ? vld1q_z(bias + c, p) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc2 = bias ? vld1q_z(bias + c, p) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc3 = bias ? vld1q_z(bias + c, p) : vdupq_n_f16((float16_t)0.0f);

            for (int32_t k = 0; k < rhs_cols; ++k)
            {
                float16x8_t vrhs;
                if (valid_cols == block_cols)
                {
                    vrhs = vld1q(rhs_block + (size_t)k * block_cols);
                }
                else
                {
                    vrhs = vld1q_z(rhs_block + (size_t)k * block_cols, p);
                }
                vacc0 = vfmaq(vacc0, vrhs, lhs_row0[k]);
                vacc1 = vfmaq(vacc1, vrhs, lhs_row1[k]);
                vacc2 = vfmaq(vacc2, vrhs, lhs_row2[k]);
                vacc3 = vfmaq(vacc3, vrhs, lhs_row3[k]);
            }

            vacc0 = arm_nn_clamp_mve_f16(vacc0, vmin, vmax);
            vacc1 = arm_nn_clamp_mve_f16(vacc1, vmin, vmax);
            vacc2 = arm_nn_clamp_mve_f16(vacc2, vmin, vmax);
            vacc3 = arm_nn_clamp_mve_f16(vacc3, vmin, vmax);

            if (valid_cols == block_cols)
            {
                vst1q(dst_row0 + c, vacc0);
                vst1q(dst_row1 + c, vacc1);
                vst1q(dst_row2 + c, vacc2);
                vst1q(dst_row3 + c, vacc3);
            }
            else
            {
                vst1q_p(dst_row0 + c, vacc0, p);
                vst1q_p(dst_row1 + c, vacc1, p);
                vst1q_p(dst_row2 + c, vacc2, p);
                vst1q_p(dst_row3 + c, vacc3, p);
            }
        }
    }
    #endif

    for (; r < lhs_rows; ++r)
    {
        const float16_t *lhs_row = lhs + (size_t)r * rhs_cols;
        float16_t *dst_row = dst + (size_t)r * row_address_offset;

        for (int32_t b = 0; b < rhs_blocks; ++b)
        {
            const int32_t c = b * block_cols;
            const int32_t remaining = rhs_rows - c;
            const int32_t valid_cols = remaining < block_cols ? remaining : block_cols;
            const float16_t *rhs_block = rhs_packed + (size_t)b * rhs_cols * block_cols;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
            const mve_pred16_t p = vctp16q((uint32_t)valid_cols);
            float16x8_t vacc = bias ? vld1q_z(bias + c, p) : vdupq_n_f16((float16_t)0.0f);

            for (int32_t k = 0; k < rhs_cols; ++k)
            {
                const float16x8_t vrhs = (valid_cols == block_cols) ? vld1q(rhs_block + (size_t)k * block_cols)
                                                                    : vld1q_z(rhs_block + (size_t)k * block_cols, p);
                vacc = vfmaq(vacc, vrhs, lhs_row[k]);
            }

            vacc = arm_nn_clamp_mve_f16(vacc, vmin, vmax);
            if (valid_cols == block_cols)
            {
                vst1q(dst_row + c, vacc);
            }
            else
            {
                vst1q_p(dst_row + c, vacc, p);
            }
    #else
            for (int32_t lane = 0; lane < valid_cols; ++lane)
            {
                _Float16 acc = bias ? (_Float16)bias[c + lane] : (_Float16)0.0f;
                for (int32_t k = 0; k < rhs_cols; ++k)
                {
                    acc += (_Float16)lhs_row[k] * (_Float16)rhs_block[(size_t)k * block_cols + lane];
                }
                dst_row[c + lane] = arm_nn_clamp_scalar_f16((float16_t)acc, activation_min, activation_max);
            }
    #endif
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of groupSupport group
 */

#endif /* ARM_NN_ENABLE_F16 */
