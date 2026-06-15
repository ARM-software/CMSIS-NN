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
 * Title:        arm_nn_mat_mult_nt_t_f32.c
 * Description:  Support: matrix multiply (lhs non-transposed, rhs transposed)
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

/**
 * @ingroup Private
 */

/**
 * @addtogroup groupSupport
 * @{
 */

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_NN_MAT_MULT_NT_T_F32_MVE_BLOCK_ROWS (4)
        #define ARM_NN_MAT_MULT_NT_T_F32_MVE_DUAL_BLOCK_ROWS (8)
    #endif

    #if !defined(ARM_MATH_MVEF) || defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE float32_t dot_nt_t_f32_scalar(const float32_t *__RESTRICT lhs_row,
                                              const float32_t *__RESTRICT rhs_row,
                                              int32_t len)
{
    float32_t acc = 0.0f;
    for (int32_t i = 0; i < len; ++i)
    {
        acc += lhs_row[i] * rhs_row[i];
    }
    return acc;
}
    #endif

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE float32_t dot_nt_t_f32_mve(const float32_t *__RESTRICT lhs_row,
                                           const float32_t *__RESTRICT rhs_row,
                                           int32_t len)
{
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (int32_t i = 0; i < len; i += ARM_NN_MAT_MULT_NT_T_F32_MVE_BLOCK_ROWS)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(len - i));
        vacc = vfmaq_m(vacc, vld1q_z(lhs_row + i, p), vld1q_z(rhs_row + i, p), p);
    }

    return arm_nn_vec_reduce_add_f32(vacc);
}
    #endif

/* Refer header file for details. */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_f32(const float32_t *__RESTRICT lhs,
                                             const float32_t *__RESTRICT rhs,
                                             const float32_t *__RESTRICT bias,
                                             float32_t *__RESTRICT dst,
                                             int32_t lhs_rows,
                                             int32_t rhs_rows,
                                             int32_t rhs_cols,
                                             int32_t row_address_offset,
                                             float32_t activation_min,
                                             float32_t activation_max)
{
    if (!lhs || !rhs || !dst || lhs_rows <= 0 || rhs_rows <= 0 || rhs_cols <= 0 || row_address_offset <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const uint32_t rhs_cols_u32 = (uint32_t)rhs_cols;
    const uint32x4_t offsets = vmulq(vidupq_u32((uint32_t)0, 1), rhs_cols_u32);
    const uint32x4_t offsets_hi = vaddq(offsets, (uint32_t)(ARM_NN_MAT_MULT_NT_T_F32_MVE_BLOCK_ROWS * rhs_cols_u32));
    const float32x4_t vmin = vdupq_n_f32(activation_min);
    const float32x4_t vmax = vdupq_n_f32(activation_max);
    #endif

    for (int32_t r = 0; r < lhs_rows; ++r)
    {
        const float32_t *lhs_row = lhs + (size_t)r * rhs_cols;
        float32_t *dst_row = dst + (size_t)r * row_address_offset;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        int32_t c = 0;
        for (; c + ARM_NN_MAT_MULT_NT_T_F32_MVE_DUAL_BLOCK_ROWS <= rhs_rows;
             c += ARM_NN_MAT_MULT_NT_T_F32_MVE_DUAL_BLOCK_ROWS)
        {
            const float32_t *rhs_block = rhs + (size_t)c * rhs_cols;
            float32x4_t vacc_lo = bias ? vld1q(bias + c) : vdupq_n_f32(0.0f);
            float32x4_t vacc_hi = bias ? vld1q(bias + c + ARM_NN_MAT_MULT_NT_T_F32_MVE_BLOCK_ROWS) : vdupq_n_f32(0.0f);

            for (int32_t k = 0; k < rhs_cols; ++k)
            {
                const float32_t lhs_v = lhs_row[k];
                const float32x4_t vrhs_lo = vldrwq_gather_shifted_offset(rhs_block + k, offsets);
                const float32x4_t vrhs_hi = vldrwq_gather_shifted_offset(rhs_block + k, offsets_hi);
                vacc_lo = vfmaq(vacc_lo, vrhs_lo, lhs_v);
                vacc_hi = vfmaq(vacc_hi, vrhs_hi, lhs_v);
            }

            vacc_lo = arm_nn_clamp_mve_f32(vacc_lo, vmin, vmax);
            vacc_hi = arm_nn_clamp_mve_f32(vacc_hi, vmin, vmax);
            vst1q(dst_row + c, vacc_lo);
            vst1q(dst_row + c + ARM_NN_MAT_MULT_NT_T_F32_MVE_BLOCK_ROWS, vacc_hi);
        }

        for (; c + ARM_NN_MAT_MULT_NT_T_F32_MVE_BLOCK_ROWS <= rhs_rows; c += ARM_NN_MAT_MULT_NT_T_F32_MVE_BLOCK_ROWS)
        {
            const float32_t *rhs_block = rhs + (size_t)c * rhs_cols;
            float32x4_t vacc = bias ? vld1q(bias + c) : vdupq_n_f32(0.0f);

            for (int32_t k = 0; k < rhs_cols; ++k)
            {
                const float32x4_t vrhs = vldrwq_gather_shifted_offset(rhs_block + k, offsets);
                vacc = vfmaq(vacc, vrhs, lhs_row[k]);
            }

            vacc = arm_nn_clamp_mve_f32(vacc, vmin, vmax);
            vst1q(dst_row + c, vacc);
        }
    #else
        int32_t c = 0;
    #endif

        for (; c < rhs_rows; ++c)
        {
            const float32_t *rhs_row = rhs + (size_t)c * rhs_cols;
            float32_t acc = bias ? bias[c] : 0.0f;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            acc += dot_nt_t_f32_mve(lhs_row, rhs_row, rhs_cols);
    #else
            acc += dot_nt_t_f32_scalar(lhs_row, rhs_row, rhs_cols);
    #endif
            dst_row[c] = CLAMP(acc, activation_max, activation_min);
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of groupSupport group
 */

#endif /* ARM_NN_ENABLE_F32 */
