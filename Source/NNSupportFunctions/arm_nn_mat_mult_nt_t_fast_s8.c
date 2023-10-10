/*
 * SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_mat_mult_s8_nt_t_fast_s8
 * Description:  Matrix multiplication support function with the right-hand-side (rhs) matrix transposed
 *
 * $Date:        09 October 2023
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

/*
 * s8 matrix multiplication with the right-hand-side matrix transposed
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_fast_s8(const int8_t *lhs,
                                                 const int8_t *rhs,
                                                 const int32_t *ker_sum,
                                                 int8_t *dst,
                                                 const int32_t *dst_multipliers,
                                                 const int32_t *dst_shifts,
                                                 const int32_t lhs_rows,
                                                 const int32_t rhs_rows,
                                                 const int32_t rhs_cols,
                                                 const int32_t lhs_offset,
                                                 const int32_t dst_offset,
                                                 const int32_t activation_min,
                                                 const int32_t activation_max,
                                                 const int32_t lhs_cols_offset)
{

#if defined(ARM_MATH_MVEI)
    int i_items = 0;

    (void)lhs_offset;

    for (; i_items <= (lhs_rows - 3); i_items += 3)
    {
        for (int i = 0; i <= rhs_rows - 2; i += 2)
        {
            int32_t acc_ch_1_n1 = 0;
            int32_t acc_ch_1_n2 = 0;
            int32_t acc_ch_1_n3 = 0;

            int32_t acc_ch_2_n1 = 0;
            int32_t acc_ch_2_n2 = 0;
            int32_t acc_ch_2_n3 = 0;

            const int8_t *col_base_1 = rhs + i * rhs_cols;
            const int8_t *col_base_2 = col_base_1 + rhs_cols;

            const int8_t *lhs_vec_1 = lhs;
            const int8_t *lhs_vec_2 = lhs + lhs_cols_offset;
            const int8_t *lhs_vec_3 = lhs + (2 * lhs_cols_offset);

            int32_t sum_k1 = ker_sum[i];
            int32_t sum_k2 = ker_sum[i + 1];

            // Note: If operand initialization is moved around, use '&' constraint to
            // specify earlyclobber operands.
            __ASM volatile(" .p2align 2                             \n"
                           "   wlstp.8         lr, %[cnt], 1f       \n"
                           "   mov             %[out0], 0           \n"
                           "   mov             %[out1], 0           \n"
                           "   mov             %[out2], 0           \n"
                           "   mov             %[out3], 0           \n"
                           "   mov             %[out4], 0           \n"
                           "   mov             %[out5], 0           \n"
                           "   vldrb.8         q2, [%[row0]], #16   \n"
                           "   vldrb.8         q0, [%[col0]], #16   \n"
                           "2:                                      \n"
                           "   vmladava.s8    %[out0],  q0, q2      \n"
                           "   vldrb.8         q3, [%[row1]], #16   \n"
                           "   vmladava.s8     %[out1], q0, q3      \n"
                           "   vldrb.8         q4, [%[row2]], #16   \n"
                           "   vmladava.s8     %[out2], q0, q4      \n"
                           "   vldrb.8         q1, [%[col1]], #16   \n"
                           "   vmladava.s8     %[out3], q1, q2      \n"
                           "   vldrb.8         q0, [%[col0]], #16   \n"
                           "   vmladava.s8     %[out4], q1, q3      \n"
                           "   vldrb.8         q2, [%[row0]], #16   \n"
                           "   vmladava.s8     %[out5], q1, q4      \n"
                           "   letp            lr, 2b               \n"
                           "1:                                      \n"
                           : [col0] "+r"(col_base_1),
                             [col1] "+r"(col_base_2),
                             [row0] "+r"(lhs_vec_1),
                             [row1] "+r"(lhs_vec_2),
                             [row2] "+r"(lhs_vec_3),
                             [out0] "=&Te"(acc_ch_1_n1),
                             [out1] "=&Te"(acc_ch_1_n2),
                             [out2] "=&Te"(acc_ch_1_n3),
                             [out3] "=&Te"(acc_ch_2_n1),
                             [out4] "=&Te"(acc_ch_2_n2),
                             [out5] "=&Te"(acc_ch_2_n3)
                           : [cnt] "r"(rhs_cols)
                           : "q0", "q1", "q2", "q3", "q4", "memory", "r14");

            int32x4_t res_1 = {acc_ch_1_n1, acc_ch_1_n2, acc_ch_1_n3, 0};
            res_1 = vaddq_n_s32(res_1, sum_k1);
            int32x4_t res_2 = {acc_ch_2_n1, acc_ch_2_n2, acc_ch_2_n3, 0};
            res_2 = vaddq_n_s32(res_2, sum_k2);

            res_1 = arm_requantize_mve(res_1, dst_multipliers[i], dst_shifts[i]);
            res_1 = vaddq_n_s32(res_1, dst_offset);
            res_1 = vmaxq_s32(res_1, vdupq_n_s32(activation_min));
            res_1 = vminq_s32(res_1, vdupq_n_s32(activation_max));

            const uint32x4_t scatter_offset = {0, (uint32_t)rhs_rows, (uint32_t)rhs_rows * 2, 0};
            const mve_pred16_t p = vctp32q(3);
            vstrbq_scatter_offset_p_s32(dst, scatter_offset, res_1, p);
            dst++;

            res_2 = arm_requantize_mve(res_2, dst_multipliers[i + 1], dst_shifts[i + 1]);
            res_2 = vaddq_n_s32(res_2, dst_offset);
            res_2 = vmaxq_s32(res_2, vdupq_n_s32(activation_min));
            res_2 = vminq_s32(res_2, vdupq_n_s32(activation_max));
            vstrbq_scatter_offset_p_s32(dst, scatter_offset, res_2, p);
            dst++;
        }
        lhs += 3 * lhs_cols_offset;
        dst += (2 * rhs_rows);
    }
#else
    (void)lhs;
    (void)rhs;
    (void)ker_sum;
    (void)dst;
    (void)dst_multipliers;
    (void)dst_shifts;
    (void)lhs_rows;
    (void)rhs_rows;
    (void)rhs_cols;
    (void)lhs_offset;
    (void)dst_offset;
    (void)activation_min;
    (void)activation_max;
    (void)lhs_cols_offset;

    return ARM_CMSIS_NN_FAILURE;

#endif
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
