/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_mat_mult_nt_t_s8_packed.c
 * Description:  MVE matrix multiplication with packed s8 weights
 *
 * $Date:        20 October 2026
 * $Revision:    V.1.0.0
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

arm_cmsis_nn_status arm_nn_mat_mult_nt_t_s8_packed(const int8_t *lhs,
                                                   const int8_t *rhs,
                                                   const int32_t *effective_bias,
                                                   int8_t *dst,
                                                   const int32_t *dst_multipliers,
                                                   const int32_t *dst_shifts,
                                                   const int32_t lhs_rows,
                                                   const int32_t rhs_rows,
                                                   const int32_t rhs_cols,
                                                   const int32_t dst_offset,
                                                   const int32_t activation_min,
                                                   const int32_t activation_max,
                                                   const int32_t row_address_offset,
                                                   const int32_t lhs_cols_offset)
{
#if defined(ARM_MATH_MVEI)
    const int32_t block_bytes = ((rhs_cols + 15) & ~15) * 4;
    int32_t i_items = 0;
    for (; i_items <= (lhs_rows - 2); i_items += 2)
    {
        for (int32_t i = 0; i < rhs_rows; i += 4)
        {
            const int8_t *col_base = rhs + (i / 4) * block_bytes;
            const int8_t *ip_row_0 = lhs;
            const int8_t *ip_row_1 = lhs + lhs_cols_offset;
    #if defined(ARM_MATH_AUTOVECTORIZE)
            int32_t a00 = 0;
            int32_t a10 = 0;
            int32_t a01 = 0;
            int32_t a11 = 0;
            int32_t a02 = 0;
            int32_t a12 = 0;
            int32_t a03 = 0;
            int32_t a13 = 0;
            for (int32_t k = 0; k < rhs_cols; k++)
            {
                const int32_t offset = (k / 16) * 64 + (k & 15);
                a00 += ip_row_0[k] * col_base[offset + 0];
                a01 += ip_row_0[k] * col_base[offset + 16];
                a02 += ip_row_0[k] * col_base[offset + 32];
                a03 += ip_row_0[k] * col_base[offset + 48];
                a10 += ip_row_1[k] * col_base[offset + 0];
                a11 += ip_row_1[k] * col_base[offset + 16];
                a12 += ip_row_1[k] * col_base[offset + 32];
                a13 += ip_row_1[k] * col_base[offset + 48];
            }
    #else
            /* Eight even-numbered accumulator registers are required, including
             * LR, so use explicit loop control rather than wlstp/letp.
             *
             * Process 32 columns per iteration. Defer the two a03 products in
             * q7/q5 and q2/q0 to interleave them with the next iteration's loads.
             * Seed these pairs with zeros in the preamble and complete the pending
             * products after the final iteration.
             *
             * Restore the 0-31 remainder on exit, then handle an optional
             * 16-column block and a predicated tail. Guard all full input loads.
             * Preserve VPR around the tail using q6, which the main loop needs
             * for dot products.
             *
             * Early-clobber constraints prevent operand overlap; the low-register
             * counter constraint permits 16-bit loop-control instructions.
             */

            /* PERF: Using LR as an accumulator incurs a Cortex-M55 penalty
             * (measured 2-3 extra cycles). However, avoiding LR through register
             * shuffling was slower in model benchmarks.
             * Ref: Cortex-M55 Software Optimization Guide, Table 3-2 (p.18).
             * https://developer.arm.com/documentation/102692/latest/
             */
            register int32_t a00 __asm("r0") = 0;
            register int32_t a01 __asm("r2") = 0;
            register int32_t a02 __asm("r4") = 0;
            register int32_t a03 __asm("r6") = 0;
            register int32_t a10 __asm("r8") = 0;
            register int32_t a11 __asm("r10") = 0;
            register int32_t a12 __asm("r12") = 0;
            register int32_t a13 __asm("lr") = 0;
            int32_t count = rhs_cols;
            __ASM volatile("subs %[count], %[count], #32\n"
                           "blo 2f\n"
                           "vmov.i32 q7, #0\n"
                           "vmov.i32 q5, #0\n"
                           "vmov.i32 q2, #0\n"
                           "vmov.i32 q0, #0\n"
                           ".p2align 2\n"
                           "1:\n"
                           "vldrb.8 q4, [%[col_base]], #16\n"
                           "vmladava.s8 %[a03], q7, q5\n"
                           "vldrb.8 q1, [%[ip_row_1]], #16\n"
                           "vmladava.s8 %[a03], q2, q0\n"
                           "vldrb.8 q7, [%[ip_row_0]], #16\n"
                           "vmladava.s8 %[a10], q1, q4\n"
                           "vldrb.8 q6, [%[col_base]], #16\n"
                           "vmladava.s8 %[a00], q7, q4\n"
                           "vldrb.8 q2, [%[ip_row_0]], #16\n"
                           "vmladava.s8 %[a11], q1, q6\n"
                           "vldrb.8 q4, [%[col_base]], #16\n"
                           "vmladava.s8 %[a01], q7, q6\n"
                           "vldrb.8 q3, [%[ip_row_1]], #16\n"
                           "vmladava.s8 %[a12], q1, q4\n"
                           "vldrb.8 q5, [%[col_base]], #16\n"
                           "vmladava.s8 %[a02], q7, q4\n"
                           "vldrb.8 q6, [%[col_base]], #16\n"
                           "vmladava.s8 %[a13], q1, q5\n"
                           "vldrb.8 q4, [%[col_base]], #16\n"
                           "vmladava.s8 %[a10], q3, q6\n"
                           "vmladava.s8 %[a00], q2, q6\n"
                           "vldrb.8 q1, [%[col_base]], #16\n"
                           "vmladava.s8 %[a11], q3, q4\n"
                           "vmladava.s8 %[a01], q2, q4\n"
                           "vldrb.8 q0, [%[col_base]], #16\n"
                           "vmladava.s8 %[a12], q3, q1\n"
                           "vmladava.s8 %[a02], q2, q1\n"
                           "vmladava.s8 %[a13], q3, q0\n"
                           "subs %[count], %[count], #32\n"
                           "bhs 1b\n"
                           "vmladava.s8 %[a03], q7, q5\n"
                           "vmladava.s8 %[a03], q2, q0\n"
                           "2:\n"
                           "adds %[count], %[count], #32\n"
                           "cmp %[count], #16\n"
                           "blo 3f\n"
                           "vldrb.8 q0, [%[ip_row_0]], #16\n"
                           "vldrb.8 q1, [%[ip_row_1]], #16\n"
                           "vldrb.8 q4, [%[col_base]], #16\n"
                           "vmladava.s8 %[a00], q0, q4\n"
                           "vmladava.s8 %[a10], q1, q4\n"
                           "vldrb.8 q5, [%[col_base]], #16\n"
                           "vmladava.s8 %[a01], q0, q5\n"
                           "vmladava.s8 %[a11], q1, q5\n"
                           "vldrb.8 q4, [%[col_base]], #16\n"
                           "vmladava.s8 %[a02], q0, q4\n"
                           "vmladava.s8 %[a12], q1, q4\n"
                           "vldrb.8 q5, [%[col_base]], #16\n"
                           "vmladava.s8 %[a03], q0, q5\n"
                           "vmladava.s8 %[a13], q1, q5\n"
                           "subs %[count], %[count], #16\n"
                           "3:\n"
                           "cmp %[count], #0\n"
                           "beq 4f\n"
                           "vmov.32 q6[1], %[count]\n"
                           "vmrs %[count], vpr\n"
                           "vmov.32 q6[0], %[count]\n"
                           "vmov.32 %[count], q6[1]\n"
                           "vctp.8 %[count]\n"
                           "vpst\n"
                           "vldrbt.8 q0, [%[ip_row_0]]\n"
                           "vpst\n"
                           "vldrbt.8 q1, [%[ip_row_1]]\n"
                           "vldrb.8 q4, [%[col_base]], #16\n"
                           "vmladava.s8 %[a00], q0, q4\n"
                           "vmladava.s8 %[a10], q1, q4\n"
                           "vldrb.8 q5, [%[col_base]], #16\n"
                           "vmladava.s8 %[a01], q0, q5\n"
                           "vmladava.s8 %[a11], q1, q5\n"
                           "vldrb.8 q4, [%[col_base]], #16\n"
                           "vmladava.s8 %[a02], q0, q4\n"
                           "vmladava.s8 %[a12], q1, q4\n"
                           "vldrb.8 q5, [%[col_base]], #16\n"
                           "vmladava.s8 %[a03], q0, q5\n"
                           "vmladava.s8 %[a13], q1, q5\n"
                           "vmov.32 %[count], q6[0]\n"
                           "vmsr vpr, %[count]\n"
                           "4:\n"
                           : [ip_row_0] "+&r"(ip_row_0),
                             [ip_row_1] "+&r"(ip_row_1),
                             [col_base] "+&r"(col_base),
                             [count] "+&l"(count),
                             [a00] "+&r"(a00),
                             [a01] "+&r"(a01),
                             [a02] "+&r"(a02),
                             [a03] "+&r"(a03),
                             [a10] "+&r"(a10),
                             [a11] "+&r"(a11),
                             [a12] "+&r"(a12),
                             [a13] "+&r"(a13)
                           :
                           : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "cc", "memory");
    #endif
            int32x4_t acc_n0 = (int32x4_t){a00, a01, a02, a03};
            int32x4_t acc_n1 = (int32x4_t){a10, a11, a12, a13};

            const mve_pred16_t pred = vctp32q(MIN(rhs_rows - i, 4));
            const int32x4_t bias = vldrwq_z_s32(effective_bias + i, pred);
            const int32x4_t multiplier = vldrwq_z_s32(dst_multipliers + i, pred);
            const int32x4_t shift = vldrwq_z_s32(dst_shifts + i, pred);
            acc_n0 = arm_requantize_mve_32x4(vaddq_s32(acc_n0, bias), multiplier, shift);
            acc_n0 = vaddq_n_s32(acc_n0, dst_offset);
            acc_n0 = vmaxq_s32(acc_n0, vdupq_n_s32(activation_min));
            acc_n0 = vminq_s32(acc_n0, vdupq_n_s32(activation_max));
            vstrbq_p_s32(dst + i, acc_n0, pred);

            acc_n1 = arm_requantize_mve_32x4(vaddq_s32(acc_n1, bias), multiplier, shift);
            acc_n1 = vaddq_n_s32(acc_n1, dst_offset);
            acc_n1 = vmaxq_s32(acc_n1, vdupq_n_s32(activation_min));
            acc_n1 = vminq_s32(acc_n1, vdupq_n_s32(activation_max));
            vstrbq_p_s32(dst + row_address_offset + i, acc_n1, pred);
        }
        lhs += 2 * lhs_cols_offset;
        dst += 2 * row_address_offset;
    }

    for (; i_items < lhs_rows; i_items++)
    {
        for (int32_t i = 0; i < rhs_rows; i += 4)
        {
            const int8_t *col_base = rhs + (i / 4) * block_bytes;
            const int8_t *lhs_vec = lhs;
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    #if defined(ARM_MATH_AUTOVECTORIZE)
            for (int32_t k = 0; k < rhs_cols; k++)
            {
                const int32_t offset = (k / 16) * 64 + (k & 15);
                a0 += lhs_vec[k] * col_base[offset];
                a1 += lhs_vec[k] * col_base[offset + 16];
                a2 += lhs_vec[k] * col_base[offset + 32];
                a3 += lhs_vec[k] * col_base[offset + 48];
            }
    #else
            for (int32_t k = 0; k < rhs_cols; k += 16)
            {
                const int8x16_t in = vldrbq_z_s8(lhs_vec + k, vctp8q(MIN(rhs_cols - k, 16)));
                a0 = vmladavaq_s8(a0, in, vldrbq_s8(col_base));
                a1 = vmladavaq_s8(a1, in, vldrbq_s8(col_base + 16));
                a2 = vmladavaq_s8(a2, in, vldrbq_s8(col_base + 32));
                a3 = vmladavaq_s8(a3, in, vldrbq_s8(col_base + 48));
                col_base += 64;
            }
    #endif
            int32x4_t acc_n0 = {a0, a1, a2, a3};

            const mve_pred16_t pred = vctp32q(MIN(rhs_rows - i, 4));
            const int32x4_t bias = vldrwq_z_s32(effective_bias + i, pred);
            const int32x4_t multiplier = vldrwq_z_s32(dst_multipliers + i, pred);
            const int32x4_t shift = vldrwq_z_s32(dst_shifts + i, pred);
            acc_n0 = arm_requantize_mve_32x4(vaddq_s32(acc_n0, bias), multiplier, shift);
            acc_n0 = vaddq_n_s32(acc_n0, dst_offset);
            acc_n0 = vmaxq_s32(acc_n0, vdupq_n_s32(activation_min));
            acc_n0 = vminq_s32(acc_n0, vdupq_n_s32(activation_max));
            vstrbq_p_s32(dst + i, acc_n0, pred);
        }
        lhs += lhs_cols_offset;
        dst += row_address_offset;
    }
    return ARM_CMSIS_NN_SUCCESS;
#else
    (void)lhs;
    (void)rhs;
    (void)effective_bias;
    (void)dst;
    (void)dst_multipliers;
    (void)dst_shifts;
    (void)lhs_rows;
    (void)rhs_rows;
    (void)rhs_cols;
    (void)dst_offset;
    (void)activation_min;
    (void)activation_max;
    (void)row_address_offset;
    (void)lhs_cols_offset;
    return ARM_CMSIS_NN_NO_IMPL_ERROR;
#endif
}
