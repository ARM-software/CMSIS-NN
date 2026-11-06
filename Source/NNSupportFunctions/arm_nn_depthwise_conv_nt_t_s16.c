/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_depthwise_conv_nt_t_s16.c
 * Description:  Depthwise convolution on matrices with no padding.
 *
 * $Date:        13 November 2026
 * $Revision:    V.1.0.2
 *
 * Target Processor:  Cortex-M processors with MVE extension
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"
#include <stdbool.h>

#if defined(ARM_MATH_MVEI)
__STATIC_FORCEINLINE void arm_nn_depthwise_conv_nt_t_s16_block(const int16_t *lhs,
                                                               const int8_t *rhs,
                                                               const uint16_t num_ch,
                                                               const int32_t *out_shift,
                                                               const int32_t *out_mult,
                                                               const int32_t activation_min,
                                                               const int32_t activation_max,
                                                               const uint16_t row_x_col,
                                                               const int64_t *const bias,
                                                               const int32_t offset,
                                                               int16_t *out,
                                                               const bool is_tail,
                                                               const uint32_t tail_ch,
                                                               const mve_pred16_t p)
{
    const int8_t *rhs_0 = rhs + offset;
    const int16_t *lhs_0 = lhs + offset;
    const int16_t *lhs_1 = lhs + row_x_col * num_ch + offset;
    const int16_t *lhs_2 = lhs + (row_x_col * num_ch * 2) + offset;
    const int16_t *lhs_3 = lhs + (row_x_col * num_ch * 3) + offset;

    int32x4_t out_0 = vdupq_n_s32(0);
    int32x4_t out_1 = vdupq_n_s32(0);
    int32x4_t out_2 = vdupq_n_s32(0);
    int32x4_t out_3 = vdupq_n_s32(0);

    for (int i_row_x_col = 0; i_row_x_col < row_x_col; i_row_x_col++)
    {
        const int32x4_t ker_0 = is_tail ? vldrbq_z_s32(rhs_0, p) : vldrbq_s32(rhs_0);

        int32x4_t ip_0 = is_tail ? vldrhq_z_s32(lhs_0, p) : vldrhq_s32(lhs_0);
        out_0 += vmulq_s32(ip_0, ker_0);

        int32x4_t ip_1 = is_tail ? vldrhq_z_s32(lhs_1, p) : vldrhq_s32(lhs_1);
        out_1 += vmulq_s32(ip_1, ker_0);

        int32x4_t ip_2 = is_tail ? vldrhq_z_s32(lhs_2, p) : vldrhq_s32(lhs_2);
        out_2 += vmulq_s32(ip_2, ker_0);

        int32x4_t ip_3 = is_tail ? vldrhq_z_s32(lhs_3, p) : vldrhq_s32(lhs_3);
        out_3 += vmulq_s32(ip_3, ker_0);

        lhs_0 += num_ch;
        lhs_1 += num_ch;
        lhs_2 += num_ch;
        lhs_3 += num_ch;

        rhs_0 += num_ch;
    }

    const uint32_t req_ch = is_tail ? tail_ch : 4U;
    for (uint32_t i_requantize = 0; i_requantize < req_ch; i_requantize++)
    {
        int32_t reduced_multiplier = REDUCE_MULTIPLIER(out_mult[offset + i_requantize]);
        int32_t shift = out_shift[offset + i_requantize];
        int64_t in_requantize_0 = (int64_t)out_0[i_requantize];
        int64_t in_requantize_1 = (int64_t)out_1[i_requantize];
        int64_t in_requantize_2 = (int64_t)out_2[i_requantize];
        int64_t in_requantize_3 = (int64_t)out_3[i_requantize];

        if (bias)
        {
            in_requantize_0 += bias[offset + i_requantize];
            in_requantize_1 += bias[offset + i_requantize];
            in_requantize_2 += bias[offset + i_requantize];
            in_requantize_3 += bias[offset + i_requantize];
        }

        out_0[i_requantize] = arm_nn_requantize_s64(in_requantize_0, reduced_multiplier, shift);
        out_1[i_requantize] = arm_nn_requantize_s64(in_requantize_1, reduced_multiplier, shift);
        out_2[i_requantize] = arm_nn_requantize_s64(in_requantize_2, reduced_multiplier, shift);
        out_3[i_requantize] = arm_nn_requantize_s64(in_requantize_3, reduced_multiplier, shift);
    }

    out_0 = vmaxq_s32(out_0, vdupq_n_s32(activation_min));
    out_0 = vminq_s32(out_0, vdupq_n_s32(activation_max));

    out_1 = vmaxq_s32(out_1, vdupq_n_s32(activation_min));
    out_1 = vminq_s32(out_1, vdupq_n_s32(activation_max));

    out_2 = vmaxq_s32(out_2, vdupq_n_s32(activation_min));
    out_2 = vminq_s32(out_2, vdupq_n_s32(activation_max));

    out_3 = vmaxq_s32(out_3, vdupq_n_s32(activation_min));
    out_3 = vminq_s32(out_3, vdupq_n_s32(activation_max));

    if (is_tail)
    {
        vstrhq_p_s32(out, out_0, p);
        vstrhq_p_s32(out + num_ch, out_1, p);
        vstrhq_p_s32(out + 2 * num_ch, out_2, p);
        vstrhq_p_s32(out + 3 * num_ch, out_3, p);
    }
    else
    {
        vstrhq_s32(out, out_0);
        vstrhq_s32(out + num_ch, out_1);
        vstrhq_s32(out + 2 * num_ch, out_2);
        vstrhq_s32(out + 3 * num_ch, out_3);
    }
}
#endif

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

/*
 * Depthwise convolution of rhs matrix with 4 lhs matrices with no padding. Dimensions are the same for lhs and rhs.
 *
 * Refer header file for details.
 *
 */
int16_t *arm_nn_depthwise_conv_nt_t_s16(const int16_t *lhs,
                                        const int8_t *rhs,
                                        const uint16_t num_ch,
                                        const int32_t *out_shift,
                                        const int32_t *out_mult,
                                        const int32_t activation_min,
                                        const int32_t activation_max,
                                        const uint16_t row_x_col,
                                        const int64_t *const output_bias,
                                        int16_t *out)
{
#if defined(ARM_MATH_MVEI)

    const int32_t full_blocks = num_ch / 4;
    const uint32_t tail_ch = (uint32_t)num_ch & 0x3U;
    int32_t offset = 0;

    for (int i_loop_cnt = 0; i_loop_cnt < full_blocks; i_loop_cnt++, offset += 4, out += 4)
    {
        arm_nn_depthwise_conv_nt_t_s16_block(lhs,
                                             rhs,
                                             num_ch,
                                             out_shift,
                                             out_mult,
                                             activation_min,
                                             activation_max,
                                             row_x_col,
                                             output_bias,
                                             offset,
                                             out,
                                             false,
                                             4U,
                                             0);
    }

    if (tail_ch != 0)
    {
        arm_nn_depthwise_conv_nt_t_s16_block(lhs,
                                             rhs,
                                             num_ch,
                                             out_shift,
                                             out_mult,
                                             activation_min,
                                             activation_max,
                                             row_x_col,
                                             output_bias,
                                             offset,
                                             out,
                                             true,
                                             tail_ch,
                                             vctp32q(tail_ch));
        out += tail_ch;
    }

    return out + (3 * num_ch);
#else
    (void)lhs;
    (void)rhs;
    (void)num_ch;
    (void)out_shift;
    (void)out_mult;
    (void)activation_min;
    (void)activation_max;
    (void)row_x_col;
    (void)output_bias;
    (void)out;
    return NULL;
#endif
}

/**
 * @} end of Doxygen group
 */
