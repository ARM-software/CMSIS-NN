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
 * Title:        arm_nn_depthwise_conv_nt_t_f16.c
 * Description:  Support: depthwise NT_T kernel for f16
 *
 * $Date:        09 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

arm_cmsis_nn_status arm_nn_depthwise_conv_nt_t_f16(const float16_t *__RESTRICT lhs,
                                                   const float16_t *__RESTRICT rhs,
                                                   const float16_t *__RESTRICT bias,
                                                   float16_t *__RESTRICT out,
                                                   int32_t lhs_rows,
                                                   int32_t total_ch,
                                                   int32_t row_x_col,
                                                   int32_t out_row_stride,
                                                   float16_t activation_min,
                                                   float16_t activation_max)
{
    if (!lhs || !rhs || !out || lhs_rows < 1 || lhs_rows > 4 || total_ch <= 0 || row_x_col <= 0 ||
        out_row_stride < total_ch)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    int32_t c = 0;
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t v_act_min = vdupq_n_f16(activation_min);
    const float16x8_t v_act_max = vdupq_n_f16(activation_max);
    for (; c + 8 <= total_ch; c += 8)
    {
        float16x8_t acc0 = bias ? vld1q(bias + c) : vdupq_n_f16((float16_t)0.0f);
        float16x8_t acc1 = acc0;
        float16x8_t acc2 = acc0;
        float16x8_t acc3 = acc0;

        const float16_t *lhs_row0_base = lhs + c;
        const float16_t *lhs_row1_base = lhs + (size_t)row_x_col * total_ch + c;
        const float16_t *lhs_row2_base = lhs + (size_t)2 * row_x_col * total_ch + c;
        const float16_t *lhs_row3_base = lhs + (size_t)3 * row_x_col * total_ch + c;
        switch (lhs_rows)
        {
        case 1: {
            for (int32_t k = 0; k < row_x_col; ++k)
            {
                const size_t offset = (size_t)k * total_ch;
                const float16x8_t w = vld1q(rhs + offset + c);
                acc0 = vfmaq(acc0, vld1q(lhs_row0_base + offset), w);
            }
            acc0 = arm_nn_clamp_mve_f16(acc0, v_act_min, v_act_max);
            vst1q(out + c, acc0);
            break;
        }
        case 2: {
            for (int32_t k = 0; k < row_x_col; ++k)
            {
                const size_t offset = (size_t)k * total_ch;
                const float16x8_t w = vld1q(rhs + offset + c);
                acc0 = vfmaq(acc0, vld1q(lhs_row0_base + offset), w);
                acc1 = vfmaq(acc1, vld1q(lhs_row1_base + offset), w);
            }
            acc0 = arm_nn_clamp_mve_f16(acc0, v_act_min, v_act_max);
            vst1q(out + c, acc0);
            acc1 = arm_nn_clamp_mve_f16(acc1, v_act_min, v_act_max);
            vst1q(out + out_row_stride + c, acc1);
            break;
        }
        case 3: {
            for (int32_t k = 0; k < row_x_col; ++k)
            {
                const size_t offset = (size_t)k * total_ch;
                const float16x8_t w = vld1q(rhs + offset + c);
                acc0 = vfmaq(acc0, vld1q(lhs_row0_base + offset), w);
                acc1 = vfmaq(acc1, vld1q(lhs_row1_base + offset), w);
                acc2 = vfmaq(acc2, vld1q(lhs_row2_base + offset), w);
            }
            acc0 = arm_nn_clamp_mve_f16(acc0, v_act_min, v_act_max);
            vst1q(out + c, acc0);
            acc1 = arm_nn_clamp_mve_f16(acc1, v_act_min, v_act_max);
            vst1q(out + out_row_stride + c, acc1);
            acc2 = arm_nn_clamp_mve_f16(acc2, v_act_min, v_act_max);
            vst1q(out + (size_t)2 * out_row_stride + c, acc2);
            break;
        }
        default: {
            for (int32_t k = 0; k < row_x_col; ++k)
            {
                const size_t offset = (size_t)k * total_ch;
                const float16x8_t w = vld1q(rhs + offset + c);
                acc0 = vfmaq(acc0, vld1q(lhs_row0_base + offset), w);
                acc1 = vfmaq(acc1, vld1q(lhs_row1_base + offset), w);
                acc2 = vfmaq(acc2, vld1q(lhs_row2_base + offset), w);
                acc3 = vfmaq(acc3, vld1q(lhs_row3_base + offset), w);
            }
            acc0 = arm_nn_clamp_mve_f16(acc0, v_act_min, v_act_max);
            vst1q(out + c, acc0);
            acc1 = arm_nn_clamp_mve_f16(acc1, v_act_min, v_act_max);
            vst1q(out + out_row_stride + c, acc1);
            acc2 = arm_nn_clamp_mve_f16(acc2, v_act_min, v_act_max);
            vst1q(out + (size_t)2 * out_row_stride + c, acc2);
            acc3 = arm_nn_clamp_mve_f16(acc3, v_act_min, v_act_max);
            vst1q(out + (size_t)3 * out_row_stride + c, acc3);
            break;
        }
        }
    }

    if (c < total_ch)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(total_ch - c));
        float16x8_t acc0 = bias ? vld1q_z(bias + c, p) : vdupq_n_f16((float16_t)0.0f);
        float16x8_t acc1 = acc0;
        float16x8_t acc2 = acc0;
        float16x8_t acc3 = acc0;

        const float16_t *lhs_row0_base = lhs + c;
        const float16_t *lhs_row1_base = lhs + (size_t)row_x_col * total_ch + c;
        const float16_t *lhs_row2_base = lhs + (size_t)2 * row_x_col * total_ch + c;
        const float16_t *lhs_row3_base = lhs + (size_t)3 * row_x_col * total_ch + c;
        switch (lhs_rows)
        {
        case 1: {
            for (int32_t k = 0; k < row_x_col; ++k)
            {
                const size_t offset = (size_t)k * total_ch;
                const float16x8_t w = vld1q_z(rhs + offset + c, p);
                acc0 = vfmaq(acc0, vld1q_z(lhs_row0_base + offset, p), w);
            }
            acc0 = arm_nn_clamp_mve_f16(acc0, v_act_min, v_act_max);
            vst1q_p(out + c, acc0, p);
            break;
        }
        case 2: {
            for (int32_t k = 0; k < row_x_col; ++k)
            {
                const size_t offset = (size_t)k * total_ch;
                const float16x8_t w = vld1q_z(rhs + offset + c, p);
                acc0 = vfmaq(acc0, vld1q_z(lhs_row0_base + offset, p), w);
                acc1 = vfmaq(acc1, vld1q_z(lhs_row1_base + offset, p), w);
            }
            acc0 = arm_nn_clamp_mve_f16(acc0, v_act_min, v_act_max);
            vst1q_p(out + c, acc0, p);
            acc1 = arm_nn_clamp_mve_f16(acc1, v_act_min, v_act_max);
            vst1q_p(out + out_row_stride + c, acc1, p);
            break;
        }
        case 3: {
            for (int32_t k = 0; k < row_x_col; ++k)
            {
                const size_t offset = (size_t)k * total_ch;
                const float16x8_t w = vld1q_z(rhs + offset + c, p);
                acc0 = vfmaq(acc0, vld1q_z(lhs_row0_base + offset, p), w);
                acc1 = vfmaq(acc1, vld1q_z(lhs_row1_base + offset, p), w);
                acc2 = vfmaq(acc2, vld1q_z(lhs_row2_base + offset, p), w);
            }
            acc0 = arm_nn_clamp_mve_f16(acc0, v_act_min, v_act_max);
            vst1q_p(out + c, acc0, p);
            acc1 = arm_nn_clamp_mve_f16(acc1, v_act_min, v_act_max);
            vst1q_p(out + out_row_stride + c, acc1, p);
            acc2 = arm_nn_clamp_mve_f16(acc2, v_act_min, v_act_max);
            vst1q_p(out + (size_t)2 * out_row_stride + c, acc2, p);
            break;
        }
        default: {
            for (int32_t k = 0; k < row_x_col; ++k)
            {
                const size_t offset = (size_t)k * total_ch;
                const float16x8_t w = vld1q_z(rhs + offset + c, p);
                acc0 = vfmaq(acc0, vld1q_z(lhs_row0_base + offset, p), w);
                acc1 = vfmaq(acc1, vld1q_z(lhs_row1_base + offset, p), w);
                acc2 = vfmaq(acc2, vld1q_z(lhs_row2_base + offset, p), w);
                acc3 = vfmaq(acc3, vld1q_z(lhs_row3_base + offset, p), w);
            }
            acc0 = arm_nn_clamp_mve_f16(acc0, v_act_min, v_act_max);
            vst1q_p(out + c, acc0, p);
            acc1 = arm_nn_clamp_mve_f16(acc1, v_act_min, v_act_max);
            vst1q_p(out + out_row_stride + c, acc1, p);
            acc2 = arm_nn_clamp_mve_f16(acc2, v_act_min, v_act_max);
            vst1q_p(out + (size_t)2 * out_row_stride + c, acc2, p);
            acc3 = arm_nn_clamp_mve_f16(acc3, v_act_min, v_act_max);
            vst1q_p(out + (size_t)3 * out_row_stride + c, acc3, p);
            break;
        }
        }
        c = total_ch;
    }
#endif

    for (int32_t row = 0; row < lhs_rows; ++row)
    {
        float16_t *out_row = out + (size_t)row * out_row_stride;
        for (int32_t cc = c; cc < total_ch; ++cc)
        {
            _Float16 acc = bias ? (_Float16)bias[cc] : (_Float16)0.0f;
            for (int32_t k = 0; k < row_x_col; ++k)
            {
                const float16_t *lhs_val = lhs + ((size_t)row * row_x_col + k) * total_ch + cc;
                const float16_t *rhs_val = rhs + (size_t)k * total_ch + cc;
                acc += (_Float16)(*lhs_val) * (_Float16)(*rhs_val);
            }
            out_row[cc] = (float16_t)CLAMP(acc, (_Float16)activation_max, (_Float16)activation_min);
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of supportConvolution group
 */
