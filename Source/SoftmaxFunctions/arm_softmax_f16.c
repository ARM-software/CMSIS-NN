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
 * Title:        arm_softmax_f16.c
 * Description:  Softmax function (float16)
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture with FP16 support
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 */

/**
 * @addtogroup Softmax
 * @{
 */

/* Refer header file for details. */
arm_cmsis_nn_status arm_softmax_f16(const float16_t *input, int32_t rows, int32_t cols, float16_t *output)
{
    if (input == NULL || output == NULL || rows < 1 || cols < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    #ifndef NN_DISABLE_SPECIALIZATION
    if (rows == 1 && cols == 2)
    {
        arm_nn_softmax_1x2_f16(input, output);
        return ARM_CMSIS_NN_SUCCESS;
    }
    #endif

    for (int32_t r = 0; r < rows; ++r)
    {
        const float16_t *in_row = input + r * cols;
        float16_t *out_row = output + r * cols;

        /* Subtracting row max avoids overflow in exp and preserves relative probabilities. */
        float16_t max_val;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        {
            const _Float16 f16_finite_max = (_Float16)ARM_NN_F16_FINITE_MAX;
            const _Float16 f16_finite_lowest = -f16_finite_max;
            const float16x8_t v_low = vdupq_n_f16((float16_t)f16_finite_lowest);
            float16x8_t v_max = v_low;
            int32_t c = 0;
            for (; c < cols; c += 8)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(cols - c));
                float16x8_t vin = vld1q_z(in_row + c, p);
                vin = vpselq(vin, v_low, p);
                v_max = vmaxnmq(v_max, vin);
            }
            max_val = vmaxnmvq_f16((float16_t)f16_finite_lowest, v_max);
        }
    #else
        _Float16 max_val_scalar = (_Float16)in_row[0];
        for (int32_t c = 1; c < cols; ++c)
        {
            const _Float16 in_val = (_Float16)in_row[c];
            if (in_val > max_val_scalar)
            {
                max_val_scalar = in_val;
            }
        }
        max_val = (float16_t)max_val_scalar;
    #endif

        /* Compute exp(x - max) and sum */
        float32_t sum = 0.0f;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        const float16x8_t v_max = vdupq_n_f16(max_val);
        const float16x8_t v_zero = vdupq_n_f16((float16_t)0.0f);
        int32_t c = 0;
        for (; c < cols; c += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(cols - c));
            const float16x8_t x = vsubq(vld1q_z(in_row + c, p), v_max);
            float16x8_t e = arm_nn_vexpq_poly_mve_f16(x);
            e = vpselq(e, v_zero, p);
            vst1q_p(out_row + c, e, p);
            sum += (float32_t)arm_nn_vec_reduce_add_f16(e);
        }
    #else
        for (int32_t c = 0; c < cols; ++c)
        {
            const _Float16 x = (_Float16)in_row[c];
            const float16_t e = arm_nn_softmax_exp_scalar_f16((float16_t)(x - (_Float16)max_val));
            out_row[c] = e;
            sum += (float32_t)e;
        }
    #endif

        /*
         * For any finite row, subtracting the row max guarantees at least one
         * exponent term is exp(0) = 1, so the accumulated sum should stay >= 1.
         * Treat non-positive sums as invalid input rather than silently
         * normalizing a degenerate row.
         */
        if (sum <= 0.0f)
        {
            return ARM_CMSIS_NN_ARG_ERROR;
        }

        /* Normalize */
        const _Float16 inv_sum = (_Float16)(1.0f / sum);
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        const float16x8_t v_inv = vdupq_n_f16((float16_t)inv_sum);
        int32_t c_norm = 0;
        for (; c_norm < cols; c_norm += 8)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(cols - c_norm));
            const float16x8_t out = vmulq(vld1q_z(out_row + c_norm, p), v_inv);
            vst1q_p(out_row + c_norm, out, p);
        }
    #else
        for (int32_t c = 0; c < cols; ++c)
        {
            out_row[c] = (float16_t)((_Float16)out_row[c] * inv_sum);
        }
    #endif
    }

    return ARM_CMSIS_NN_SUCCESS;
}

void arm_nn_softmax_1x2_f16(const float16_t in[2], float16_t out[2])
{
    _Float16 e0;
    _Float16 e1;
    if ((_Float16)in[0] >= (_Float16)in[1])
    {
        e0 = (_Float16)1;
        e1 = arm_nn_softmax_exp_scalar_f16((float16_t)((_Float16)in[1] - (_Float16)in[0]));
    }
    else
    {
        e1 = (_Float16)1;
        e0 = arm_nn_softmax_exp_scalar_f16((float16_t)((_Float16)in[0] - (_Float16)in[1]));
    }
    const _Float16 sum = e0 + e1;
    out[0] = (float16_t)(e0 / sum);
    out[1] = (float16_t)(e1 / sum);
}

/**
 * @} end of Softmax group
 */

#endif /* ARM_NN_ENABLE_F16 */
