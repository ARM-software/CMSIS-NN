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
 * Title:        arm_softmax_f32.c
 * Description:  Softmax function (float32)
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

/**
 * @ingroup Public
 */

/**
 * @addtogroup Softmax
 * @{
 */
/* Refer header file for details. */
arm_cmsis_nn_status
arm_softmax_f32(const float32_t *input, const int32_t num_rows, const int32_t row_size, float32_t *output)
{
    if (input == NULL || output == NULL || num_rows < 1 || row_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    #ifndef NN_DISABLE_SPECIALIZATION
    if (num_rows == 1 && row_size == 2)
    {
        arm_nn_softmax_1x2_f32(input, output);
        return ARM_CMSIS_NN_SUCCESS;
    }
    #endif

    for (int32_t row = 0; row < num_rows; ++row)
    {
        float32_t max_val = input[0];
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        {
            float32x4_t v_max = vdupq_n_f32(ARM_NN_F32_FINITE_LOWEST);
            const float32x4_t v_low = vdupq_n_f32(ARM_NN_F32_FINITE_LOWEST);
            int32_t col_max = 0;
            for (; col_max < row_size; col_max += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(row_size - col_max));
                float32x4_t vin = vld1q_z(input + col_max, p);
                vin = vpselq(vin, v_low, p);
                v_max = vmaxnmq(v_max, vin);
            }
            max_val = vmaxnmvq_f32(ARM_NN_F32_FINITE_LOWEST, v_max);
        }
    #else
        for (int32_t col = 1; col < row_size; ++col)
        {
            if (input[col] > max_val)
            {
                max_val = input[col];
            }
        }
    #endif

        float32_t sum = 0.0f;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        const float32x4_t v_max = vdupq_n_f32(max_val);
        const float32x4_t v_zero = vdupq_n_f32(0.0f);
        int32_t col_exp = 0;
        for (; col_exp < row_size; col_exp += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(row_size - col_exp));
            const float32x4_t x = vsubq(vld1q_z(input + col_exp, p), v_max);
            float32x4_t e = arm_nn_vexpq_poly_mve_f32(x);
            e = vpselq(e, v_zero, p);
            vst1q_p(output + col_exp, e, p);
            sum += arm_nn_vec_reduce_add_f32(e);
        }
    #else
        for (int32_t col = 0; col < row_size; ++col)
        {
            float32_t e = arm_nn_softmax_exp_scalar_f32(input[col] - max_val);
            output[col] = e;
            sum += e;
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

        const float32_t inv_sum = 1.0f / sum;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        const float32x4_t v_inv = vdupq_n_f32(inv_sum);
        int32_t col_norm = 0;
        for (; col_norm < row_size; col_norm += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(row_size - col_norm));
            const float32x4_t out = vmulq(vld1q_z(output + col_norm, p), v_inv);
            vst1q_p(output + col_norm, out, p);
        }
    #else
        for (int32_t col = 0; col < row_size; ++col)
        {
            output[col] *= inv_sum;
        }
    #endif

        input += row_size;
        output += row_size;
    }

    return ARM_CMSIS_NN_SUCCESS;
}

void arm_nn_softmax_1x2_f32(const float32_t in[2], float32_t out[2])
{
    float32_t e0;
    float32_t e1;
    if (in[0] >= in[1])
    {
        e0 = 1.0f;
        e1 = arm_nn_softmax_exp_scalar_f32(in[1] - in[0]);
    }
    else
    {
        e1 = 1.0f;
        e0 = arm_nn_softmax_exp_scalar_f32(in[0] - in[1]);
    }
    const float32_t sum = e0 + e1;
    out[0] = e0 / sum;
    out[1] = e1 / sum;
}

/**
 * @} end of Softmax group
 */

#endif /* ARM_NN_ENABLE_F32 */
