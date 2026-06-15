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
 * Title:        arm_nn_activation_f16.c
 * Description:  Activation functions (float16)
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Acti
 * @{
 */

/* Refer header file for details. */
arm_cmsis_nn_status arm_nn_activation_f16(const float16_t *input,
                                          float16_t *output,
                                          int32_t size,
                                          arm_nn_activation_type_flt type,
                                          float16_t act_param)
{
    if (input == NULL || output == NULL || size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const size_t n = (size_t)size;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (type == ARM_NN_FLT_ACT_TANH)
    {
        for (size_t i = 0; i < n; i += 8U)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(n - i));
            float16x8_t v = vld1q_z(&input[i], p);
            v = arm_nn_vtanh_lut_direct_mve_f16(v);
            vst1q_p(&output[i], v, p);
        }
        return ARM_CMSIS_NN_SUCCESS;
    }

    if (type == ARM_NN_FLT_ACT_HARDSWISH)
    {
        for (size_t i = 0; i < n; i += 8U)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(n - i));
            float16x8_t v = vld1q_z(&input[i], p);
            v = arm_nn_vhardswish_mve_f16(v);
            vst1q_p(&output[i], v, p);
        }
        return ARM_CMSIS_NN_SUCCESS;
    }

    if (type == ARM_NN_FLT_ACT_RELU)
    {
        const float16x8_t v_zero = vdupq_n_f16((float16_t)0.0f);
        for (size_t i = 0; i < n; i += 8U)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(n - i));
            float16x8_t v = vld1q_z(&input[i], p);
            v = vmaxnmq(v, v_zero);
            vst1q_p(&output[i], v, p);
        }
        return ARM_CMSIS_NN_SUCCESS;
    }

    if (type == ARM_NN_FLT_ACT_RELU6)
    {
        const float16x8_t v_zero = vdupq_n_f16((float16_t)0.0f);
        const float16x8_t v_six = vdupq_n_f16((float16_t)6.0f);
        for (size_t i = 0; i < n; i += 8U)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(n - i));
            float16x8_t v = vld1q_z(&input[i], p);
            v = arm_nn_clamp_mve_f16(v, v_zero, v_six);
            vst1q_p(&output[i], v, p);
        }
        return ARM_CMSIS_NN_SUCCESS;
    }
    #endif

    for (size_t i = 0; i < n; ++i)
    {
        output[i] = arm_nn_apply_activation_type_f16(input[i], type, act_param);
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Acti group
 */

#endif /* ARM_NN_ENABLE_F16 */
