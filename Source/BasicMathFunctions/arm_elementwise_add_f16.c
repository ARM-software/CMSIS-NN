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
 * Title:        arm_elementwise_add_f16.c
 * Description:  Elementwise add for float16 tensors
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

arm_cmsis_nn_status arm_elementwise_add_f16(const float16_t *input_1_vect,
                                            const float16_t *input_2_vect,
                                            float16_t *output,
                                            float16_t out_activation_min,
                                            float16_t out_activation_max,
                                            int32_t block_size)
{
    if (!input_1_vect || !input_2_vect || !output || block_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    int32_t i = 0;
    const float16x8_t vmin = vdupq_n_f16(out_activation_min);
    const float16x8_t vmax = vdupq_n_f16(out_activation_max);
    for (; i < block_size; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(block_size - i));
        const float16x8_t va = vld1q_z(&input_1_vect[i], p);
        const float16x8_t vb = vld1q_z(&input_2_vect[i], p);
        float16x8_t vr = vaddq(va, vb);
        vr = arm_nn_clamp_mve_f16(vr, vmin, vmax);
        vstrhq_p(&output[i], vr, p);
    }
#else
    for (int32_t i = 0; i < block_size; ++i)
    {
        const _Float16 v = (_Float16)input_1_vect[i] + (_Float16)input_2_vect[i];
        output[i] = arm_nn_clamp_scalar_f16((float16_t)v, out_activation_min, out_activation_max);
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of groupElementwise group
 */
