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
 * Title:        arm_elementwise_mul_f32.c
 * Description:  Elementwise multiply for float32 tensors
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

arm_cmsis_nn_status arm_elementwise_mul_f32(const float32_t *input_1_vect,
                                            const float32_t *input_2_vect,
                                            float32_t *output,
                                            float32_t out_activation_min,
                                            float32_t out_activation_max,
                                            int32_t block_size)
{
    if (!input_1_vect || !input_2_vect || !output || block_size < 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    int32_t i = 0;
    const float32x4_t vmin = vdupq_n_f32(out_activation_min);
    const float32x4_t vmax = vdupq_n_f32(out_activation_max);
    for (; i < block_size; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(block_size - i));
        const float32x4_t va = vld1q_z(&input_1_vect[i], p);
        const float32x4_t vb = vld1q_z(&input_2_vect[i], p);
        float32x4_t vr = vmulq(va, vb);
        vr = arm_nn_clamp_mve_f32(vr, vmin, vmax);
        vstrwq_p(&output[i], vr, p);
    }
#else
    for (int32_t i = 0; i < block_size; ++i)
    {
        const float32_t v = input_1_vect[i] * input_2_vect[i];
        output[i] = CLAMP(v, out_activation_max, out_activation_min);
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of groupElementwise group
 */
