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
 * Title:        arm_elementwise_mul_s16_s8.c
 * Description:  Elementwise multiplication of 16 bit input with 8 bit output
 *
 * $Date:        8 September 2022
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup groupSupport
 */

/**
 * @addtogroup BasicMath
 * @{
 */

/*
 * s16 elementwise multiplication with s8 output
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_elementwise_mul_s16_s8(const int16_t *input_1_vect,
                                               const int16_t *input_2_vect,
                                               int8_t *output,
                                               const int32_t out_offset,
                                               const int32_t out_mult,
                                               const int32_t out_shift,
                                               const int32_t block_size)
{
    int32_t loop_count = block_size;
    while (loop_count > 0)
    {
        int16_t input_1 = *input_1_vect++;
        int16_t input_2 = *input_2_vect++;

        int32_t mul_res = input_1 * input_2;
        mul_res = arm_nn_requantize(mul_res, out_mult, out_shift) + out_offset;

        mul_res = CLAMP(mul_res, NN_Q7_MAX, NN_Q7_MIN);
        *output++ = (int8_t)mul_res;

        loop_count--;
    }

    return ARM_CMSIS_NN_SUCCESS;
}
/**
 * @} end of BasicMath group
 */
