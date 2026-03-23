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
 * Title:        arm_reshape_f32.c
 * Description:  Reshape a float32 vector
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
 * @addtogroup Reshape
 * @{
 */

/* Refer header file for details. */
void arm_reshape_f32(const float32_t *input, float32_t *output, const uint32_t total_size)
{
    if (total_size == 0U || input == output)
    {
        return;
    }

    arm_memcpy_f32(output, input, total_size);
}

/**
 * @} end of Reshape group
 */

#endif /* ARM_NN_ENABLE_F32 */
