/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
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
 * Title:        arm_get_buffer_size_common.h
 * Description:  Shared float buffer-size helper utilities
 *
 * $Date:        30 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_GET_BUFFER_SIZE_COMMON_H
#define ARM_GET_BUFFER_SIZE_COMMON_H

#include "arm_nn_math_types.h"

static inline int arm_nn_checked_size_mul(size_t lhs, size_t rhs, size_t *out)
{
    if (lhs != 0U && rhs > (((size_t)-1) / lhs))
    {
        return 0;
    }

    *out = lhs * rhs;
    return 1;
}

static inline int32_t arm_nn_size_to_i32_or_zero(size_t size) { return (size > (size_t)INT32_MAX) ? 0 : (int32_t)size; }

#endif /* ARM_GET_BUFFER_SIZE_COMMON_H */
