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
 * Title:        arm_convolve_s8_pack_weights.c
 * Description:  Pack s8 convolution weights for AOT inference
 *
 * $Date:        28 October 2026
 * $Revision:    V.1.0.0
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include <limits.h>
#include <stdint.h>

#if ARM_NN_ENABLE_AOT

arm_cmsis_nn_status arm_convolve_s8_pack_weights(const cmsis_nn_dims *filter_dims,
                                                 const int8_t *filter_data,
                                                 const int32_t groups,
                                                 int8_t *packed_filter)
{
    const int32_t size = arm_convolve_s8_packed_get_filter_size(filter_dims, groups);
    if (!size || !filter_data || !packed_filter)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t cols = filter_dims->h * filter_dims->w * filter_dims->c;
    const int32_t rows = filter_dims->n / groups;
    const uintptr_t source_size = (uintptr_t)filter_dims->n * cols;
    const uintptr_t src = (uintptr_t)filter_data;
    const uintptr_t dst = (uintptr_t)packed_filter;
    /* Difference comparisons avoid wrapping pointer-range endpoints. */
    if (dst >= src ? dst - src < source_size : src - dst < (uintptr_t)size)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    const int32_t padded_cols = (cols + 15) & ~15;
    const int32_t padded_rows = (rows + 3) & ~3;
    for (int32_t group = 0; group < groups; group++)
    {
        const int8_t *group_filter = filter_data + group * rows * cols;
        for (int32_t oc = 0; oc < padded_rows; oc += 4)
        {
            for (int32_t ic = 0; ic < padded_cols; ic += 16)
            {
                for (int32_t lane = 0; lane < 4; lane++)
                {
                    for (int32_t k = 0; k < 16; k++)
                    {
                        *packed_filter++ =
                            oc + lane < rows && ic + k < cols ? group_filter[(oc + lane) * cols + ic + k] : 0;
                    }
                }
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}

#endif /* ARM_NN_ENABLE_AOT */
