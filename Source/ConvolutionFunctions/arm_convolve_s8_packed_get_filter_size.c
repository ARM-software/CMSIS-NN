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
 * Title:        arm_convolve_s8_packed_get_filter_size.c
 * Description:  Query storage for packed s8 convolution weights
 *
 * $Date:        28 October 2026
 * $Revision:    V.1.0.0
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include <limits.h>
#include <stdint.h>

#if ARM_NN_ENABLE_AOT

int32_t arm_convolve_s8_packed_get_filter_size(const cmsis_nn_dims *filter_dims, const int32_t groups)
{
    if (!filter_dims || groups <= 0 || filter_dims->n <= 0 || filter_dims->h <= 0 || filter_dims->w <= 0 ||
        filter_dims->c <= 0 || filter_dims->n % groups != 0)
    {
        return 0;
    }
    /* Bound each multiplication before evaluating it, even for INT32_MAX dimensions. */
    int64_t cols = filter_dims->h;
    if (cols > INT32_MAX / filter_dims->w)
    {
        return 0;
    }
    cols *= filter_dims->w;
    if (cols > INT32_MAX / filter_dims->c)
    {
        return 0;
    }
    cols *= filter_dims->c;
    cols = (cols + 15) & ~INT64_C(15);
    const int64_t rows = ((int64_t)filter_dims->n / groups + 3) & ~INT64_C(3);
    if (cols > INT32_MAX / rows || rows * cols > INT32_MAX / groups)
    {
        return 0;
    }
    return (int32_t)(groups * rows * cols);
}

#endif /* ARM_NN_ENABLE_AOT */
