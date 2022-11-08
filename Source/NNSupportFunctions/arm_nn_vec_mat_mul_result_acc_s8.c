/*
 * SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office.com>
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
 * Title:        arm_nn_vec_mat_mul_result_acc_s8.c
 * Description:  Multiplies a matrix by a vector and accumulate with output.
 *
 * $Date:        8 September 2022
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportLSTM
 * @{
 */

/*
 *  Refer to header file for details.
 */
void arm_nn_vec_mat_mul_result_acc_s8(const int8_t *lhs,
                                      const int8_t *rhs,
                                      const int32_t *bias,
                                      int16_t *dst,
                                      const int32_t output_offset,
                                      const int32_t multiplier,
                                      const int32_t shift,
                                      const int32_t rhs_cols,
                                      const int32_t rhs_rows,
                                      const int32_t batch)
{
    for (int i_batch = 0; i_batch < batch; ++i_batch)
    {
        const int8_t *rhs_0 = rhs;
        for (int i_rhs_rows = 0; i_rhs_rows < rhs_rows; ++i_rhs_rows)
        {
            const int8_t *lhs_vec = lhs + i_batch * rhs_cols;
            int32_t acc = bias[i_rhs_rows];
            for (int i_rhs_cols = 0; i_rhs_cols < rhs_cols; ++i_rhs_cols)
            {
                acc += (*lhs_vec++) * (*rhs_0++);
            }
            acc = arm_nn_requantize(acc, multiplier, shift);
            acc += output_offset;

            acc += *dst;
            acc = CLAMP(acc, NN_Q15_MAX, NN_Q15_MIN);
            *dst++ = (int16_t)acc;
        }
    }
}

/**
 * @} end of supportLSTM group
 */
