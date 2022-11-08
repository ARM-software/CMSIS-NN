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
 * Title:        arm_nn_lstm_update_cell_state_s16.c
 * Description:  Update cell state for an incremental step of LSTM function.
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
 * Update cell state for a single LSTM iteration step, int8x8_16 version.
 *
 * Refer to header file for more details
 */
void arm_nn_lstm_update_cell_state_s16(const int32_t n_block,
                                       const int32_t cell_state_scale,
                                       int16_t *cell_state,
                                       const int16_t *input_gate,
                                       const int16_t *forget_gate,
                                       const int16_t *cell_gate)
{
    const int32_t cell_scale = 30 + cell_state_scale;
    for (int i = 0; i < n_block; ++i)
    {
        int32_t value = cell_state[i] * forget_gate[i];
        int32_t value_1 = input_gate[i] * cell_gate[i];

        value = arm_nn_divide_by_power_of_two(value, 15);
        value_1 = arm_nn_divide_by_power_of_two(value_1, cell_scale);

        cell_state[i] = CLAMP(value + value_1, NN_Q15_MAX, NN_Q15_MIN);
    }
}
/**
 * @} end of supportLSTM group
 */
