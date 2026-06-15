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
 * Title:        arm_lstm_unidirectional_f16.c
 * Description:  Unidirectional LSTM (float16)
 *
 * $Date:        26 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture with FP16 support
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup LSTM
 * @{
 */

/*
 * Unidirectional LSTM (float16)
 *
 * Refer to header file for details.
 *
 */

arm_cmsis_nn_status arm_lstm_unidirectional_f16(const float16_t *input,
                                                float16_t *output,
                                                const cmsis_nn_lstm_params_f16 *params,
                                                cmsis_nn_lstm_context_f16 *buffers)
{
    if (!input || !output || !params || !buffers || !buffers->cell_state)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    float16_t *hidden_in = NULL;
    arm_memset_f16(
        buffers->cell_state, (float16_t)0.0f, (uint32_t)((size_t)params->batch_size * (size_t)params->hidden_size));

    if (params->time_major)
    {
        for (int32_t t = 0; t < params->time_steps; t++)
        {
            const float16_t *data_in = input + (size_t)t * (size_t)params->batch_size * (size_t)params->input_size;
            float16_t *hidden_out = output + (size_t)t * (size_t)params->batch_size * (size_t)params->hidden_size;
            arm_cmsis_nn_status status = arm_nn_lstm_step_f16(data_in, hidden_in, hidden_out, params, buffers, 1);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
            hidden_in = hidden_out;
        }
    }
    else
    {
        for (int32_t t = 0; t < params->time_steps; t++)
        {
            const float16_t *data_in = input + (size_t)t * (size_t)params->input_size;
            float16_t *hidden_out = output + (size_t)t * (size_t)params->hidden_size;
            arm_cmsis_nn_status status =
                arm_nn_lstm_step_f16(data_in, hidden_in, hidden_out, params, buffers, params->time_steps);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
            hidden_in = hidden_out;
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/** @} */
