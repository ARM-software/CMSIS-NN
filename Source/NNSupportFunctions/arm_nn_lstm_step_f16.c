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
 * Title:        arm_nn_lstm_step_f16.c
 * Description:  Update LSTM function for a single iteration step (float16)
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture with FP16 support
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportLSTM
 * @{
 */

/*
 * Calculate the output state tensor of an LSTM step, float16 version.
 * Refer to header file for details.
 */

__STATIC_INLINE float16_t arm_nn_lstm_activate_gate_f16(float16_t x, arm_nn_activation_type_flt type)
{
    if (type == ARM_NN_FLT_ACT_TANH)
    {
        return arm_nn_tanh_scalar_ref_f16(x);
    }
    return arm_nn_sigmoid_scalar_f16(x);
}

__STATIC_INLINE float16_t arm_nn_lstm_dot_f16(const float16_t *lhs, const float16_t *rhs, int32_t count)
{
    _Float16 acc = (_Float16)0.0f;

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);

    for (int32_t i = 0; i < count; i += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(count - i));
        vacc = vfmaq(vacc, vld1q_z(lhs + i, p), vld1q_z(rhs + i, p));
    }

    acc += (_Float16)arm_nn_vec_reduce_add_f16(vacc);
#else
    for (int32_t i = 0; i < count; ++i)
    {
        acc += (_Float16)lhs[i] * (_Float16)rhs[i];
    }
#endif

    return (float16_t)acc;
}

static float16_t arm_nn_lstm_gate_compute_f16(const cmsis_nn_lstm_gate_f16 *gate,
                                              const float16_t *input,
                                              const float16_t *hidden,
                                              int32_t input_size,
                                              int32_t hidden_size,
                                              int32_t h_idx)
{
    _Float16 acc = (_Float16)0.0f;
    if (gate->bias)
    {
        acc = (_Float16)gate->bias[h_idx];
    }
    if (gate->input_weights)
    {
        const float16_t *w_in = gate->input_weights + (size_t)h_idx * (size_t)input_size;
        acc += (_Float16)arm_nn_lstm_dot_f16(input, w_in, input_size);
    }
    if (hidden && gate->hidden_weights)
    {
        const float16_t *w_h = gate->hidden_weights + (size_t)h_idx * (size_t)hidden_size;
        acc += (_Float16)arm_nn_lstm_dot_f16(hidden, w_h, hidden_size);
    }
    return arm_nn_lstm_activate_gate_f16((float16_t)acc, gate->activation_type);
}

arm_cmsis_nn_status arm_nn_lstm_step_f16(const float16_t *data_in,
                                         const float16_t *hidden_in,
                                         float16_t *hidden_out,
                                         const cmsis_nn_lstm_params_f16 *params,
                                         cmsis_nn_lstm_context_f16 *buffers,
                                         const int32_t batch_offset)
{
    if (!data_in || !hidden_out || !params || !buffers || !buffers->cell_state || batch_offset <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = params->batch_size;
    const int32_t input_size = params->input_size;
    const int32_t hidden_size = params->hidden_size;
    const _Float16 cell_clip = (_Float16)params->cell_clip;

    float16_t *cell_state = buffers->cell_state;

    for (int32_t b = 0; b < batch; b++)
    {
        const float16_t *x = data_in + (size_t)b * (size_t)batch_offset * (size_t)input_size;
        const float16_t *h_prev =
            hidden_in ? (hidden_in + (size_t)b * (size_t)batch_offset * (size_t)hidden_size) : NULL;
        float16_t *c_prev = cell_state + (size_t)b * (size_t)hidden_size;
        float16_t *h_out = hidden_out + (size_t)b * (size_t)batch_offset * (size_t)hidden_size;

        int32_t h = 0;
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        for (; h + 8 <= hidden_size; h += 8)
        {
            float16_t f_gate[8];
            float16_t i_gate[8];
            float16_t g_gate[8];
            float16_t o_gate[8];

            for (int32_t lane = 0; lane < 8; ++lane)
            {
                const int32_t idx = h + lane;
                f_gate[lane] =
                    arm_nn_lstm_gate_compute_f16(&params->forget_gate, x, h_prev, input_size, hidden_size, idx);
                i_gate[lane] =
                    arm_nn_lstm_gate_compute_f16(&params->input_gate, x, h_prev, input_size, hidden_size, idx);
                g_gate[lane] =
                    arm_nn_lstm_gate_compute_f16(&params->cell_gate, x, h_prev, input_size, hidden_size, idx);
                o_gate[lane] =
                    arm_nn_lstm_gate_compute_f16(&params->output_gate, x, h_prev, input_size, hidden_size, idx);
            }

            const float16x8_t vf = vld1q(f_gate);
            const float16x8_t vi = vld1q(i_gate);
            const float16x8_t vg = vld1q(g_gate);
            const float16x8_t vo = vld1q(o_gate);
            const float16x8_t vc_prev = vld1q(c_prev + h);

            float16x8_t vc = vfmaq(vmulq(vf, vc_prev), vi, vg);
            if (cell_clip > (_Float16)0.0f)
            {
                const _Float16 clip = (_Float16)cell_clip;
                vc = vmaxnmq(vc, vdupq_n_f16((float16_t)-clip));
                vc = vminnmq(vc, vdupq_n_f16(clip));
            }
            vst1q(c_prev + h, vc);
            vst1q(h_out + h, vmulq(vo, arm_nn_vtanh_lut_direct_mve_f16(vc)));
        }
#endif
        for (; h < hidden_size; h++)
        {
            const float16_t f =
                arm_nn_lstm_gate_compute_f16(&params->forget_gate, x, h_prev, input_size, hidden_size, h);
            const float16_t i =
                arm_nn_lstm_gate_compute_f16(&params->input_gate, x, h_prev, input_size, hidden_size, h);
            const float16_t g = arm_nn_lstm_gate_compute_f16(&params->cell_gate, x, h_prev, input_size, hidden_size, h);
            const float16_t o =
                arm_nn_lstm_gate_compute_f16(&params->output_gate, x, h_prev, input_size, hidden_size, h);

            _Float16 c = (_Float16)((_Float16)f * (_Float16)c_prev[h] + (_Float16)i * (_Float16)g);
            if (cell_clip > (_Float16)0.0f)
            {
                const _Float16 clip = (_Float16)cell_clip;
                c = CLAMP(c, clip, -clip);
            }
            c_prev[h] = (float16_t)c;
            h_out[h] = (float16_t)((_Float16)o * (_Float16)arm_nn_tanh_scalar_ref_f16((float16_t)c));
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/** @} end of supportLSTM group */
