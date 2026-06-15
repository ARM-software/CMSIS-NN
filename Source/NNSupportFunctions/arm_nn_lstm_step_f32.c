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
 * Title:        arm_nn_lstm_step_f32.c
 * Description:  Update LSTM function for a single iteration step (float32)
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
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
 * Calculate the output state tensor of an LSTM step, float32 version.
 * Refer to header file for details.
 */

__STATIC_INLINE float32_t arm_nn_lstm_activate_gate_f32(float32_t x, arm_nn_activation_type_flt type)
{
    if (type == ARM_NN_FLT_ACT_TANH)
    {
        return arm_nn_tanh_scalar_ref_f32(x);
    }
    return arm_nn_sigmoid_scalar_f32(x);
}

__STATIC_INLINE float32_t arm_nn_lstm_dot_f32(const float32_t *lhs, const float32_t *rhs, int32_t count)
{
    float32_t acc = 0.0f;

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    float32x4_t vacc = vdupq_n_f32(0.0f);

    for (int32_t i = 0; i < count; i += 4)
    {
        const mve_pred16_t p = vctp32q((uint32_t)(count - i));
        vacc = vfmaq(vacc, vld1q_z(lhs + i, p), vld1q_z(rhs + i, p));
    }

    acc += arm_nn_vec_reduce_add_f32(vacc);
#else
    for (int32_t i = 0; i < count; ++i)
    {
        acc += lhs[i] * rhs[i];
    }
#endif

    return acc;
}

static float32_t arm_nn_lstm_gate_compute_f32(const cmsis_nn_lstm_gate_f32 *gate,
                                              const float32_t *input,
                                              const float32_t *hidden,
                                              int32_t input_size,
                                              int32_t hidden_size,
                                              int32_t h_idx)
{
    float32_t acc = 0.0f;
    if (gate->bias)
    {
        acc = gate->bias[h_idx];
    }
    if (gate->input_weights)
    {
        const float32_t *w_in = gate->input_weights + (size_t)h_idx * (size_t)input_size;
        acc += arm_nn_lstm_dot_f32(input, w_in, input_size);
    }
    if (hidden && gate->hidden_weights)
    {
        const float32_t *w_h = gate->hidden_weights + (size_t)h_idx * (size_t)hidden_size;
        acc += arm_nn_lstm_dot_f32(hidden, w_h, hidden_size);
    }
    return arm_nn_lstm_activate_gate_f32(acc, gate->activation_type);
}

arm_cmsis_nn_status arm_nn_lstm_step_f32(const float32_t *data_in,
                                         const float32_t *hidden_in,
                                         float32_t *hidden_out,
                                         const cmsis_nn_lstm_params_f32 *params,
                                         cmsis_nn_lstm_context_f32 *buffers,
                                         const int32_t batch_offset)
{
    if (!data_in || !hidden_out || !params || !buffers || !buffers->cell_state || batch_offset <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = params->batch_size;
    const int32_t input_size = params->input_size;
    const int32_t hidden_size = params->hidden_size;
    const float32_t cell_clip = params->cell_clip;

    float32_t *cell_state = buffers->cell_state;

    for (int32_t b = 0; b < batch; b++)
    {
        const float32_t *x = data_in + (size_t)b * (size_t)batch_offset * (size_t)input_size;
        const float32_t *h_prev =
            hidden_in ? (hidden_in + (size_t)b * (size_t)batch_offset * (size_t)hidden_size) : NULL;
        float32_t *c_prev = cell_state + (size_t)b * (size_t)hidden_size;
        float32_t *h_out = hidden_out + (size_t)b * (size_t)batch_offset * (size_t)hidden_size;

        int32_t h = 0;
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        for (; h + 4 <= hidden_size; h += 4)
        {
            float32_t f_gate[4];
            float32_t i_gate[4];
            float32_t g_gate[4];
            float32_t o_gate[4];

            for (int32_t lane = 0; lane < 4; ++lane)
            {
                const int32_t idx = h + lane;
                f_gate[lane] =
                    arm_nn_lstm_gate_compute_f32(&params->forget_gate, x, h_prev, input_size, hidden_size, idx);
                i_gate[lane] =
                    arm_nn_lstm_gate_compute_f32(&params->input_gate, x, h_prev, input_size, hidden_size, idx);
                g_gate[lane] =
                    arm_nn_lstm_gate_compute_f32(&params->cell_gate, x, h_prev, input_size, hidden_size, idx);
                o_gate[lane] =
                    arm_nn_lstm_gate_compute_f32(&params->output_gate, x, h_prev, input_size, hidden_size, idx);
            }

            const float32x4_t vf = vld1q(f_gate);
            const float32x4_t vi = vld1q(i_gate);
            const float32x4_t vg = vld1q(g_gate);
            const float32x4_t vo = vld1q(o_gate);
            const float32x4_t vc_prev = vld1q(c_prev + h);

            float32x4_t vc = vfmaq(vmulq(vf, vc_prev), vi, vg);
            if (cell_clip > 0.0f)
            {
                const float32x4_t v_clip = vdupq_n_f32(cell_clip);
                vc = vmaxnmq(vc, vnegq(v_clip));
                vc = vminnmq(vc, v_clip);
            }
            vst1q(c_prev + h, vc);
            vst1q(h_out + h, vmulq(vo, arm_nn_vtanh_lut_direct_mve_f32(vc)));
        }
#endif
        for (; h < hidden_size; h++)
        {
            const float32_t f =
                arm_nn_lstm_gate_compute_f32(&params->forget_gate, x, h_prev, input_size, hidden_size, h);
            const float32_t i =
                arm_nn_lstm_gate_compute_f32(&params->input_gate, x, h_prev, input_size, hidden_size, h);
            const float32_t g = arm_nn_lstm_gate_compute_f32(&params->cell_gate, x, h_prev, input_size, hidden_size, h);
            const float32_t o =
                arm_nn_lstm_gate_compute_f32(&params->output_gate, x, h_prev, input_size, hidden_size, h);

            float32_t c = f * c_prev[h] + i * g;
            if (cell_clip > 0.0f)
            {
                c = CLAMP(c, cell_clip, -cell_clip);
            }
            c_prev[h] = c;
            h_out[h] = o * arm_nn_tanh_scalar_ref_f32(c);
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/** @} end of supportLSTM group */
