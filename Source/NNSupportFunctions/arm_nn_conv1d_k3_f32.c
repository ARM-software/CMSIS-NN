/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv1d_k3_f32.c
 * Description:  Support: NHWC 1D convolution kernel size 3 for f32
 *
 * $Date:        30 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_NN_CONV1D_K3_NHWC_F32_OC2_BLOCK (2)
    #endif

void arm_nn_conv1d_k3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float32_t *__RESTRICT kernel,
                               const float32_t *__RESTRICT b,
                               float32_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w)
{
    (void)in_w;

    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float32_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float32_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float32_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        float32_t *y = out + (size_t)ow * (size_t)out_c;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        int32_t oc = 0;

        for (; oc + ARM_NN_CONV1D_K3_NHWC_F32_OC2_BLOCK <= out_c; oc += ARM_NN_CONV1D_K3_NHWC_F32_OC2_BLOCK)
        {
            const float32_t *w_base0 = kernel + (size_t)(oc + 0) * 3U * (size_t)in_c;
            const float32_t *w_base1 = kernel + (size_t)(oc + 1) * 3U * (size_t)in_c;
            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);

            for (int32_t ic = 0; ic < in_c; ic += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(in_c - ic));
                float32x4_t vx = vld1q_z(x0 + ic, p);
                vacc0 = vfmaq_m(vacc0, vx, vld1q_z(w_base0 + 0 * in_c + ic, p), p);
                vacc1 = vfmaq_m(vacc1, vx, vld1q_z(w_base1 + 0 * in_c + ic, p), p);

                vx = vld1q_z(x1 + ic, p);
                vacc0 = vfmaq_m(vacc0, vx, vld1q_z(w_base0 + 1 * in_c + ic, p), p);
                vacc1 = vfmaq_m(vacc1, vx, vld1q_z(w_base1 + 1 * in_c + ic, p), p);

                vx = vld1q_z(x2 + ic, p);
                vacc0 = vfmaq_m(vacc0, vx, vld1q_z(w_base0 + 2 * in_c + ic, p), p);
                vacc1 = vfmaq_m(vacc1, vx, vld1q_z(w_base1 + 2 * in_c + ic, p), p);
            }

            y[oc + 0] = (b ? b[oc + 0] : 0.0f) + arm_nn_vec_reduce_add_f32(vacc0);
            y[oc + 1] = (b ? b[oc + 1] : 0.0f) + arm_nn_vec_reduce_add_f32(vacc1);
        }

        for (; oc < out_c; ++oc)
        {
            const float32_t *w0 = kernel + (size_t)oc * 3U * (size_t)in_c;
            const float32_t *w1 = w0 + in_c;
            const float32_t *w2 = w1 + in_c;
            float32x4_t vacc = vdupq_n_f32(0.0f);

            for (int32_t ic = 0; ic < in_c; ic += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(in_c - ic));
                vacc = vfmaq_m(vacc, vld1q_z(x0 + ic, p), vld1q_z(w0 + ic, p), p);
                vacc = vfmaq_m(vacc, vld1q_z(x1 + ic, p), vld1q_z(w1 + ic, p), p);
                vacc = vfmaq_m(vacc, vld1q_z(x2 + ic, p), vld1q_z(w2 + ic, p), p);
            }

            y[oc] = (b ? b[oc] : 0.0f) + arm_nn_vec_reduce_add_f32(vacc);
        }
    #else
        for (int32_t oc = 0; oc < out_c; ++oc)
        {
            const float32_t *w = kernel + (size_t)oc * 3U * (size_t)in_c;
            float32_t acc = b ? b[oc] : 0.0f;
            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                acc += x0[ic] * w[ic];
                acc += x1[ic] * w[in_c + ic];
                acc += x2[ic] * w[2 * in_c + ic];
            }
            y[oc] = acc;
        }
    #endif
    }
}

#endif /* ARM_NN_ENABLE_F32 */
