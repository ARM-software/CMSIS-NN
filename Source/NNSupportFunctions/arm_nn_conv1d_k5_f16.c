/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv1d_k5_f16.c
 * Description:  Support: NHWC 1D convolution kernel size 5 for f16
 *
 * $Date:        31 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        #define ARM_NN_CONV1D_K5_NHWC_F16_OC4_BLOCK (4)
        #define ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK (16)
    #endif

void arm_nn_conv1d_k5_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const float16_t *__RESTRICT kernel,
                               const float16_t *__RESTRICT b,
                               float16_t *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w)
{
    (void)in_w;

    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        float16_t *y = out + (size_t)ow * (size_t)out_c;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
        int32_t oc = 0;

        if (in_c == 16 && b != NULL)
        {
            const float16x8_t vx00 = vld1q(x0 + 0);
            const float16x8_t vx01 = vld1q(x1 + 0);
            const float16x8_t vx02 = vld1q(x2 + 0);
            const float16x8_t vx03 = vld1q(x3 + 0);
            const float16x8_t vx04 = vld1q(x4 + 0);
            const float16x8_t vx10 = vld1q(x0 + 8);
            const float16x8_t vx11 = vld1q(x1 + 8);
            const float16x8_t vx12 = vld1q(x2 + 8);
            const float16x8_t vx13 = vld1q(x3 + 8);
            const float16x8_t vx14 = vld1q(x4 + 8);

            for (; oc + ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK <= out_c; oc += ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK)
            {
                float16x8_t partials[ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK];

                for (int32_t of = 0; of < ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK; ++of)
                {
                    const float16_t *w_base = kernel + (size_t)(oc + of) * 5U * (size_t)in_c;
                    float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);

                    vacc = vfmaq(vacc, vx00, vld1q(w_base + 0 * in_c + 0));
                    vacc = vfmaq(vacc, vx01, vld1q(w_base + 1 * in_c + 0));
                    vacc = vfmaq(vacc, vx02, vld1q(w_base + 2 * in_c + 0));
                    vacc = vfmaq(vacc, vx03, vld1q(w_base + 3 * in_c + 0));
                    vacc = vfmaq(vacc, vx04, vld1q(w_base + 4 * in_c + 0));
                    partials[of] = vacc;
                }

                for (int32_t of = 0; of < ARM_NN_CONV1D_K5_NHWC_F16_OC16_BLOCK; ++of)
                {
                    const float16_t *w_base = kernel + (size_t)(oc + of) * 5U * (size_t)in_c + 8;
                    float16x8_t vacc = partials[of];

                    vacc = vfmaq(vacc, vx10, vld1q(w_base + 0 * in_c));
                    vacc = vfmaq(vacc, vx11, vld1q(w_base + 1 * in_c));
                    vacc = vfmaq(vacc, vx12, vld1q(w_base + 2 * in_c));
                    vacc = vfmaq(vacc, vx13, vld1q(w_base + 3 * in_c));
                    vacc = vfmaq(vacc, vx14, vld1q(w_base + 4 * in_c));

                    y[oc + of] = (float16_t)((_Float16)b[oc + of] + (_Float16)arm_nn_vec_reduce_add_f16(vacc));
                }
            }
        }

        for (; oc + ARM_NN_CONV1D_K5_NHWC_F16_OC4_BLOCK <= out_c; oc += ARM_NN_CONV1D_K5_NHWC_F16_OC4_BLOCK)
        {
            const float16_t *w_base0 = kernel + (size_t)(oc + 0) * 5U * (size_t)in_c;
            const float16_t *w_base1 = kernel + (size_t)(oc + 1) * 5U * (size_t)in_c;
            const float16_t *w_base2 = kernel + (size_t)(oc + 2) * 5U * (size_t)in_c;
            const float16_t *w_base3 = kernel + (size_t)(oc + 3) * 5U * (size_t)in_c;
            float16x8_t vacc0 = vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc2 = vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc3 = vdupq_n_f16((float16_t)0.0f);
            for (int32_t ic = 0; ic < in_c; ic += 8)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(in_c - ic));
                const float16x8_t vx0 = vld1q_z(x0 + ic, p);
                const float16x8_t vx1 = vld1q_z(x1 + ic, p);
                const float16x8_t vx2 = vld1q_z(x2 + ic, p);
                const float16x8_t vx3 = vld1q_z(x3 + ic, p);
                const float16x8_t vx4 = vld1q_z(x4 + ic, p);

                vacc0 = vfmaq(vacc0, vx0, vld1q_z(w_base0 + 0 * in_c + ic, p));
                vacc0 = vfmaq(vacc0, vx1, vld1q_z(w_base0 + 1 * in_c + ic, p));
                vacc0 = vfmaq(vacc0, vx2, vld1q_z(w_base0 + 2 * in_c + ic, p));
                vacc0 = vfmaq(vacc0, vx3, vld1q_z(w_base0 + 3 * in_c + ic, p));
                vacc0 = vfmaq(vacc0, vx4, vld1q_z(w_base0 + 4 * in_c + ic, p));

                vacc1 = vfmaq(vacc1, vx0, vld1q_z(w_base1 + 0 * in_c + ic, p));
                vacc1 = vfmaq(vacc1, vx1, vld1q_z(w_base1 + 1 * in_c + ic, p));
                vacc1 = vfmaq(vacc1, vx2, vld1q_z(w_base1 + 2 * in_c + ic, p));
                vacc1 = vfmaq(vacc1, vx3, vld1q_z(w_base1 + 3 * in_c + ic, p));
                vacc1 = vfmaq(vacc1, vx4, vld1q_z(w_base1 + 4 * in_c + ic, p));

                vacc2 = vfmaq(vacc2, vx0, vld1q_z(w_base2 + 0 * in_c + ic, p));
                vacc2 = vfmaq(vacc2, vx1, vld1q_z(w_base2 + 1 * in_c + ic, p));
                vacc2 = vfmaq(vacc2, vx2, vld1q_z(w_base2 + 2 * in_c + ic, p));
                vacc2 = vfmaq(vacc2, vx3, vld1q_z(w_base2 + 3 * in_c + ic, p));
                vacc2 = vfmaq(vacc2, vx4, vld1q_z(w_base2 + 4 * in_c + ic, p));

                vacc3 = vfmaq(vacc3, vx0, vld1q_z(w_base3 + 0 * in_c + ic, p));
                vacc3 = vfmaq(vacc3, vx1, vld1q_z(w_base3 + 1 * in_c + ic, p));
                vacc3 = vfmaq(vacc3, vx2, vld1q_z(w_base3 + 2 * in_c + ic, p));
                vacc3 = vfmaq(vacc3, vx3, vld1q_z(w_base3 + 3 * in_c + ic, p));
                vacc3 = vfmaq(vacc3, vx4, vld1q_z(w_base3 + 4 * in_c + ic, p));
            }

            const _Float16 acc0 =
                (b ? (_Float16)b[oc + 0] : (_Float16)0.0f) + (_Float16)arm_nn_vec_reduce_add_f16(vacc0);
            const _Float16 acc1 =
                (b ? (_Float16)b[oc + 1] : (_Float16)0.0f) + (_Float16)arm_nn_vec_reduce_add_f16(vacc1);
            const _Float16 acc2 =
                (b ? (_Float16)b[oc + 2] : (_Float16)0.0f) + (_Float16)arm_nn_vec_reduce_add_f16(vacc2);
            const _Float16 acc3 =
                (b ? (_Float16)b[oc + 3] : (_Float16)0.0f) + (_Float16)arm_nn_vec_reduce_add_f16(vacc3);

            y[oc + 0] = (float16_t)acc0;
            y[oc + 1] = (float16_t)acc1;
            y[oc + 2] = (float16_t)acc2;
            y[oc + 3] = (float16_t)acc3;
        }

        for (; oc < out_c; ++oc)
        {
            const float16_t *w0 = kernel + (size_t)oc * 5U * (size_t)in_c;
            const float16_t *w1 = w0 + in_c;
            const float16_t *w2 = w1 + in_c;
            const float16_t *w3 = w2 + in_c;
            const float16_t *w4 = w3 + in_c;
            _Float16 acc = b ? (_Float16)b[oc] : (_Float16)0.0f;
            float16x8_t vacc = vdupq_n_f16((float16_t)0.0f);
            for (int32_t ic = 0; ic < in_c; ic += 8)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(in_c - ic));
                vacc = vfmaq(vacc, vld1q_z(x0 + ic, p), vld1q_z(w0 + ic, p));
                vacc = vfmaq(vacc, vld1q_z(x1 + ic, p), vld1q_z(w1 + ic, p));
                vacc = vfmaq(vacc, vld1q_z(x2 + ic, p), vld1q_z(w2 + ic, p));
                vacc = vfmaq(vacc, vld1q_z(x3 + ic, p), vld1q_z(w3 + ic, p));
                vacc = vfmaq(vacc, vld1q_z(x4 + ic, p), vld1q_z(w4 + ic, p));
            }
            acc += (_Float16)arm_nn_vec_reduce_add_f16(vacc);
            y[oc] = (float16_t)acc;
        }
    #else
        for (int32_t oc = 0; oc < out_c; ++oc)
        {
            const float16_t *w = kernel + (size_t)oc * 5U * (size_t)in_c;
            float32_t acc = b ? (float32_t)b[oc] : 0.0f;
            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                acc += (float32_t)x0[ic] * (float32_t)w[ic];
                acc += (float32_t)x1[ic] * (float32_t)w[in_c + ic];
                acc += (float32_t)x2[ic] * (float32_t)w[2 * in_c + ic];
                acc += (float32_t)x3[ic] * (float32_t)w[3 * in_c + ic];
                acc += (float32_t)x4[ic] * (float32_t)w[4 * in_c + ic];
            }
            y[oc] = (float16_t)acc;
        }
    #endif
    }
}

#endif /* ARM_NN_ENABLE_F16 */
