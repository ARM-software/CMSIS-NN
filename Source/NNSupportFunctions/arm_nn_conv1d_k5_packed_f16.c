/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv1d_k5_packed_f16.c
 * Description:  Support: NHWC 1D convolution kernel size 5 for packed f16 weights
 *
 * $Date:        29 Apr 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

void arm_nn_conv1d_k5_packed_f16(const float16_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float16_t *__RESTRICT kernel_packed,
                                 const float16_t *__RESTRICT b,
                                 float16_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w)
{
    (void)in_w;

    const int32_t block_cols = 8;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    int32_t ow = 0;
    for (; ow + 1 < out_w; ow += 2)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        const float16_t *x5 = x_nhwc + (size_t)(ow + 5) * (size_t)in_c;
        float16_t *y0 = out + (size_t)(ow + 0) * (size_t)out_c;
        float16_t *y1 = out + (size_t)(ow + 1) * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 5U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                const float16x8_t vw0 = vld1q(w_ic + 0U * in_c * block_cols);
                const float16x8_t vw1 = vld1q(w_ic + 1U * in_c * block_cols);
                const float16x8_t vw2 = vld1q(w_ic + 2U * in_c * block_cols);
                const float16x8_t vw3 = vld1q(w_ic + 3U * in_c * block_cols);
                const float16x8_t vw4 = vld1q(w_ic + 4U * in_c * block_cols);

                vacc0 = vfmaq(vacc0, vw0, x0[ic]);
                vacc1 = vfmaq(vacc1, vw0, x1[ic]);
                vacc0 = vfmaq(vacc0, vw1, x1[ic]);
                vacc1 = vfmaq(vacc1, vw1, x2[ic]);
                vacc0 = vfmaq(vacc0, vw2, x2[ic]);
                vacc1 = vfmaq(vacc1, vw2, x3[ic]);
                vacc0 = vfmaq(vacc0, vw3, x3[ic]);
                vacc1 = vfmaq(vacc1, vw3, x4[ic]);
                vacc0 = vfmaq(vacc0, vw4, x4[ic]);
                vacc1 = vfmaq(vacc1, vw4, x5[ic]);
            }

            vst1q(y0 + oc, vacc0);
            vst1q(y1 + oc, vacc1);
        }

        if (oc < out_c)
        {
            const int32_t valid_cols = out_c - oc;
            const mve_pred16_t p = vctp16q((uint32_t)valid_cols);
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 5U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                const float16x8_t vw0 = vld1q_z(w_ic + 0U * in_c * block_cols, p);
                const float16x8_t vw1 = vld1q_z(w_ic + 1U * in_c * block_cols, p);
                const float16x8_t vw2 = vld1q_z(w_ic + 2U * in_c * block_cols, p);
                const float16x8_t vw3 = vld1q_z(w_ic + 3U * in_c * block_cols, p);
                const float16x8_t vw4 = vld1q_z(w_ic + 4U * in_c * block_cols, p);

                vacc0 = vfmaq(vacc0, vw0, x0[ic]);
                vacc1 = vfmaq(vacc1, vw0, x1[ic]);
                vacc0 = vfmaq(vacc0, vw1, x1[ic]);
                vacc1 = vfmaq(vacc1, vw1, x2[ic]);
                vacc0 = vfmaq(vacc0, vw2, x2[ic]);
                vacc1 = vfmaq(vacc1, vw2, x3[ic]);
                vacc0 = vfmaq(vacc0, vw3, x3[ic]);
                vacc1 = vfmaq(vacc1, vw3, x4[ic]);
                vacc0 = vfmaq(vacc0, vw4, x4[ic]);
                vacc1 = vfmaq(vacc1, vw4, x5[ic]);
            }

            vst1q_p(y0 + oc, vacc0, p);
            vst1q_p(y1 + oc, vacc1, p);
        }
    }

    if (ow < out_w)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        float16_t *y = out + (size_t)ow * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 5U * (size_t)in_c * block_cols;
            float16x8_t vacc = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                vacc = vfmaq(vacc, vld1q(w_ic + 0U * in_c * block_cols), x0[ic]);
                vacc = vfmaq(vacc, vld1q(w_ic + 1U * in_c * block_cols), x1[ic]);
                vacc = vfmaq(vacc, vld1q(w_ic + 2U * in_c * block_cols), x2[ic]);
                vacc = vfmaq(vacc, vld1q(w_ic + 3U * in_c * block_cols), x3[ic]);
                vacc = vfmaq(vacc, vld1q(w_ic + 4U * in_c * block_cols), x4[ic]);
            }

            vst1q(y + oc, vacc);
        }

        if (oc < out_c)
        {
            const int32_t valid_cols = out_c - oc;
            const mve_pred16_t p = vctp16q((uint32_t)valid_cols);
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 5U * (size_t)in_c * block_cols;
            float16x8_t vacc = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                vacc = vfmaq(vacc, vld1q_z(w_ic + 0U * in_c * block_cols, p), x0[ic]);
                vacc = vfmaq(vacc, vld1q_z(w_ic + 1U * in_c * block_cols, p), x1[ic]);
                vacc = vfmaq(vacc, vld1q_z(w_ic + 2U * in_c * block_cols, p), x2[ic]);
                vacc = vfmaq(vacc, vld1q_z(w_ic + 3U * in_c * block_cols, p), x3[ic]);
                vacc = vfmaq(vacc, vld1q_z(w_ic + 4U * in_c * block_cols, p), x4[ic]);
            }

            vst1q_p(y + oc, vacc, p);
        }
    }

    #else
    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        float16_t *y = out + (size_t)ow * (size_t)out_c;

        for (int32_t oc = 0; oc < out_c; ++oc)
        {
            const int32_t lane = oc % block_cols;
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 5U * (size_t)in_c * block_cols;
            _Float16 acc = b ? (_Float16)b[oc] : (_Float16)0.0f;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                acc += (_Float16)x0[ic] * (_Float16)w_ic[(size_t)0 * in_c * block_cols + lane];
                acc += (_Float16)x1[ic] * (_Float16)w_ic[(size_t)1 * in_c * block_cols + lane];
                acc += (_Float16)x2[ic] * (_Float16)w_ic[(size_t)2 * in_c * block_cols + lane];
                acc += (_Float16)x3[ic] * (_Float16)w_ic[(size_t)3 * in_c * block_cols + lane];
                acc += (_Float16)x4[ic] * (_Float16)w_ic[(size_t)4 * in_c * block_cols + lane];
            }

            y[oc] = (float16_t)acc;
        }
    }
    #endif
}

#endif /* ARM_NN_ENABLE_F16 */
