/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_depthwise_conv1d_k3_f32.c
 * Description:  Support: NHWC depthwise 1D convolution kernel size 3 for f32
 *
 * $Date:        30 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

void arm_nn_depthwise_conv1d_k3_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                         int32_t in_c,
                                         int32_t in_w,
                                         const float32_t *__RESTRICT kernel,
                                         const float32_t *__RESTRICT b,
                                         float32_t *__RESTRICT out,
                                         int32_t out_w)
{
    (void)in_w;

    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float32_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float32_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float32_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        float32_t *dst = out + (size_t)ow * (size_t)in_c;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
        for (int32_t c = 0; c < in_c; c += 4)
        {
            const mve_pred16_t p = vctp32q((uint32_t)(in_c - c));
            float32x4_t acc = b ? vld1q_z(b + c, p) : vdupq_n_f32(0.0f);
            acc = vfmaq_m(acc, vld1q_z(x0 + c, p), vld1q_z(kernel + c, p), p);
            acc = vfmaq_m(acc, vld1q_z(x1 + c, p), vld1q_z(kernel + in_c + c, p), p);
            acc = vfmaq_m(acc, vld1q_z(x2 + c, p), vld1q_z(kernel + (2 * in_c) + c, p), p);
            vst1q_p(dst + c, acc, p);
        }
    #else
        for (int32_t c = 0; c < in_c; ++c)
        {
            float32_t acc = b ? b[c] : 0.0f;
            acc += x0[c] * kernel[c];
            acc += x1[c] * kernel[in_c + c];
            acc += x2[c] * kernel[2 * in_c + c];
            dst[c] = acc;
        }
    #endif
    }
}

#endif /* ARM_NN_ENABLE_F32 */
