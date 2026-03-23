/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_depthwise_conv2x5_f16.c
 * Description:  Support: NHWC depthwise 2x5 convolution kernel for f16
 *
 * $Date:        30 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

void arm_nn_depthwise_conv2x5_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                       int32_t batches,
                                       int32_t in_c,
                                       int32_t in_w,
                                       int32_t ch_mult,
                                       const float16_t *__RESTRICT kernel,
                                       const float16_t *__RESTRICT b,
                                       float16_t *__RESTRICT out,
                                       int32_t out_w,
                                       float16_t act_min,
                                       float16_t act_max)
{
    const int32_t out_c = in_c * ch_mult;
    const size_t in_batch_stride = (size_t)2U * (size_t)in_w * (size_t)in_c;
    const size_t out_batch_stride = (size_t)out_w * (size_t)out_c;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float16x8_t v_act_min = vdupq_n_f16(act_min);
    const float16x8_t v_act_max = vdupq_n_f16(act_max);
    #endif

    for (int32_t batch = 0; batch < batches; ++batch)
    {
        const float16_t *in_batch = x_nhwc + (size_t)batch * in_batch_stride;
        float16_t *out_batch = out + (size_t)batch * out_batch_stride;

        for (int32_t ow = 0; ow < out_w; ++ow)
        {
            const float16_t *row0 = in_batch + (size_t)ow * (size_t)in_c;
            const float16_t *row1 = in_batch + ((size_t)in_w + (size_t)ow) * (size_t)in_c;

            for (int32_t c = 0; c < in_c; ++c)
            {
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                const float16_t x00 = row0[(size_t)0 * (size_t)in_c + (size_t)c];
                const float16_t x01 = row0[(size_t)1 * (size_t)in_c + (size_t)c];
                const float16_t x02 = row0[(size_t)2 * (size_t)in_c + (size_t)c];
                const float16_t x03 = row0[(size_t)3 * (size_t)in_c + (size_t)c];
                const float16_t x04 = row0[(size_t)4 * (size_t)in_c + (size_t)c];
                const float16_t x10 = row1[(size_t)0 * (size_t)in_c + (size_t)c];
                const float16_t x11 = row1[(size_t)1 * (size_t)in_c + (size_t)c];
                const float16_t x12 = row1[(size_t)2 * (size_t)in_c + (size_t)c];
                const float16_t x13 = row1[(size_t)3 * (size_t)in_c + (size_t)c];
                const float16_t x14 = row1[(size_t)4 * (size_t)in_c + (size_t)c];

                for (int32_t m = 0; m < ch_mult; m += 8)
                {
                    const mve_pred16_t p = vctp16q((uint32_t)(ch_mult - m));
                    const int32_t oc = c * ch_mult + m;
                    float16x8_t vacc = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);

                    vacc = vfmaq(vacc, vdupq_n_f16(x00), vld1q_z(kernel + (size_t)0 * (size_t)out_c + (size_t)oc, p));
                    vacc = vfmaq(vacc, vdupq_n_f16(x01), vld1q_z(kernel + (size_t)1 * (size_t)out_c + (size_t)oc, p));
                    vacc = vfmaq(vacc, vdupq_n_f16(x02), vld1q_z(kernel + (size_t)2 * (size_t)out_c + (size_t)oc, p));
                    vacc = vfmaq(vacc, vdupq_n_f16(x03), vld1q_z(kernel + (size_t)3 * (size_t)out_c + (size_t)oc, p));
                    vacc = vfmaq(vacc, vdupq_n_f16(x04), vld1q_z(kernel + (size_t)4 * (size_t)out_c + (size_t)oc, p));
                    vacc = vfmaq(vacc, vdupq_n_f16(x10), vld1q_z(kernel + (size_t)5 * (size_t)out_c + (size_t)oc, p));
                    vacc = vfmaq(vacc, vdupq_n_f16(x11), vld1q_z(kernel + (size_t)6 * (size_t)out_c + (size_t)oc, p));
                    vacc = vfmaq(vacc, vdupq_n_f16(x12), vld1q_z(kernel + (size_t)7 * (size_t)out_c + (size_t)oc, p));
                    vacc = vfmaq(vacc, vdupq_n_f16(x13), vld1q_z(kernel + (size_t)8 * (size_t)out_c + (size_t)oc, p));
                    vacc = vfmaq(vacc, vdupq_n_f16(x14), vld1q_z(kernel + (size_t)9 * (size_t)out_c + (size_t)oc, p));

                    vacc = vmaxnmq(vacc, v_act_min);
                    vacc = vminnmq(vacc, v_act_max);
                    vst1q_p(out_batch + (size_t)ow * (size_t)out_c + (size_t)oc, vacc, p);
                }
    #else
                for (int32_t m = 0; m < ch_mult; ++m)
                {
                    const int32_t oc = c * ch_mult + m;
                    float32_t acc = b ? (float32_t)b[oc] : 0.0f;
                    for (int32_t kx = 0; kx < 5; ++kx)
                    {
                        const size_t base0 = (size_t)kx * (size_t)out_c + (size_t)oc;
                        const size_t base1 = ((size_t)5U + (size_t)kx) * (size_t)out_c + (size_t)oc;
                        acc += (float32_t)row0[(size_t)kx * (size_t)in_c + (size_t)c] * (float32_t)kernel[base0];
                        acc += (float32_t)row1[(size_t)kx * (size_t)in_c + (size_t)c] * (float32_t)kernel[base1];
                    }
                    acc = MAX(acc, (float32_t)act_min);
                    acc = MIN(acc, (float32_t)act_max);
                    out_batch[(size_t)ow * (size_t)out_c + (size_t)oc] = (float16_t)acc;
                }
    #endif
            }
        }
    }
}

#endif /* ARM_NN_ENABLE_F16 */
