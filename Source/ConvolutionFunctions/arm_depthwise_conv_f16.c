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
 * Title:        arm_depthwise_conv_f16.c
 * Description:  Convolution: depthwise (float16)
 *
 * $Date:        23 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_depthwise_conv_opt_common.h"
#include "Internal/arm_depthwise_conv_opt_f16.h"
#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/* Number of packed output rows processed per depthwise NT-T tile. */
#define ARM_NN_DW_NT_T_F16_TILE_ROWS (4)

__STATIC_INLINE void
arm_depthwise_accumulate_vec_f16(float16_t *acc, const float16_t *input, const float16_t *kernel, int32_t channels)
{
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t c = 0; c < channels; c += 8)
    {
        const mve_pred16_t p = vctp16q((uint32_t)(channels - c));
        float16x8_t vacc = vld1q_z(acc + c, p);
        vacc = vfmaq(vacc, vld1q_z(input + c, p), vld1q_z(kernel + c, p));
        vst1q_p(acc + c, vacc, p);
    }
#else
    for (int32_t c = 0; c < channels; ++c)
    {
        const _Float16 sum = (_Float16)acc[c] + (_Float16)input[c] * (_Float16)kernel[c];
        acc[c] = (float16_t)sum;
    }
#endif
}

static void arm_depthwise_conv_nhwc_fast_chmult1_kc_f16(const float16_t *input,
                                                        int32_t input_batches,
                                                        int32_t input_x,
                                                        int32_t input_y,
                                                        int32_t input_ch,
                                                        const float16_t *kernel,
                                                        int32_t kernel_x,
                                                        int32_t kernel_y,
                                                        int32_t pad_x,
                                                        int32_t pad_y,
                                                        int32_t stride_x,
                                                        int32_t stride_y,
                                                        const float16_t *bias,
                                                        float16_t *output,
                                                        int32_t output_x,
                                                        int32_t output_y,
                                                        float16_t output_activation_min,
                                                        float16_t output_activation_max)
{
    const int32_t in_batch_stride = input_x * input_y * input_ch;
    const int32_t out_batch_stride = output_x * output_y * input_ch;

    for (int32_t i_batch = 0; i_batch < input_batches; ++i_batch)
    {
        const float16_t *input_b = input + i_batch * in_batch_stride;
        float16_t *output_b = output + i_batch * out_batch_stride;

        for (int32_t i_out_y = 0; i_out_y < output_y; ++i_out_y)
        {
            const int32_t base_idx_y = (i_out_y * stride_y) - pad_y;
            const int32_t ker_y_start = (base_idx_y < 0) ? -base_idx_y : 0;
            const int32_t ker_y_end = (kernel_y < (input_y - base_idx_y)) ? kernel_y : (input_y - base_idx_y);

            for (int32_t i_out_x = 0; i_out_x < output_x; ++i_out_x)
            {
                const int32_t base_idx_x = (i_out_x * stride_x) - pad_x;
                const int32_t ker_x_start = (base_idx_x < 0) ? -base_idx_x : 0;
                const int32_t ker_x_end = (kernel_x < (input_x - base_idx_x)) ? kernel_x : (input_x - base_idx_x);

                float16_t *out_px = output_b + ((size_t)i_out_y * output_x + i_out_x) * input_ch;
                for (int32_t c = 0; c < input_ch; ++c)
                {
                    out_px[c] = bias ? bias[c] : (float16_t)0;
                }

                for (int32_t i_ker_y = ker_y_start; i_ker_y < ker_y_end; ++i_ker_y)
                {
                    const int32_t idx_y = base_idx_y + i_ker_y;
                    for (int32_t i_ker_x = ker_x_start; i_ker_x < ker_x_end; ++i_ker_x)
                    {
                        const int32_t idx_x = base_idx_x + i_ker_x;
                        const float16_t *in_px = input_b + ((size_t)idx_y * input_x + idx_x) * input_ch;
                        const float16_t *w_px = kernel + ((size_t)i_ker_y * kernel_x + i_ker_x) * input_ch;
                        arm_depthwise_accumulate_vec_f16(out_px, in_px, w_px, input_ch);
                    }
                }

                arm_nn_vector_clamp_f16(out_px, input_ch, output_activation_min, output_activation_max);
            }
        }
    }
}

static arm_cmsis_nn_status arm_depthwise_conv_nhwc_fast_chmult1_kc_nt_t_f16(const cmsis_nn_context *ctx,
                                                                            const float16_t *input,
                                                                            int32_t input_batches,
                                                                            int32_t input_x,
                                                                            int32_t input_y,
                                                                            int32_t input_ch,
                                                                            const float16_t *kernel,
                                                                            int32_t kernel_x,
                                                                            int32_t kernel_y,
                                                                            int32_t pad_x,
                                                                            int32_t pad_y,
                                                                            int32_t stride_x,
                                                                            int32_t stride_y,
                                                                            const float16_t *bias,
                                                                            float16_t *output,
                                                                            int32_t output_x,
                                                                            int32_t output_y,
                                                                            float16_t output_activation_min,
                                                                            float16_t output_activation_max)
{
    if (!ctx || !ctx->buf || ctx->size <= 0)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    const int32_t kernel_size = kernel_x * kernel_y;
    if (kernel_size <= 0 || input_ch <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const size_t lhs_row_elems = (size_t)kernel_size * (size_t)input_ch;
    const size_t lhs_row_bytes = lhs_row_elems * sizeof(float16_t);
    if (lhs_row_bytes == 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const size_t max_rows_from_buf = (size_t)ctx->size / lhs_row_bytes;
    int32_t tile_row_limit = (int32_t)max_rows_from_buf;
    if (tile_row_limit > ARM_NN_DW_NT_T_F16_TILE_ROWS)
    {
        tile_row_limit = ARM_NN_DW_NT_T_F16_TILE_ROWS;
    }
    if (tile_row_limit < 1)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    float16_t *lhs_buffer = (float16_t *)ctx->buf;
    const int32_t in_batch_stride = input_x * input_y * input_ch;
    const int32_t out_batch_stride = output_x * output_y * input_ch;

    for (int32_t i_batch = 0; i_batch < input_batches; ++i_batch)
    {
        const float16_t *input_b = input + i_batch * in_batch_stride;
        float16_t *output_b = output + i_batch * out_batch_stride;

        for (int32_t i_out_y = 0; i_out_y < output_y; ++i_out_y)
        {
            const int32_t base_idx_y = (i_out_y * stride_y) - pad_y;

            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x += tile_row_limit)
            {
                int32_t tile_rows = output_x - i_out_x;
                if (tile_rows > tile_row_limit)
                {
                    tile_rows = tile_row_limit;
                }

                float16_t *lhs_ptr = lhs_buffer;
                for (int32_t row = 0; row < tile_rows; ++row)
                {
                    const int32_t base_idx_x = ((i_out_x + row) * stride_x) - pad_x;

                    for (int32_t i_ker_y = 0; i_ker_y < kernel_y; ++i_ker_y)
                    {
                        const int32_t idx_y = base_idx_y + i_ker_y;
                        for (int32_t i_ker_x = 0; i_ker_x < kernel_x; ++i_ker_x)
                        {
                            const int32_t idx_x = base_idx_x + i_ker_x;
                            if ((uint32_t)idx_y >= (uint32_t)input_y || (uint32_t)idx_x >= (uint32_t)input_x)
                            {
                                arm_memset_f16(lhs_ptr, (float16_t)0.0f, (uint32_t)input_ch);
                            }
                            else
                            {
                                const float16_t *in_px = input_b + ((size_t)idx_y * input_x + idx_x) * input_ch;
                                arm_memcpy_f16(lhs_ptr, in_px, (uint32_t)input_ch);
                            }
                            lhs_ptr += input_ch;
                        }
                    }
                }

                float16_t *out_ptr = output_b + ((size_t)i_out_y * output_x + i_out_x) * input_ch;
                arm_cmsis_nn_status status = arm_nn_depthwise_conv_nt_t_f16(lhs_buffer,
                                                                            kernel,
                                                                            bias,
                                                                            out_ptr,
                                                                            tile_rows,
                                                                            input_ch,
                                                                            kernel_size,
                                                                            input_ch,
                                                                            output_activation_min,
                                                                            output_activation_max);
                if (status != ARM_CMSIS_NN_SUCCESS)
                {
                    return status;
                }
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
__STATIC_INLINE bool arm_depthwise_conv_nhwc_convert_to_conv_f16(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                                 const cmsis_nn_dims *input_dims,
                                                                 const cmsis_nn_dims *output_dims)
{
    return dw_conv_params && input_dims && output_dims && input_dims->c == 1 &&
        output_dims->c >= CONVERT_DW_CONV_WITH_ONE_INPUT_CH_AND_OUTPUT_CH_ABOVE_THRESHOLD;
}

__STATIC_INLINE void arm_depthwise_pack_conv_kernel_nt_n_f16(const float16_t *kernel,
                                                             int32_t output_c,
                                                             int32_t kernel_h,
                                                             int32_t kernel_w,
                                                             arm_nn_dw_kernel_layout_f16 kernel_layout,
                                                             float16_t *packed_kernel)
{
    const int32_t kernel_elems = kernel_h * kernel_w;
    const int32_t block_cols = 8;
    const int32_t packed_output_c = ARM_NN_ROUND_UP(output_c, block_cols);

    for (int32_t n_base = 0; n_base < packed_output_c; n_base += block_cols)
    {
        for (int32_t k = 0; k < kernel_elems; ++k)
        {
            float16_t *dst = packed_kernel + (size_t)n_base * kernel_elems + (size_t)k * block_cols;
            for (int32_t lane = 0; lane < block_cols; ++lane)
            {
                const int32_t oc = n_base + lane;
                if (oc < output_c)
                {
                    dst[lane] = (kernel_layout == ARM_NN_DW_KERNEL_CK) ? kernel[(size_t)oc * kernel_elems + k]
                                                                       : kernel[(size_t)k * output_c + oc];
                }
                else
                {
                    dst[lane] = (float16_t)0.0f;
                }
            }
        }
    }
}

static arm_cmsis_nn_status arm_depthwise_conv_nhwc_to_conv_packed_f16(const cmsis_nn_context *ctx,
                                                                      const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                                      const cmsis_nn_dims *input_dims,
                                                                      const float16_t *input,
                                                                      const cmsis_nn_dims *filter_dims,
                                                                      const float16_t *packed_kernel,
                                                                      const float16_t *bias,
                                                                      const cmsis_nn_dims *output_dims,
                                                                      float16_t *output)
{
    if (!ctx || !ctx->buf || ctx->size <= 0 || !dw_conv_params || !input_dims || !input || !packed_kernel ||
        !output_dims || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_w = input_dims->w;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const int32_t patch_len = kernel_h * kernel_w;
    const int32_t output_positions = output_h * output_w;

    if (input_dims->c != 1 || output_c <= 0 || kernel_h <= 0 || kernel_w <= 0 || patch_len <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t stride_h = dw_conv_params->stride.h;
    const int32_t stride_w = dw_conv_params->stride.w;
    const int32_t pad_h = dw_conv_params->padding.h;
    const int32_t pad_w = dw_conv_params->padding.w;
    const int32_t dil_h = dw_conv_params->dilation.h;
    const int32_t dil_w = dw_conv_params->dilation.w;
    const size_t row_bytes = (size_t)patch_len * sizeof(float16_t);
    const int32_t max_rows = (int32_t)((size_t)ctx->size / row_bytes);

    if (max_rows <= 0)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    /*
     * The converted depthwise path always has input_c == 1, so each patch row
     * is simply the kernel footprint scalars. Packing that directly here avoids
     * the general NHWC patch helper overhead.
     */
    float16_t *lhs_buffer = (float16_t *)ctx->buf;
    const int32_t in_batch_stride = input_h * input_w;
    const int32_t out_batch_stride = output_h * output_w * output_c;

    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input + (size_t)b * in_batch_stride;
        float16_t *output_b = output + (size_t)b * out_batch_stride;

        for (int32_t pos = 0; pos < output_positions; pos += max_rows)
        {
            const int32_t rows = ((output_positions - pos) < max_rows) ? (output_positions - pos) : max_rows;

            for (int32_t r = 0; r < rows; ++r)
            {
                const int32_t out_pos = pos + r;
                const int32_t out_y = out_pos / output_w;
                const int32_t out_x = out_pos - out_y * output_w;
                const int32_t base_idx_y = (out_y * stride_h) - pad_h;
                const int32_t base_idx_x = (out_x * stride_w) - pad_w;
                float16_t *patch_row = lhs_buffer + (size_t)r * patch_len;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const int32_t idx_y = base_idx_y + dil_h * ky;
                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const int32_t idx_x = base_idx_x + dil_w * kx;
                        *patch_row++ = ((uint32_t)idx_y < (uint32_t)input_h && (uint32_t)idx_x < (uint32_t)input_w)
                            ? input_b[idx_y * input_w + idx_x]
                            : (float16_t)0.0f;
                    }
                }
            }

            arm_cmsis_nn_status status = arm_nn_mat_mult_nt_n_packed_f16(lhs_buffer,
                                                                         packed_kernel,
                                                                         bias,
                                                                         output_b + (size_t)pos * output_c,
                                                                         rows,
                                                                         output_c,
                                                                         patch_len,
                                                                         output_c,
                                                                         dw_conv_params->activation.min,
                                                                         dw_conv_params->activation.max);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                return status;
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status arm_depthwise_conv_nhwc_to_conv_f16(const cmsis_nn_context *ctx,
                                                               const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                               const cmsis_nn_dims *input_dims,
                                                               const float16_t *input,
                                                               const cmsis_nn_dims *filter_dims,
                                                               const float16_t *kernel,
                                                               const cmsis_nn_dims *bias_dims,
                                                               const float16_t *bias,
                                                               const cmsis_nn_dims *output_dims,
                                                               float16_t *output,
                                                               arm_nn_dw_kernel_layout_f16 kernel_layout)
{
    if (!ctx || !ctx->buf || ctx->size <= 0)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    const int32_t output_c = output_dims->c;
    const int32_t kernel_h = filter_dims->h;
    const int32_t kernel_w = filter_dims->w;
    const size_t kernel_row_elems = (size_t)kernel_h * (size_t)kernel_w;
    const size_t packed_output_c = (size_t)ARM_NN_ROUND_UP(output_c, 8);
    const size_t kernel_bytes = packed_output_c * kernel_row_elems * sizeof(float16_t);
    uint8_t *ctx_bytes = (uint8_t *)ctx->buf;
    const float16_t *conv_kernel;
    size_t scratch_offset = 0U;

    if ((size_t)ctx->size < kernel_bytes)
    {
        return ARM_CMSIS_NN_NO_IMPL_ERROR;
    }

    /*
     * The converted conv path stores the filter in packed NTxN scratch memory
     * and then consumes it as float16_t data. Require float16_t-aligned scratch
     * here so the typed view is explicit and safe.
     */
    if (((uintptr_t)ctx->buf & (sizeof(float16_t) - 1U)) != 0U)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (kernel_layout != ARM_NN_DW_KERNEL_CK && kernel_layout != ARM_NN_DW_KERNEL_KC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    float16_t *packed_kernel = (float16_t *)ctx->buf;
    arm_depthwise_pack_conv_kernel_nt_n_f16(kernel, output_c, kernel_h, kernel_w, kernel_layout, packed_kernel);
    conv_kernel = packed_kernel;
    scratch_offset = kernel_bytes;

    cmsis_nn_context conv_ctx = {0};
    if ((size_t)ctx->size > scratch_offset)
    {
        conv_ctx.buf = ctx_bytes + scratch_offset;
        conv_ctx.size = ctx->size - (int32_t)scratch_offset;
    }

    const cmsis_nn_conv_params_f16 conv_params = {.stride = dw_conv_params->stride,
                                                  .padding = dw_conv_params->padding,
                                                  .dilation = dw_conv_params->dilation,
                                                  .activation = dw_conv_params->activation,
                                                  .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims conv_filter_dims = {.n = output_c, .h = kernel_h, .w = kernel_w, .c = input_dims->c};
    (void)bias_dims;
    (void)conv_filter_dims;
    (void)conv_params;

    return arm_depthwise_conv_nhwc_to_conv_packed_f16(
        &conv_ctx, dw_conv_params, input_dims, input, filter_dims, conv_kernel, bias, output_dims, output);
}
#endif

static void arm_depthwise_conv_f16_generic(const float16_t *input,
                                           const int32_t input_batches,
                                           const int32_t input_x,
                                           const int32_t input_y,
                                           const int32_t input_ch,
                                           const float16_t *kernel,
                                           const int32_t ch_mult,
                                           const int32_t kernel_x,
                                           const int32_t kernel_y,
                                           const int32_t pad_x,
                                           const int32_t pad_y,
                                           const int32_t stride_x,
                                           const int32_t stride_y,
                                           const float16_t *bias,
                                           float16_t *output,
                                           const int32_t output_x,
                                           const int32_t output_y,
                                           const float16_t output_activation_min,
                                           const float16_t output_activation_max,
                                           const int32_t dilation_x,
                                           const int32_t dilation_y,
                                           const cmsis_nn_dw_conv_params_f16 *params,
                                           arm_nn_dw_kernel_layout_f16 kernel_layout)
{
    (void)params;
    const int32_t output_ch = input_ch * ch_mult;
    const int32_t in_batch_stride = input_x * input_y * input_ch;
    const int32_t out_batch_stride = output_x * output_y * output_ch;

    for (int32_t i_batch = 0; i_batch < input_batches; i_batch++)
    {
        const float16_t *input_b = input + i_batch * in_batch_stride;
        float16_t *output_b = output + i_batch * out_batch_stride;

        for (int32_t i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            const int32_t base_idx_y = (i_out_y * stride_y) - pad_y;
            for (int32_t i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_idx_x = (i_out_x * stride_x) - pad_x;
                for (int32_t i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                {
                    for (int32_t i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
                    {
                        const int32_t idx_out_ch = i_ch_mult + i_input_ch * ch_mult;
                        _Float16 acc_0 = (_Float16)0;

                        int32_t ker_y_start;
                        int32_t ker_x_start;
                        int32_t ker_y_end;
                        int32_t ker_x_end;

                        if (dilation_x > 1)
                        {
                            const int32_t start_x_max = (-base_idx_x + dilation_x - 1) / dilation_x;
                            ker_x_start = (start_x_max > 0) ? start_x_max : 0;
                            const int32_t end_min_x = (input_x - base_idx_x + dilation_x - 1) / dilation_x;
                            ker_x_end = (kernel_x < end_min_x) ? kernel_x : end_min_x;
                        }
                        else
                        {
                            ker_x_start = (base_idx_x < 0) ? -base_idx_x : 0;
                            const int32_t end_min_x = input_x - base_idx_x;
                            ker_x_end = (kernel_x < end_min_x) ? kernel_x : end_min_x;
                        }

                        if (dilation_y > 1)
                        {
                            const int32_t start_y_max = (-base_idx_y + dilation_y - 1) / dilation_y;
                            ker_y_start = (start_y_max > 0) ? start_y_max : 0;
                            const int32_t end_min_y = (input_y - base_idx_y + dilation_y - 1) / dilation_y;
                            ker_y_end = (kernel_y < end_min_y) ? kernel_y : end_min_y;
                        }
                        else
                        {
                            ker_y_start = (base_idx_y < 0) ? -base_idx_y : 0;
                            const int32_t end_min_y = input_y - base_idx_y;
                            ker_y_end = (kernel_y < end_min_y) ? kernel_y : end_min_y;
                        }

                        if (bias)
                        {
                            acc_0 = (_Float16)bias[idx_out_ch];
                        }

                        for (int32_t i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                        {
                            const int32_t idx_y = base_idx_y + dilation_y * i_ker_y;
                            for (int32_t i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                            {
                                const int32_t idx_x = base_idx_x + dilation_x * i_ker_x;
                                const int32_t idx_0 =
                                    arm_depthwise_conv_input_index_nhwc(idx_x, idx_y, i_input_ch, input_x, input_ch);
                                int32_t ker_idx_0;
                                if (kernel_layout == ARM_NN_DW_KERNEL_CK)
                                {
                                    ker_idx_0 = idx_out_ch * (kernel_x * kernel_y) + i_ker_y * kernel_x + i_ker_x;
                                }
                                else
                                {
                                    ker_idx_0 = (i_ker_y * kernel_x + i_ker_x) * (output_ch) + idx_out_ch;
                                }

                                acc_0 += (_Float16)input_b[idx_0] * (_Float16)kernel[ker_idx_0];
                            }
                        }

                        acc_0 = CLAMP(acc_0, (_Float16)output_activation_max, (_Float16)output_activation_min);
                        const int32_t out_idx =
                            arm_depthwise_conv_output_index_nhwc(i_out_x, i_out_y, idx_out_ch, output_x, output_ch);
                        output_b[out_idx] = (float16_t)acc_0;
                    }
                }
            }
        }
    }
}

static arm_cmsis_nn_status arm_depthwise_conv_f16_validate(const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const cmsis_nn_dims *output_dims,
                                                           const float16_t *input,
                                                           const float16_t *kernel,
                                                           const float16_t *output,
                                                           arm_nn_dw_kernel_layout_f16 forced_kernel_layout,
                                                           arm_nn_dw_kernel_layout_f16 *kernel_layout)
{
    if (!dw_conv_params || !input_dims || !filter_dims || !output_dims || !input || !kernel || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    *kernel_layout = forced_kernel_layout;

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status arm_depthwise_conv_nhwc_dispatch_f16(const cmsis_nn_context *ctx,
                                                                const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                                const cmsis_nn_dims *input_dims,
                                                                const float16_t *input,
                                                                const cmsis_nn_dims *filter_dims,
                                                                const float16_t *kernel,
                                                                const cmsis_nn_dims *bias_dims,
                                                                const float16_t *bias,
                                                                const cmsis_nn_dims *output_dims,
                                                                float16_t *output,
                                                                arm_nn_dw_kernel_layout_f16 kernel_layout)
{
#ifndef NN_DISABLE_SPECIALIZATION
    /* First try exact-shape NHWC specializations such as 1D-k3, 2x5, and 3x3 kernels. */
    ARM_DW_DISPATCH(arm_dw_spec_nhwc_f16,
                    ARM_DW_ARRAY_SIZE(arm_dw_spec_nhwc_f16),
                    ctx,
                    dw_conv_params,
                    input_dims,
                    input,
                    filter_dims,
                    kernel,
                    bias_dims,
                    bias,
                    output_dims,
                    output,
                    kernel_layout);
#endif

    /* Then try the broader ch_mult=1 fast NHWC routes before falling back to the generic kernel. */
    if (dw_conv_params->ch_mult == 1 && dw_conv_params->dilation.w == 1 && dw_conv_params->dilation.h == 1)
    {
        if (arm_depthwise_conv_nhwc_fast_chmult1_kc_nt_t_f16(ctx,
                                                             input,
                                                             input_dims->n,
                                                             input_dims->w,
                                                             input_dims->h,
                                                             input_dims->c,
                                                             kernel,
                                                             filter_dims->w,
                                                             filter_dims->h,
                                                             dw_conv_params->padding.w,
                                                             dw_conv_params->padding.h,
                                                             dw_conv_params->stride.w,
                                                             dw_conv_params->stride.h,
                                                             bias,
                                                             output,
                                                             output_dims->w,
                                                             output_dims->h,
                                                             dw_conv_params->activation.min,
                                                             dw_conv_params->activation.max) == ARM_CMSIS_NN_SUCCESS)
        {
            return ARM_CMSIS_NN_SUCCESS;
        }

        arm_depthwise_conv_nhwc_fast_chmult1_kc_f16(input,
                                                    input_dims->n,
                                                    input_dims->w,
                                                    input_dims->h,
                                                    input_dims->c,
                                                    kernel,
                                                    filter_dims->w,
                                                    filter_dims->h,
                                                    dw_conv_params->padding.w,
                                                    dw_conv_params->padding.h,
                                                    dw_conv_params->stride.w,
                                                    dw_conv_params->stride.h,
                                                    bias,
                                                    output,
                                                    output_dims->w,
                                                    output_dims->h,
                                                    dw_conv_params->activation.min,
                                                    dw_conv_params->activation.max);
        return ARM_CMSIS_NN_SUCCESS;
    }

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* For one-input-channel cases on MVE, reusing the float convolution wrapper can be faster. */
    if (arm_depthwise_conv_nhwc_convert_to_conv_f16(dw_conv_params, input_dims, output_dims))
    {
        arm_cmsis_nn_status conv_status = arm_depthwise_conv_nhwc_to_conv_f16(ctx,
                                                                              dw_conv_params,
                                                                              input_dims,
                                                                              input,
                                                                              filter_dims,
                                                                              kernel,
                                                                              bias_dims,
                                                                              bias,
                                                                              output_dims,
                                                                              output,
                                                                              kernel_layout);
        if (conv_status == ARM_CMSIS_NN_SUCCESS)
        {
            return conv_status;
        }
    }
#endif

    /* Generic NHWC fallback for remaining channel-multiplier, dilation, or kernel-layout cases. */
    arm_depthwise_conv_f16_generic(input,
                                   input_dims->n,
                                   input_dims->w,
                                   input_dims->h,
                                   input_dims->c,
                                   kernel,
                                   dw_conv_params->ch_mult,
                                   filter_dims->w,
                                   filter_dims->h,
                                   dw_conv_params->padding.w,
                                   dw_conv_params->padding.h,
                                   dw_conv_params->stride.w,
                                   dw_conv_params->stride.h,
                                   bias,
                                   output,
                                   output_dims->w,
                                   output_dims->h,
                                   dw_conv_params->activation.min,
                                   dw_conv_params->activation.max,
                                   dw_conv_params->dilation.w,
                                   dw_conv_params->dilation.h,
                                   dw_conv_params,
                                   kernel_layout);

    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_depthwise_nhwc_conv_f16(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const float16_t *input,
                                                const cmsis_nn_dims *filter_dims,
                                                const float16_t *kernel,
                                                const cmsis_nn_dims *bias_dims,
                                                const float16_t *bias,
                                                const cmsis_nn_dims *output_dims,
                                                float16_t *output)
{
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    arm_cmsis_nn_status status = arm_depthwise_conv_f16_validate(dw_conv_params,
                                                                 input_dims,
                                                                 filter_dims,
                                                                 output_dims,
                                                                 input,
                                                                 kernel,
                                                                 output,
                                                                 ARM_NN_DW_KERNEL_KC,
                                                                 &kernel_layout);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return status;
    }

    return arm_depthwise_conv_nhwc_dispatch_f16(ctx,
                                                dw_conv_params,
                                                input_dims,
                                                input,
                                                filter_dims,
                                                kernel,
                                                bias_dims,
                                                bias,
                                                output_dims,
                                                output,
                                                kernel_layout);
}

arm_cmsis_nn_status arm_depthwise_conv_f16(const cmsis_nn_context *ctx,
                                           const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const float16_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const float16_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output,
                                           arm_nn_tensor_layout layout)
{
    if (!dw_conv_params || layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    return arm_depthwise_nhwc_conv_f16(
        ctx, dw_conv_params, input_dims, input, filter_dims, kernel, bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status arm_depthwise_conv_wrapper_f16(const cmsis_nn_context *ctx,
                                                   const cmsis_nn_dw_conv_params_f16 *dw_conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const float16_t *input,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const float16_t *kernel,
                                                   const cmsis_nn_dims *bias_dims,
                                                   const float16_t *bias,
                                                   const cmsis_nn_dims *output_dims,
                                                   float16_t *output)
{
    arm_nn_dw_kernel_layout_f16 kernel_layout;
    arm_cmsis_nn_status status = arm_depthwise_conv_f16_validate(dw_conv_params,
                                                                 input_dims,
                                                                 filter_dims,
                                                                 output_dims,
                                                                 input,
                                                                 kernel,
                                                                 output,
                                                                 ARM_NN_DW_KERNEL_KC,
                                                                 &kernel_layout);
    if (status != ARM_CMSIS_NN_SUCCESS)
    {
        return status;
    }

    return arm_depthwise_conv_nhwc_dispatch_f16(ctx,
                                                dw_conv_params,
                                                input_dims,
                                                input,
                                                filter_dims,
                                                kernel,
                                                bias_dims,
                                                bias,
                                                output_dims,
                                                output,
                                                kernel_layout);
}
/**
 * @} end of NNConv group
 */
