/*
 * SPDX-FileCopyrightText: Copyright 2023-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_get_buffer_sizes_s4.c
 * Description:  Collection of get buffer size functions for the various s4 convolution layer functions.
 *
 * $Date:        02 November 2026
 * $Revision:    V.1.1.2
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_compiler.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup NNConv
 */

/**
 * @addtogroup GetBufferSizeNNConv
 * @{
 */

__STATIC_INLINE int32_t arm_convolve_s4_get_buffer_size_mve(const cmsis_nn_dims *input_dims,
                                                            const cmsis_nn_dims *filter_dims)
{
    int32_t col_length = input_dims->c * filter_dims->w * filter_dims->h;
    // Get number of complete lanes with int8 elements (multiple of 16) for given col_length. This is dependent on
    // implementation of arm_nn_mat_mult_nt_t_s4
    col_length = (col_length + 15) / 16;
    // 4 -> number of im2col buffers, 16 -> 16 elements per Q register
    return 4 * col_length * 16 * (int32_t)sizeof(int8_t);
}

__STATIC_INLINE int32_t arm_convolve_1_x_n_s4_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                                  const cmsis_nn_dims *input_dims,
                                                                  const cmsis_nn_dims *filter_dims,
                                                                  const cmsis_nn_dims *output_dims)
{
    (void)conv_params;
    (void)input_dims;
    (void)filter_dims;
    (void)output_dims;
    return 0;
}

int32_t arm_convolve_s4_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    const int32_t rhs_cols = filter_dims->w * filter_dims->h * input_dims->c;
    return (2 * rhs_cols) * (int32_t)sizeof(int16_t);
}

int32_t arm_convolve_1_x_n_s4_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_dims *input_dims,
                                              const cmsis_nn_dims *filter_dims,
                                              const cmsis_nn_dims *output_dims)
{
#if !defined(ARM_MATH_MVEI)
    (void)conv_params;
    (void)output_dims;

    return arm_convolve_s4_get_buffer_size(input_dims, filter_dims);
#else
    return arm_convolve_1_x_n_s4_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
#endif
}

int32_t arm_convolve_1x1_s4_fast_get_buffer_size(const cmsis_nn_dims *input_dims)
{
    (void)input_dims;
    return 0;
}

/*
 * Get the required buffer size for arm_convolve_wrapper_s4. This is the
 * recommended convolve wrapper s4 function.
 *
 * Refer to header file for details.
 *
 */
int32_t arm_convolve_wrapper_s4_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims)
{
#if defined(ARM_MATH_MVEI)
    return arm_convolve_wrapper_s4_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
#else
    (void)output_dims;
    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))
    {
        if (arm_nn_is_convolve_1x1_fast(conv_params))
        {
            return arm_convolve_1x1_s4_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return arm_convolve_s4_get_buffer_size(input_dims, filter_dims);
    }
#endif
}

int32_t arm_convolve_wrapper_s4_get_buffer_size_mve(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims)

{
    if (arm_nn_is_convolve_1x1(conv_params, input_dims, filter_dims))
    {
        if (arm_nn_is_convolve_1x1_fast(conv_params))
        {
            return arm_convolve_1x1_s4_fast_get_buffer_size(input_dims);
        }
        else
        {
            return 0;
        }
    }
    else if (arm_nn_is_convolve_1_x_n_s4(conv_params, input_dims, filter_dims, output_dims))
    {
        return arm_convolve_1_x_n_s4_get_buffer_size_mve(conv_params, input_dims, filter_dims, output_dims);
    }
    else
    {
        return arm_convolve_s4_get_buffer_size_mve(input_dims, filter_dims);
    }
}

int32_t arm_convolve_wrapper_s4_get_buffer_size_dsp(const cmsis_nn_conv_params *conv_params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const cmsis_nn_dims *filter_dims,
                                                    const cmsis_nn_dims *output_dims)
{
    return arm_convolve_wrapper_s4_get_buffer_size(conv_params, input_dims, filter_dims, output_dims);
}
/**
 * @} end of GetBufferSizeNNConv group
 */
