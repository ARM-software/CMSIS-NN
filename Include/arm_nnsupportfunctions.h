/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nnsupportfunctions.h
 * Description:  Public header file of support functions for CMSIS NN Library
 *
 * $Date:        23 Mars 2023
 * $Revision:    V.16.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef _ARM_NNSUPPORTFUNCTIONS_H_
#define _ARM_NNSUPPORTFUNCTIONS_H_

#include "Internal/arm_nn_compiler.h"
#include "arm_nn_math_types.h"
#include "arm_nn_types.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define USE_FAST_DW_CONV_S16_FUNCTION(dw_conv_params, filter_dims, input_dims)                                         \
    (dw_conv_params->ch_mult == 1 && dw_conv_params->dilation.w == 1 && dw_conv_params->dilation.h == 1 &&             \
     filter_dims->w * filter_dims->h < 512)

#define LEFT_SHIFT(_shift) (_shift > 0 ? _shift : 0)
#define RIGHT_SHIFT(_shift) (_shift > 0 ? 0 : -_shift)
#define MASK_IF_ZERO(x) (x) == 0 ? ~0 : 0
#define MASK_IF_NON_ZERO(x) (x) != 0 ? ~0 : 0
#define SELECT_USING_MASK(mask, a, b) ((mask) & (a)) ^ (~(mask) & (b))

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define CLAMP(x, h, l) MAX(MIN((x), (h)), (l))
#define REDUCE_MULTIPLIER(_mult) ((_mult < 0x7FFF0000) ? ((_mult + (1 << 15)) >> 16) : 0x7FFF)

// Number of channels processed in a block for DW Conv(MVE)
// Requirement: Greater than 0 & less than 128
// This can be fine tuned to match number of input channels for best performance.
// A layer with lower number of channels than CH_IN_BLOCK_MVE will result in higher
// scratch buffer usage and a layer with higher number of channels than CH_IN_BLOCK_MVE
// will result in lower scratch buffer usage.
#define CH_IN_BLOCK_MVE (124)

/**
 * @brief definition to pack four 8 bit values.
 */
#define PACK_S8x4_32x1(v0, v1, v2, v3)                                                                                 \
    ((((int32_t)(v0) << 0) & (int32_t)0x000000FF) | (((int32_t)(v1) << 8) & (int32_t)0x0000FF00) |                     \
     (((int32_t)(v2) << 16) & (int32_t)0x00FF0000) | (((int32_t)(v3) << 24) & (int32_t)0xFF000000))

/**
 * @brief definition to pack two 16 bit values.
 */
#define PACK_Q15x2_32x1(v0, v1) (((int32_t)v0 & (int32_t)0xFFFF) | ((int32_t)v1 << 16))

/**
 * @brief Union for SIMD access of q31/s16/s8 types
 */
union arm_nnword
{
    int32_t word;
    /**< q31 type */
    int16_t half_words[2];
    /**< s16 type */
    int8_t bytes[4];
    /**< s8 type */
};

/**
 * @brief Union for data type long long
 */
struct arm_nn_double
{
    uint32_t low;
    int32_t high;
};

union arm_nn_long_long
{
    int64_t long_long;
    struct arm_nn_double word;
};

/**
 * @defgroup groupSupport Private
 *
 * Internal Support functions. Not intended to be called direclty by a CMSIS-NN user.
 *
 */

/**
 * @defgroup supportConversion Data Conversion
 *
 * Perform data type conversion in-between neural network operations
 *
 */

/**
 * @brief Converts the elements from a s8 vector to a s16 vector with an added offset
 * @param[in]    src        pointer to the s8 input vector
 * @param[out]   dst        pointer to the s16 output vector
 * @param[in]    block_size length of the input vector
 * @param[in]    offset     s16 offset to be added to each input vector element.
 *
 * \par Description:
 *
 * Output elements are ordered.
 * The equation used for the conversion process is:
 *
 * <pre>
 *  dst[n] = (int16_t) src[n] + offset;   0 <= n < block_size.
 * </pre>
 *
 */
void arm_q7_to_q15_with_offset(const int8_t *src, int16_t *dst, int32_t block_size, int16_t offset);

#if defined(ARM_MATH_DSP)
/**
 * @brief Converts the elements from a s8 vector to a s16 vector with an added offset
 * @param[in]    src        pointer to the s8 input vector
 * @param[out]   dst        pointer to the s16 output vector
 * @param[in]    block_size length of the input vector
 * @param[in]    offset     s16 offset to be added to each input vector element.
 *
 * \par Description:
 *
 * No additonal ordering is done with the result that output elements are not in order.
 * Instead of ABCD order will be ACBD.
 * Note this is for processors with DSP extension only.
 * The equation used for the conversion process is:
 *
 * <pre>
 *  dst[n - 0] = (int16_t) src[n - 0] + offset;   0 <= n < block_size.
 *  dst[n - 1] = (int16_t) src[n - 2] + offset;   0 <= n < block_size.
 *  dst[n - 2] = (int16_t) src[n - 1] + offset;   0 <= n < block_size.
 *  dst[n - 3] = (int16_t) src[n - 3] + offset;   0 <= n < block_size.
 * </pre>
 *
 */
void arm_s8_to_s16_unordered_with_offset(const int8_t *src, int16_t *dst, int32_t block_size, int16_t offset);
#endif

/**
 * @brief Depthwise conv on an im2col buffer where the input channel equals output channel.
 * @param[in]    row     pointer to row
 * @param[in]    col     pointer to im2col buffer, always consists of 2 columns.
 * @param[in]    num_ch   number of channels
 * @param[in]    out_shift  pointer to per output channel requantization shift parameter.
 * @param[in]    out_mult   pointer to per output channel requantization multiplier parameter.
 * @param[in]    out_offset      output tensor offset.
 * @param[in]    activation_min   minimum value to clamp the output to. Range : int8
 * @param[in]    activation_max   maximum value to clamp the output to. Range : int8
 * @param[in]    kernel_size   number of elements in one column.
 * @param[in]    output_bias per output channel bias. Range : int32
 * @param[out]   out         pointer to output
 * @return     The function returns one of the two
 *              1. The incremented output pointer for a successful operation or
 *              2. NULL if implementation is not available.
 *
 * @details     Supported framework: TensorFlow Lite micro.
 */
int8_t *arm_nn_depthwise_conv_s8_core(const int8_t *row,
                                      const int16_t *col,
                                      const uint16_t num_ch,
                                      const int32_t *out_shift,
                                      const int32_t *out_mult,
                                      const int32_t out_offset,
                                      const int32_t activation_min,
                                      const int32_t activation_max,
                                      const uint16_t kernel_size,
                                      const int32_t *const output_bias,
                                      int8_t *out);

/**
 * @brief General Matrix-multiplication function with per-channel requantization.
 * @param[in]       input_row    pointer to row operand
 * @param[in]       input_col    pointer to col operand
 * @param[in]       output_ch    number of rows of input_row
 * @param[in]       col_batches  number of column batches. Range: 1 to 4
 * @param[in]       output_shift  pointer to per output channel requantization shift parameter.
 * @param[in]       output_mult   pointer to per output channel requantization multiplier parameter.
 * @param[in]       out_offset    output tensor offset.
 * @param[in]       col_offset    input tensor(col) offset.
 * @param[in]       row_offset    kernel offset(row). Not used.
 * @param[in]       out_activation_min   minimum value to clamp the output to. Range : int8
 * @param[in]       out_activation_max   maximum value to clamp the output to. Range : int8
 * @param[in]       row_len       number of elements in each row
 * @param[in]       bias          per output channel bias. Range : int32
 * @param[in,out]   out           pointer to output
 * @return     The function returns one of the two
 *              1. The incremented output pointer for a successful operation or
 *              2. NULL if implementation is not available.
 *
 * @details   Supported framework: TensorFlow Lite
 */
int8_t *arm_nn_mat_mult_s8(const int8_t *input_row,
                           const int8_t *input_col,
                           const uint16_t output_ch,
                           const uint16_t col_batches,
                           const int32_t *output_shift,
                           const int32_t *output_mult,
                           const int32_t out_offset,
                           const int32_t col_offset,
                           const int32_t row_offset,
                           const int16_t out_activation_min,
                           const int16_t out_activation_max,
                           const uint16_t row_len,
                           const int32_t *const bias,
                           int8_t *out);
/**
 * @brief Matrix-multiplication function for convolution with per-channel requantization for 16 bits convolution.
 * @param[in]       input_a     pointer to operand A
 * @param[in]       input_b     pointer to operand B, always consists of 2 vectors.
 * @param[in]       output_ch   number of rows of A
 * @param[in]       out_shift  pointer to per output channel requantization shift parameter.
 * @param[in]       out_mult   pointer to per output channel requantization multiplier parameter.
 * @param[in]       activation_min   minimum value to clamp the output to. Range : int16
 * @param[in]       activation_max   maximum value to clamp the output to. Range : int16
 * @param[in]       num_col_a   number of columns of A
 * @param[in]       output_bias per output channel bias. Range : int64
 * @param[in,out]   out_0       pointer to output
 * @return     The function returns one of the two
 *              1. The incremented output pointer for a successful operation or
 *              2. NULL if implementation is not available.
 *
 * @details   This function does the matrix multiplication of weight matrix for all output channels
 *            with 2 columns from im2col and produces two elements/output_channel. The outputs are
 *            clamped in the range provided by activation min and max.
 *            Supported framework: TensorFlow Lite micro.
 */
int16_t *arm_nn_mat_mult_kernel_s16(const int8_t *input_a,
                                    const int16_t *input_b,
                                    const int32_t output_ch,
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int16_t activation_min,
                                    const int16_t activation_max,
                                    const int32_t num_col_a,
                                    const int64_t *const output_bias,
                                    int16_t *out_0);

/**
 * @brief General Vector by Matrix multiplication with requantization and storage of result.
 * @param[in]       row_elements          number of row elements
 * @param[in]       skipped_row_elements  number of row elements skipped due to padding.
 *                                        row_elements + skipped_row_elements = (kernel_x * kernel_y) * input_ch
 * @param[in]       row_base_ref          pointer to row operand
 * @param[in]       col_base_ref          pointer to col operand
 * @param[out]      out_ch                Number of output channels
 * @param[in]       conv_params           Pointer to convolution parameters like offsets and activation values
 * @param[in]       quant_params          Pointer to per-channel quantization parameters
 * @param[in]       bias                  Pointer to optional per-channel bias
 * @param[out]      output                Pointer to output where int8 results are stored.
 * @return     The function performs matrix(row_base_ref) multiplication with vector(col_base_ref) and
 *             scaled result is stored in memory.
 *
 * @details Pseudo-code
 *      *output = 0
 *      sum_col = 0
 *      for (j = 0; j < out_ch; j++)
 *      for (i = 0; i < row_elements; i++)
 *          *output += row_base_ref[i] * col_base_ref[i]
 *          sum_col += col_base_ref[i]
 *      scale sum_col using quant_params and bias
 *      store result in 'output'
 *
 *
 */
arm_cmsis_nn_status arm_nn_mat_mul_core_1x_s8(int32_t row_elements,
                                              const int32_t skipped_row_elements,
                                              const int8_t *row_base_ref,
                                              const int8_t *col_base_ref,
                                              const int32_t out_ch,
                                              const cmsis_nn_conv_params *conv_params,
                                              const cmsis_nn_per_channel_quant_params *quant_params,
                                              const int32_t *bias,
                                              int8_t *output);

/**
 * @brief Matrix-multiplication with requantization & activation function for four rows and one column
 * @param[in]       row_elements  number of row elements
 * @param[in]       offset        offset between rows. Can be the same as row_elements.
 *                                For e.g, in a 1x1 conv scenario with stride as 1.
 * @param[in]       row_base      pointer to row operand
 * @param[in]       col_base      pointer to col operand
 * @param[in]       out_ch        Number of output channels
 * @param[in]       conv_params   Pointer to convolution parameters like offsets and activation values
 * @param[in]       quant_params  Pointer to per-channel quantization parameters
 * @param[in]       bias          Pointer to per-channel bias
 * @param[out]      output        Pointer to output where int8 results are stored.
 *
 * @return     The function returns the updated output pointer or NULL if implementation is not available.
 *
 * @details Compliant to TFLM int8 specification. MVE implementation only
 */
int8_t *arm_nn_mat_mul_core_4x_s8(const int32_t row_elements,
                                  const int32_t offset,
                                  const int8_t *row_base,
                                  const int8_t *col_base,
                                  const int32_t out_ch,
                                  const cmsis_nn_conv_params *conv_params,
                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                  const int32_t *bias,
                                  int8_t *output);

/**
 * @brief General Matrix-multiplication function with per-channel requantization.
 *        This function assumes:
 *        - LHS input matrix NOT transposed (nt)
 *        - RHS input matrix transposed (t)
 *
 *  @note This operation also performs the broadcast bias addition before the requantization
 *
 * @param[in]  lhs                Pointer to the LHS input matrix
 * @param[in]  rhs                Pointer to the RHS input matrix
 * @param[in]  bias               Pointer to the bias vector. The length of this vector is equal to the number of
 *                                output columns (or RHS input rows)
 * @param[out] dst                Pointer to the output matrix with "m" rows and "n" columns
 * @param[in]  dst_multipliers    Pointer to the multipliers vector needed for the per-channel requantization.
 *                                The length of this vector is equal to the number of output columns (or RHS input
 *                                rows)
 * @param[in]  dst_shifts         Pointer to the shifts vector needed for the per-channel requantization. The length
 *                                of this vector is equal to the number of output columns (or RHS input rows)
 * @param[in]  lhs_rows           Number of LHS input rows
 * @param[in]  rhs_rows           Number of RHS input rows
 * @param[in]  rhs_cols           Number of LHS/RHS input columns
 * @param[in]  lhs_offset         Offset to be applied to the LHS input value
 * @param[in]  dst_offset         Offset to be applied the output result
 * @param[in]  activation_min     Minimum value to clamp down the output. Range : int8
 * @param[in]  activation_max     Maximum value to clamp up the output. Range : int8
 * @param[in]  lhs_cols_offset    Column offset between subsequent lhs_rows
 *
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 */
arm_cmsis_nn_status arm_nn_mat_mult_nt_t_s8(const int8_t *lhs,
                                            const int8_t *rhs,
                                            const int32_t *bias,
                                            int8_t *dst,
                                            const int32_t *dst_multipliers,
                                            const int32_t *dst_shifts,
                                            const int32_t lhs_rows,
                                            const int32_t rhs_rows,
                                            const int32_t rhs_cols,
                                            const int32_t lhs_offset,
                                            const int32_t dst_offset,
                                            const int32_t activation_min,
                                            const int32_t activation_max,
                                            const int32_t lhs_cols_offset);

/**
 * @brief s8 Vector by Matrix (transposed) multiplication
 *
 * @param[in]      lhs             Input left-hand side vector
 * @param[in]      rhs             Input right-hand side matrix (transposed)
 * @param[in]      bias            Input bias
 * @param[out]     dst             Output vector
 * @param[in]      lhs_offset      Offset to be added to the input values of the left-hand side vector.
 *                                 Range: -127 to 128
 * @param[in]      dst_offset      Offset to be added to the output values. Range: -127 to 128
 * @param[in]      dst_multiplier  Output multiplier
 * @param[in]      dst_shift       Output shift
 * @param[in]      rhs_cols        Number of columns in the right-hand side input matrix
 * @param[in]      rhs_rows        Number of rows in the right-hand side input matrix
 * @param[in]      activation_min  Minimum value to clamp the output to. Range: int8
 * @param[in]      activation_max  Maximum value to clamp the output to. Range: int8
 * @param[in]      address_offset  Memory position offset for dst. First output is stored at 'dst', the
 *                                 second at 'dst + address_offset' and so on. Default value is typically 1.
 *
 * @return         The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 */
arm_cmsis_nn_status arm_nn_vec_mat_mult_t_s8(const int8_t *lhs,
                                             const int8_t *rhs,
                                             const int32_t *bias,
                                             int8_t *dst,
                                             const int32_t lhs_offset,
                                             const int32_t dst_offset,
                                             const int32_t dst_multiplier,
                                             const int32_t dst_shift,
                                             const int32_t rhs_cols,
                                             const int32_t rhs_rows,
                                             const int32_t activation_min,
                                             const int32_t activation_max,
                                             const int32_t address_offset);

/**
 * @brief s16 Vector by Matrix (transposed) multiplication
 *
 * @param[in]      lhs             Input left-hand side vector
 * @param[in]      rhs             Input right-hand side matrix (transposed)
 * @param[in]      bias            Input bias
 * @param[out]     dst             Output vector
 * @param[in]      dst_multiplier  Output multiplier
 * @param[in]      dst_shift       Output shift
 * @param[in]      rhs_cols        Number of columns in the right-hand side input matrix
 * @param[in]      rhs_rows        Number of rows in the right-hand side input matrix
 * @param[in]      activation_min  Minimum value to clamp the output to. Range: int16
 * @param[in]      activation_max  Maximum value to clamp the output to. Range: int16
 *
 * @return         The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 */
arm_cmsis_nn_status arm_nn_vec_mat_mult_t_s16(const int16_t *lhs,
                                              const int8_t *rhs,
                                              const int64_t *bias,
                                              int16_t *dst,
                                              const int32_t dst_multiplier,
                                              const int32_t dst_shift,
                                              const int32_t rhs_cols,
                                              const int32_t rhs_rows,
                                              const int32_t activation_min,
                                              const int32_t activation_max);

/**
 * @brief s8 Vector by Matrix (transposed) multiplication with s16 output
 *
 * @param[in]      lhs             Input left-hand side vector
 * @param[in]      rhs             Input right-hand side matrix (transposed)
 * @param[out]     dst             Output vector
 * @param[in]      lhs_offset      Offset to be added to the input values of the left-hand side
 *                                 vector. Range: -127 to 128
 * @param[in]      scatter_offset  Address offset for dst. First output is stored at 'dst', the
 *                                 second at 'dst + scatter_offset' and so on.
 * @param[in]      dst_multiplier  Output multiplier
 * @param[in]      dst_shift       Output shift
 * @param[in]      rhs_cols        Number of columns in the right-hand side input matrix
 * @param[in]      rhs_rows        Number of rows in the right-hand side input matrix
 * @param[in]      activation_min  Minimum value to clamp the output to. Range: int16
 * @param[in]      activation_max  Maximum value to clamp the output to. Range: int16
 *
 * @return         The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 *
 */
arm_cmsis_nn_status arm_nn_vec_mat_mult_t_svdf_s8(const int8_t *lhs,
                                                  const int8_t *rhs,
                                                  int16_t *dst,
                                                  const int32_t lhs_offset,
                                                  const int32_t scatter_offset,
                                                  const int32_t dst_multiplier,
                                                  const int32_t dst_shift,
                                                  const int32_t rhs_cols,
                                                  const int32_t rhs_rows,
                                                  const int32_t activation_min,
                                                  const int32_t activation_max);

/**
 * @brief Depthwise convolution of transposed rhs matrix with 4 lhs matrices. To be used in padded cases where
 *        the padding is -lhs_offset(Range: int8). Dimensions are the same for lhs and rhs.
 *
 * @param[in]      lhs             Input left-hand side matrix
 * @param[in]      rhs             Input right-hand side matrix (transposed)
 * @param[in]      lhs_offset      LHS matrix offset(input offset). Range: -127 to 128
 * @param[in]      active_ch       Subset of total_ch processed
 * @param[in]      total_ch        Number of channels in LHS/RHS
 * @param[in]      out_shift       Per channel output shift. Length of vector is equal to number of channels
 * @param[in]      out_mult        Per channel output multiplier. Length of vector is equal to number of channels
 * @param[in]      out_offset      Offset to be added to the output values. Range: -127 to 128
 * @param[in]      activation_min  Minimum value to clamp the output to. Range: int8
 * @param[in]      activation_max  Maximum value to clamp the output to. Range: int8
 * @param[in]       row_x_col       (row_dimension * col_dimension) of LHS/RHS matrix
 * @param[in]      output_bias     Per channel output bias. Length of vector is equal to number of channels
 * @param[in]      out             Output pointer
 *
 * @return         The function returns one of the two
 *                  - Updated output pointer if an implementation is available
 *                  - NULL if no implementation is available.
 *
 * @note           If number of channels is not a multiple of 4, upto 3 elements outside the boundary will be read
 * out for the following.
 *                  - Output shift
 *                  - Output multiplier
 *                  - Output bias
 *                  - rhs
 */
arm_cmsis_nn_status arm_nn_depthwise_conv_nt_t_padded_s8(const int8_t *lhs,
                                                         const int8_t *rhs,
                                                         const int32_t lhs_offset,
                                                         const int32_t active_ch,
                                                         const int32_t total_ch,
                                                         const int32_t *out_shift,
                                                         const int32_t *out_mult,
                                                         const int32_t out_offset,
                                                         const int32_t activation_min,
                                                         const int32_t activation_max,
                                                         const uint16_t row_x_col,
                                                         const int32_t *const output_bias,
                                                         int8_t *out);

/**
 * @brief Depthwise convolution of transposed rhs matrix with 4 lhs matrices. To be used in non-padded cases.
 *        Dimensions are the same for lhs and rhs.
 *
 * @param[in]      lhs             Input left-hand side matrix
 * @param[in]      rhs             Input right-hand side matrix (transposed)
 * @param[in]      lhs_offset      LHS matrix offset(input offset). Range: -127 to 128
 * @param[in]      active_ch       Subset of total_ch processed
 * @param[in]      total_ch        Number of channels in LHS/RHS
 * @param[in]      out_shift       Per channel output shift. Length of vector is equal to number of channels.
 * @param[in]      out_mult        Per channel output multiplier. Length of vector is equal to number of channels.
 * @param[in]      out_offset      Offset to be added to the output values. Range: -127 to 128
 * @param[in]      activation_min  Minimum value to clamp the output to. Range: int8
 * @param[in]      activation_max  Maximum value to clamp the output to. Range: int8
 * @param[in]       row_x_col       (row_dimension * col_dimension) of LHS/RHS matrix
 * @param[in]      output_bias     Per channel output bias. Length of vector is equal to number of channels.
 * @param[in]      out             Output pointer
 *
 * @return         The function returns one of the two
 *                  - Updated output pointer if an implementation is available
 *                  - NULL if no implementation is available.
 *
 * @note           If number of channels is not a multiple of 4, upto 3 elements outside the boundary will be read
 * out for the following.
 *                  - Output shift
 *                  - Output multiplier
 *                  - Output bias
 *                  - rhs
 */
arm_cmsis_nn_status arm_nn_depthwise_conv_nt_t_s8(const int8_t *lhs,
                                                  const int8_t *rhs,
                                                  const int32_t lhs_offset,
                                                  const int32_t active_ch,
                                                  const int32_t total_ch,
                                                  const int32_t *out_shift,
                                                  const int32_t *out_mult,
                                                  const int32_t out_offset,
                                                  const int32_t activation_min,
                                                  const int32_t activation_max,
                                                  const uint16_t row_x_col,
                                                  const int32_t *const output_bias,
                                                  int8_t *out);

/**
 * @brief Depthwise convolution of transposed rhs matrix with 4 lhs matrices. To be used in non-padded cases.
 *        Dimensions are the same for lhs and rhs.
 *
 * @param[in]      lhs             Input left-hand side matrix
 * @param[in]      rhs             Input right-hand side matrix (transposed)
 * @param[in]      num_ch          Number of channels in LHS/RHS
 * @param[in]      out_shift       Per channel output shift. Length of vector is equal to number of channels.
 * @param[in]      out_mult        Per channel output multiplier. Length of vector is equal to number of channels.
 * @param[in]      activation_min  Minimum value to clamp the output to. Range: int8
 * @param[in]      activation_max  Maximum value to clamp the output to. Range: int8
 * @param[in]       row_x_col       (row_dimension * col_dimension) of LHS/RHS matrix
 * @param[in]      output_bias     Per channel output bias. Length of vector is equal to number of channels.
 * @param[in]      out             Output pointer
 *
 * @return         The function returns one of the two
 *                  - Updated output pointer if an implementation is available
 *                  - NULL if no implementation is available.
 *
 * @note           If number of channels is not a multiple of 4, upto 3 elements outside the boundary will be read
 * out for the following.
 *                  - Output shift
 *                  - Output multiplier
 *                  - Output bias
 *                  - rhs
 */
int16_t *arm_nn_depthwise_conv_nt_t_s16(const int16_t *lhs,
                                        const int8_t *rhs,
                                        const uint16_t num_ch,
                                        const int32_t *out_shift,
                                        const int32_t *out_mult,
                                        const int32_t activation_min,
                                        const int32_t activation_max,
                                        const uint16_t row_x_col,
                                        const int64_t *const output_bias,
                                        int16_t *out);

/**
  @brief         Read 2 s16 elements and post increment pointer.
  @param[in]     in_q15   Pointer to pointer that holds address of input.
  @return        q31 value
 */
__STATIC_FORCEINLINE int32_t arm_nn_read_q15x2_ia(const int16_t **in_q15)
{
    int32_t val;

    memcpy(&val, *in_q15, 4);
    *in_q15 += 2;

    return (val);
}

/**
  @brief         Read 4 s8 from s8 pointer and post increment pointer.
  @param[in]     in_s8       Pointer to pointer that holds address of input.
  @return        q31 value
 */
__STATIC_FORCEINLINE int32_t arm_nn_read_s8x4_ia(const int8_t **in_s8)
{
    int32_t val;
    memcpy(&val, *in_s8, 4);
    *in_s8 += 4;

    return (val);
}

/**
  @brief         Read 2 int16 values from int16 pointer.
  @param[in]     in     pointer to address of input.
  @return        s32    value
 */
__STATIC_FORCEINLINE int32_t arm_nn_read_s16x2(const int16_t *in)
{
    int32_t val;
    memcpy(&val, in, 4);

    return (val);
}

/**
  @brief         Read 4 s8 values.
  @param[in]     in_s8       pointer to address of input.
  @return        s32 value
 */
__STATIC_FORCEINLINE int32_t arm_nn_read_s8x4(const int8_t *in_s8)
{
    int32_t val;
    memcpy(&val, in_s8, 4);

    return (val);
}

/**
  @brief         Write four s8 to s8 pointer and increment pointer afterwards.
  @param[in]     in       Double pointer to input value
  @param[in]     value    Four bytes to copy
 */
__STATIC_FORCEINLINE void arm_nn_write_s8x4_ia(int8_t **in, int32_t value)
{
    memcpy(*in, &value, 4);
    *in += 4;
}

/**
 * @brief           memset optimized for MVE
 * @param[in, out]  dst         Destination pointer
 * @param[in]       val         Value to set
 * @param[in]       block_size  Number of bytes to copy.
 *
 */
__STATIC_FORCEINLINE void arm_memset_s8(int8_t *dst, const int8_t val, uint32_t block_size)
{
#if defined(ARM_MATH_MVEI)
    __asm volatile("   vdup.8                  q0, %[set_val]             \n"
                   "   wlstp.8                 lr, %[cnt], 1f             \n"
                   "2:                                                    \n"
                   "   vstrb.8                 q0, [%[in]], #16            \n"
                   "   letp                    lr, 2b                     \n"
                   "1:                                                    \n"
                   : [in] "+r"(dst)
                   : [cnt] "r"(block_size), [set_val] "r"(val)
                   : "q0", "memory", "r14");
#else
    memset(dst, val, block_size);
#endif
}

#if defined(ARM_MATH_DSP)

/**
 * @brief read and expand one s8 word into two s16 words with ordering.
 */
__STATIC_FORCEINLINE const int8_t *read_and_pad(const int8_t *source, int32_t *out1, int32_t *out2)
{
    int32_t inA = arm_nn_read_s8x4_ia(&source);
    int32_t inAbuf1 = SXTB16_RORn((uint32_t)inA, 8);
    int32_t inAbuf2 = SXTB16(inA);

    #ifndef ARM_MATH_BIG_ENDIAN
    *out2 = (int32_t)(PKHTB(inAbuf1, inAbuf2, 16));
    *out1 = (int32_t)(PKHBT(inAbuf2, inAbuf1, 16));
    #else
    *out1 = (int32_t)(PKHTB(inAbuf1, inAbuf2, 16));
    *out2 = (int32_t)(PKHBT(inAbuf2, inAbuf1, 16));
    #endif

    return source;
}

/**
 * @brief read and expand one s8 word into two s16 words with no additional ordering.
 */
__STATIC_FORCEINLINE const int8_t *read_and_pad_reordered(const int8_t *source, int32_t *out1, int32_t *out2)
{
    int32_t inA = arm_nn_read_s8x4_ia(&source);
    #ifndef ARM_MATH_BIG_ENDIAN
    *out2 = SXTB16(ROR((uint32_t)inA, 8));
    *out1 = SXTB16(inA);
    #else
    *out1 = SXTB16(ROR((uint32_t)inA, 8));
    *out2 = SXTB16(inA);
    #endif

    return source;
}

#endif

/**
 * @brief Matrix-multiplication function for convolution with per-channel requantization.
 * @param[in]       input_a     pointer to operand A
 * @param[in]       input_b     pointer to operand B, always consists of 2 vectors.
 * @param[in]       output_ch   number of rows of A
 * @param[in]       out_shift  pointer to per output channel requantization shift parameter.
 * @param[in]       out_mult   pointer to per output channel requantization multiplier parameter.
 * @param[in]       out_offset      output tensor offset.
 * @param[in]       activation_min   minimum value to clamp the output to. Range : int8
 * @param[in]       activation_max   maximum value to clamp the output to. Range : int8
 * @param[in]       num_col_a   number of columns of A
 * @param[in]       output_bias per output channel bias. Range : int32
 * @param[in,out]   out_0       pointer to output
 * @return     The function returns one of the two
 *              1. The incremented output pointer for a successful operation or
 *              2. NULL if implementation is not available.
 *
 * @details   This function does the matrix multiplication of weight matrix for all output channels
 *            with 2 columns from im2col and produces two elements/output_channel. The outputs are
 *            clamped in the range provided by activation min and max.
 *            Supported framework: TensorFlow Lite micro.
 */
int8_t *arm_nn_mat_mult_kernel_s8_s16(const int8_t *input_a,
                                      const int16_t *input_b,
                                      const uint16_t output_ch,
                                      const int32_t *out_shift,
                                      const int32_t *out_mult,
                                      const int32_t out_offset,
                                      const int16_t activation_min,
                                      const int16_t activation_max,
                                      const int32_t num_col_a,
                                      const int32_t *const output_bias,
                                      int8_t *out_0);

/**
 * @brief Common softmax function for s8 input and s8 or s16 output
 * @param[in]  input          Pointer to the input tensor
 * @param[in]  num_rows       Number of rows in the input tensor
 * @param[in]  row_size       Number of elements in each input row
 * @param[in]  mult           Input quantization multiplier
 * @param[in]  shift          Input quantization shift within the range [0, 31]
 * @param[in]  diff_min       Minimum difference with max in row. Used to check if
 *                            the quantized exponential operation can be performed
 * @param[in]  int16_output   Indicating s8 output if 0 else s16 output
 * @param[out] output         Pointer to the output tensor
 *
 * @note Supported framework: TensorFlow Lite micro (bit-accurate)
 *
 */
void arm_nn_softmax_common_s8(const int8_t *input,
                              const int32_t num_rows,
                              const int32_t row_size,
                              const int32_t mult,
                              const int32_t shift,
                              const int32_t diff_min,
                              const bool int16_output,
                              void *output);

/**
 * @brief macro for adding rounding offset
 */
#ifndef ARM_NN_TRUNCATE
    #define NN_ROUND(out_shift) ((0x1 << out_shift) >> 1)
#else
    #define NN_ROUND(out_shift) 0
#endif

// Macros for shortening quantization functions' names and avoid long lines
#define MUL_SAT(a, b) arm_nn_doubling_high_mult((a), (b))
#define MUL_SAT_MVE(a, b) arm_doubling_high_mult_mve_32x4((a), (b))
#define MUL_POW2(a, b) arm_nn_mult_by_power_of_two((a), (b))

#define DIV_POW2(a, b) arm_nn_divide_by_power_of_two((a), (b))
#define DIV_POW2_MVE(a, b) arm_divide_by_power_of_two_mve((a), (b))

#define EXP_ON_NEG(x) arm_nn_exp_on_negative_values((x))
#define ONE_OVER1(x) arm_nn_one_over_one_plus_x_for_x_in_0_1((x))

/**
 * @brief           Saturating doubling high multiply. Result matches
 *                  NEON instruction VQRDMULH.
 * @param[in]       m1        Multiplicand. Range: {NN_Q31_MIN, NN_Q31_MAX}
 * @param[in]       m2        Multiplier. Range: {NN_Q31_MIN, NN_Q31_MAX}
 * @return          Result of multiplication.
 *
 */
__STATIC_FORCEINLINE int32_t arm_nn_doubling_high_mult(const int32_t m1, const int32_t m2)
{
    int32_t result = 0;
    // Rounding offset to add for a right shift of 31
    int64_t mult = 1 << 30;

    if ((m1 < 0) ^ (m2 < 0))
    {
        mult = 1 - mult;
    }
    // Gets resolved as a SMLAL instruction
    mult = mult + (int64_t)m1 * m2;

    // Utilize all of the upper 32 bits. This is the doubling step
    // as well.
    result = (int32_t)(mult / (1ll << 31));

    if ((m1 == m2) && (m1 == (int32_t)NN_Q31_MIN))
    {
        result = NN_Q31_MAX;
    }
    return result;
}

/**
 * @brief           Doubling high multiply without saturation. This is intended
 *                  for requantization where the scale is a positive integer
 *
 * @param[in]       m1        Multiplicand. Range: {NN_Q31_MIN, NN_Q31_MAX}
 * @param[in]       m2        Multiplier Range: {NN_Q31_MIN, NN_Q31_MAX}
 * @return          Result of multiplication.
 * @note            The result of this matches that of neon instruction
 *                  VQRDMULH for m1 in range {NN_Q31_MIN, NN_Q31_MAX} and m2 in
 *                  range {NN_Q31_MIN + 1, NN_Q31_MAX}. Saturation occurs when
 *                  m1 equals m2 equals NN_Q31_MIN and that is not handled by
 *                  this function.
 *
 */
__STATIC_FORCEINLINE int32_t arm_nn_doubling_high_mult_no_sat(const int32_t m1, const int32_t m2)
{
    int32_t result = 0;
    union arm_nn_long_long mult;

    // Rounding offset to add for a right shift of 31
    mult.word.low = 1 << 30;
    mult.word.high = 0;

    // Gets resolved as a SMLAL instruction
    mult.long_long = mult.long_long + (int64_t)m1 * m2;

    // Utilize all of the upper 32 bits. This is the doubling step
    // as well.
    result = (int32_t)(mult.long_long >> 31);

    return result;
}

/**
 * @brief           Rounding divide by power of two.
 * @param[in]       dividend - Dividend
 * @param[in]       exponent - Divisor = power(2, exponent)
 *                             Range: [0, 31]
 * @return          Rounded result of division. Midpoint is rounded away from zero.
 *
 */
__STATIC_FORCEINLINE int32_t arm_nn_divide_by_power_of_two(const int32_t dividend, const int32_t exponent)
{
    int32_t result = 0;
    const int32_t remainder_mask = (1 << exponent) - 1;
    int32_t remainder = remainder_mask & dividend;

    // Basic division
    result = dividend >> exponent;

    // Adjust 'result' for rounding (mid point away from zero)
    int32_t threshold = remainder_mask >> 1;
    if (result < 0)
    {
        threshold++;
    }
    if (remainder > threshold)
    {
        result++;
    }

    return result;
}

/**
 * @brief           Requantize a given value.
 * @param[in]       val         Value to be requantized
 * @param[in]       multiplier  multiplier. Range {NN_Q31_MIN + 1, Q32_MAX}
 * @param[in]       shift       left or right shift for 'val * multiplier'
 *
 * @return          Returns (val * multiplier)/(2 ^ shift)
 *
 */
__STATIC_FORCEINLINE int32_t arm_nn_requantize(const int32_t val, const int32_t multiplier, const int32_t shift)
{
#ifdef CMSIS_NN_USE_SINGLE_ROUNDING
    const int64_t total_shift = 31 - shift;
    const int64_t new_val = val * (int64_t)multiplier;

    int32_t result = new_val >> (total_shift - 1);
    result = (result + 1) >> 1;

    return result;
#else
    return arm_nn_divide_by_power_of_two(arm_nn_doubling_high_mult_no_sat(val * (1 << LEFT_SHIFT(shift)), multiplier),
                                         RIGHT_SHIFT(shift));
#endif
}

/**
 * @brief           Requantize a given 64 bit value.
 * @param[in]       val                 Value to be requantized in the range {-(1<<47)} to {(1<<47) - 1}
 * @param[in]       reduced_multiplier  Reduced multiplier in the range {NN_Q31_MIN + 1, Q32_MAX} to {Q16_MIN + 1,
 * Q16_MAX}
 * @param[in]       shift               Left or right shift for 'val * multiplier' in the range {-31} to {7}
 *
 * @return          Returns (val * multiplier)/(2 ^ shift)
 *
 */
__STATIC_FORCEINLINE int32_t arm_nn_requantize_s64(const int64_t val,
                                                   const int32_t reduced_multiplier,
                                                   const int32_t shift)
{
    const int64_t new_val = val * reduced_multiplier;

    int32_t result = new_val >> (14 - shift); // 64->32 bit reduction
    result = (result + 1) >> 1;               // Last shift position and insert round

    return result;
}

/**
 * @brief           memcpy optimized for MVE
 * @param[in, out]  dst         Destination pointer
 * @param[in]       src         Source pointer.
 * @param[in]       block_size  Number of bytes to copy.
 *
 */
__STATIC_FORCEINLINE void arm_memcpy_s8(int8_t *__RESTRICT dst, const int8_t *__RESTRICT src, uint32_t block_size)
{
#if defined(ARM_MATH_MVEI)
    __asm volatile("   wlstp.8                 lr, %[cnt], 1f             \n"
                   "2:                                                    \n"
                   "   vldrb.8                 q0, [%[in]], #16            \n"
                   "   vstrb.8                 q0, [%[out]], #16           \n"
                   "   letp                    lr, 2b                     \n"
                   "1:                                                    \n"
                   : [in] "+r"(src), [out] "+r"(dst)
                   : [cnt] "r"(block_size)
                   : "q0", "memory", "r14");
#else
    memcpy(dst, src, block_size);
#endif
}

/**
 * @brief           memcpy wrapper for int16
 * @param[in, out]  dst         Destination pointer
 * @param[in]       src         Source pointer.
 * @param[in]       block_size  Number of bytes to copy.
 *
 */
__STATIC_FORCEINLINE void arm_memcpy_q15(int16_t *__RESTRICT dst, const int16_t *__RESTRICT src, uint32_t block_size)
{
    memcpy(dst, src, block_size);
}

#if defined(ARM_MATH_MVEI)
/**
 * @brief           Vector saturating doubling high multiply returning high half.
 * @param[in]       m1        Multiplicand
 * @param[in]       m2        Multiplier
 * @return          Result of multiplication.
 *
 */
__STATIC_FORCEINLINE int32x4_t arm_doubling_high_mult_mve(const int32x4_t m1, const int32_t m2)
{
    return vqrdmulhq_n_s32(m1, m2);
}

/**
 * @brief           Vector rounding divide by power of two.
 * @param[in]       dividend - Dividend vector
 * @param[in]       exponent - Divisor = power(2, exponent)
 *                             Range: [0, 31]
 * @return          Rounded result of division. Midpoint is rounded away from zero.
 *
 */
__STATIC_FORCEINLINE int32x4_t arm_divide_by_power_of_two_mve(const int32x4_t dividend, const int32_t exponent)
{
    const int32x4_t shift = vdupq_n_s32(-exponent);
    const int32x4_t fixup = vshrq_n_s32(vandq_s32(dividend, shift), 31);
    const int32x4_t fixed_up_dividend = vqaddq_s32(dividend, fixup);
    return vrshlq_s32(fixed_up_dividend, shift);
}

/**
 * @brief           Requantize a given vector.
 * @param[in]       val         Vector to be requantized
 * @param[in]       multiplier  multiplier
 * @param[in]       shift       shift
 *
 * @return          Returns (val * multiplier)/(2 ^ shift)
 *
 */
__STATIC_FORCEINLINE int32x4_t arm_requantize_mve(const int32x4_t val, const int32_t multiplier, const int32_t shift)
{
    #ifdef CMSIS_NN_USE_SINGLE_ROUNDING
    const int right_shift = MIN(-1, shift);
    const int left_shift = shift - right_shift;

    const int32x4_t left_shift_dup = vdupq_n_s32(left_shift);
    const int32x4_t right_shift_dup = vdupq_n_s32(right_shift);

    int32x4_t result = vqdmulhq_n_s32(vshlq_s32(val, left_shift_dup), multiplier);
    result = vrshlq_s32(result, right_shift_dup);

    return result;
    #else
    return arm_divide_by_power_of_two_mve(
        arm_doubling_high_mult_mve(vshlq_s32(val, vdupq_n_s32(LEFT_SHIFT(shift))), multiplier), RIGHT_SHIFT(shift));
    #endif
}

__STATIC_FORCEINLINE int32x4_t arm_doubling_high_mult_mve_32x4(const int32x4_t m1, const int32x4_t m2)
{
    return vqrdmulhq_s32(m1, m2);
}

__STATIC_FORCEINLINE int32x4_t arm_divide_by_power_of_two_mve_32x4(const int32x4_t dividend, const int32x4_t exponent)
{
    const int32x4_t shift = -exponent;
    const int32x4_t fixup = vshrq_n_s32(vandq_s32(dividend, shift), 31);
    const int32x4_t fixed_up_dividend = vqaddq_s32(dividend, fixup);
    return vrshlq_s32(fixed_up_dividend, shift);
}

__STATIC_FORCEINLINE int32x4_t arm_requantize_mve_32x4(const int32x4_t val,
                                                       const int32x4_t multiplier,
                                                       const int32x4_t shift)
{
    #ifdef CMSIS_NN_USE_SINGLE_ROUNDING
    const int32x4_t right_shift = vminq_s32(vdupq_n_s32(-1), shift);
    const int32x4_t left_shift = vqsubq_s32(shift, right_shift);

    int32x4_t result = vqdmulhq_s32(vshlq_s32(val, left_shift), multiplier);
    result = vrshlq_s32(result, right_shift);

    return result;
    #else
    const int32x4_t zz = vdupq_n_s32(0);
    const mve_pred16_t p = vcmpgtq_n_s32(shift, 0);

    const int32x4_t left_shift = vpselq_s32(shift, zz, p);
    const int32x4_t right_shift = -vpselq_s32(zz, shift, p);

    return arm_divide_by_power_of_two_mve_32x4(arm_doubling_high_mult_mve_32x4(vshlq_s32(val, left_shift), multiplier),
                                               right_shift);
    #endif
}
#endif

// @note The following functions are used only for softmax layer, scaled bits = 5 assumed

__STATIC_FORCEINLINE int32_t arm_nn_exp_on_negative_values(int32_t val)
{
    int32_t mask = 0;
    int32_t shift = 24;

    const int32_t val_mod_minus_quarter = (val & ((1 << shift) - 1)) - (1 << shift);
    const int32_t remainder = val_mod_minus_quarter - val;
    const int32_t x = (val_mod_minus_quarter << 5) + (1 << 28);
    const int32_t x2 = MUL_SAT(x, x);

    int32_t result = 1895147668 +
        MUL_SAT(1895147668, x + DIV_POW2(MUL_SAT(DIV_POW2(MUL_SAT(x2, x2), 2) + MUL_SAT(x2, x), 715827883) + x2, 1));

#define SELECT_IF_NON_ZERO(x)                                                                                          \
    {                                                                                                                  \
        mask = MASK_IF_NON_ZERO(remainder & (1 << shift++));                                                           \
        result = SELECT_USING_MASK(mask, MUL_SAT(result, x), result);                                                  \
    }

    SELECT_IF_NON_ZERO(1672461947)
    SELECT_IF_NON_ZERO(1302514674)
    SELECT_IF_NON_ZERO(790015084)
    SELECT_IF_NON_ZERO(290630308)
    SELECT_IF_NON_ZERO(39332535)
    SELECT_IF_NON_ZERO(720401)
    SELECT_IF_NON_ZERO(242)

#undef SELECT_IF_NON_ZERO

    mask = MASK_IF_ZERO(val);
    return SELECT_USING_MASK(mask, NN_Q31_MAX, result);
}

__STATIC_FORCEINLINE int32_t arm_nn_mult_by_power_of_two(const int32_t val, const int32_t exp)
{
    const int32_t thresh = ((1 << (31 - exp)) - 1);
    int32_t result = val << exp;
    result = SELECT_USING_MASK(MASK_IF_NON_ZERO(val > thresh), NN_Q31_MAX, result);
    result = SELECT_USING_MASK(MASK_IF_NON_ZERO(val < -thresh), NN_Q31_MIN, result);
    return result;
}

__STATIC_FORCEINLINE int32_t arm_nn_one_over_one_plus_x_for_x_in_0_1(int32_t val)
{
    const int64_t sum = (int64_t)val + (int64_t)NN_Q31_MAX;
    const int32_t half_denominator = (int32_t)((sum + (sum >= 0 ? 1 : -1)) / 2L);
    int32_t x = 1515870810 + MUL_SAT(half_denominator, -1010580540);

    const int32_t shift = (1 << 29);
    x += MUL_POW2(MUL_SAT(x, shift - MUL_SAT(half_denominator, x)), 2);
    x += MUL_POW2(MUL_SAT(x, shift - MUL_SAT(half_denominator, x)), 2);
    x += MUL_POW2(MUL_SAT(x, shift - MUL_SAT(half_denominator, x)), 2);

    return MUL_POW2(x, 1);
}

/**
  @brief         Write 2 s16 elements and post increment pointer.
  @param[in]     dest_q15  Pointer to pointer that holds address of destination.
  @param[in]     src_q31   Input value to be written.
 */
__STATIC_FORCEINLINE void arm_nn_write_q15x2_ia(int16_t **dest_q15, int32_t src_q31)
{
    int32_t val = src_q31;

    memcpy(*dest_q15, &val, 4);
    *dest_q15 += 2;
}

/**
  @brief         Write 2 s8 elements and post increment pointer.
  @param[in]     dst  Pointer to pointer that holds address of destination.
  @param[in]     src  Input value to be written.
 */
__STATIC_FORCEINLINE void arm_nn_write_s8x2_ia(int8_t **dst, int16_t src)
{
    memcpy(*dst, &src, 2);
    *dst += 2;
}

// Support functions for LSTM
/**
 * @brief Update LSTM function for an iteration step
 *
 * param[in]    input                           Input data
 * param[in]    input_to_input_weight           Input to input gate weights
 * param[in]    input_to_forget_weight          Input to forget gate weights
 * param[in]    input_to_cell_weight            Input to cell gate weights
 * param[in]    input_to_output_weight          Input to output weights
 * param[in]    recurrent_to_input_weight       Recurrent signal to input weights
 * param[in]    recurrent_to_forget_weight      Recurrent signal to forget gate weights
 * param[in]    recurrent_to_cell_weight        Recurrent signal to cell gate weighst
 * param[in]    recurrent_to_output_weight      Recurrent signal to output weights
 * param[in]    lstm                            LSTM parameters
 * param[in]    n_batch                         Batch size
 * param[in]    n_cell                          Cell size
 * param[in]    n_input                         Input size
 * param[in]    n_output                        Output size
 * param[out]   output_state                    Output state
 * param[out]   cell_state                      Internal state
 * param[out]   output                          Output signal
 * param[in] *scratch_buffers                   Struct containing scratch buffers
 */
arm_cmsis_nn_status arm_nn_lstm_step_s8_s16(const int8_t *input,
                                            const int8_t *input_to_input_weight,
                                            const int8_t *input_to_forget_weight,
                                            const int8_t *input_to_cell_weight,
                                            const int8_t *input_to_output_weight,
                                            const int8_t *recurrent_to_input_weight,
                                            const int8_t *recurrent_to_forget_weight,
                                            const int8_t *recurrent_to_cell_weight,
                                            const int8_t *recurrent_to_output_weight,
                                            const cmsis_nn_lstm_params *lstm,
                                            const int n_batch,
                                            const int n_cell,
                                            const int n_input,
                                            const int n_output,
                                            int8_t *output_state,
                                            int16_t *cell_state,
                                            int8_t *output,
                                            cmsis_nn_lstm_context *scratch_buffers);

/**
 * @brief         Updates a LSTM gate for an iteration step of LSTM function, int8x8_16 version.
 *
 * param[in]    input                           Input data
 * param[in]    input_to_gate_weights           Input to gate weights
 * param[in]    input_to_gate_bias              Input to gate weights
 * param[in]    input_to_gate_scaling           Input to gate scaling
 * param[in]    activation                      Actival min and max values
 * param[in]    output_state                    Output state
 * param[in]    recurrent_to_gate_weights       Recurrent to gate weights
 * param[in]    recurrent_to_gate_bias          Recurrent to gate bias
 * param[in]    recurrent_to_gate_scaling       Recurrent to gate scaling
 * param[in]    n_batch                         Batch size
 * param[in]    n_input                         Input size
 * param[out]   n_output                        Output size
 * param[in]    activation_type                 Activation type (sigmoid or tanh)
 * param[out]   n_cell                          Cell size
 */
void arm_nn_lstm_calculate_gate_s8_s16(const int8_t *input,
                                       const int8_t *input_to_gate_weights,
                                       const int32_t *input_to_gate_bias,
                                       const cmsis_nn_scaling input_to_gate_scaling,
                                       const int8_t *output_state,
                                       const int8_t *recurrent_to_gate_weights,
                                       const int32_t *recurrent_to_gate_bias,
                                       const cmsis_nn_scaling recurrent_to_gate_scaling,
                                       const int32_t n_batch,
                                       const int32_t n_input,
                                       const int32_t n_output,
                                       const int32_t n_cell,
                                       const arm_nn_activation_type activation_type,
                                       int16_t *gate);

/**
 * @brief       Update cell state for a single LSTM iteration step, int8x8_16 version.
 * @param[in]   n_block             total number of cells for all batches
 * @param[in]   cell_state_scale    Scaling factor of cell state
 * @param[in]   cell_state          Input/output vector, size n_batch*n_cell
 * @param[in]   input_gate          Input vector of size n_block
 * @param[in]   forget_gate         Input/scratch vector of size n_block, always modified
 * @param[in]   cell_gate           Input vector of size, n_block
 */
void arm_nn_lstm_update_cell_state_s16(const int32_t n_block,
                                       const int32_t cell_state_scale,
                                       int16_t *cell_state,
                                       const int16_t *input_gate,
                                       const int16_t *forget_gate,
                                       const int16_t *cell_gate);

/**
 * @brief       Calculate the output state tensor of an LSTM step, s8 input/output and s16 weight version.
 *
 * @param[in]       n_batch                     The number of distinct vectors in each array
 * @param[in]       n_cell                      Number of cells
 * @param[in,out]   cell_state                  Cell state, size n_batch*n_cell
 * @param[in]       cell_state_scale            Scaling of cell_state
 * @param[in]       output_gate                 Output gate
 * @param[in]       hidden_scale                Effective scaling of cell_state .* output_gate
 * @param[in]       hidden_offset               Zero point for cell_state .* output_gate
 * @param[out]      output_state                Output state
 * @param[in]       cell_gate_scratch           Scratch buffer
 */
void arm_nn_lstm_update_output_s8_s16(const int n_batch,
                                      const int n_cell,
                                      int16_t *cell_state,
                                      const int32_t cell_state_scale,
                                      const int16_t *output_gate,
                                      const cmsis_nn_scaling hidden_scale,
                                      const int32_t hidden_offset,
                                      int8_t *output_state,
                                      int16_t *cell_gate_scratch);

/**
 * @brief The result of the multiplication is accumulated to the passed result buffer.
 * Multiplies a matrix by a "batched" vector (i.e. a matrix with a batch dimension composed by input vectors independent
 * from each other).
 *
 * @param[in]   lhs_in           Batched vector
 * @param[in]   rhs_in           Weights - input matrix (H(Rows)xW(Columns))
 * @param[in]   bias             Bias vector
 * @param[out]  dst              Output
 * @param[in]   dst_offset       Output offset
 * @param[in]   dst_multiplier   Multiplier for quantization
 * @param[in]   dst_shift        Shift for quantization
 * @param[in]   rhs_cols         Vector/matarix column length
 * @param[in]   rhs_rows         Row count of matrix
 * @param[in]   batch            Batch size
 */
void arm_nn_vec_mat_mul_result_acc_s8(const int8_t *lhs_in,
                                      const int8_t *rhs_in,
                                      const int32_t *bias,
                                      int16_t *dst,
                                      const int32_t dst_offset,
                                      const int32_t dst_multiplier,
                                      const int32_t dst_shift,
                                      const int32_t rhs_cols,
                                      const int32_t rhs_rows,
                                      const int32_t batch);

/**
 * @brief s16 elementwise multiplication with s8 output
 * @param[in]       input_1_vect        pointer to input vector 1
 * @param[in]       input_2_vect        pointer to input vector 2
 * @param[in,out]   output              pointer to output vector
 * @param[in]       out_offset          output offset
 * @param[in]       out_mult            output multiplier
 * @param[in]       out_shift           output shift
 * @param[in]       block_size          number of samples
 * @return          The function returns ARM_CMSIS_NN_SUCCESS
 *
 * @details   Supported framework: TensorFlow Lite micro
 */
arm_cmsis_nn_status arm_elementwise_mul_s16_s8(const int16_t *input_1_vect,
                                               const int16_t *input_2_vect,
                                               int8_t *output,
                                               const int32_t out_offset,
                                               const int32_t out_mult,
                                               const int32_t out_shift,
                                               const int32_t block_size);

#ifdef __cplusplus
}
#endif

#endif
