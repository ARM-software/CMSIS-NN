#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from float_test_gen_utils import (
    format_generated_file,
    format_float16,
    format_float32,
    load_pregenerated_array,
    prepare_dataset_dirs,
    save_pregenerated_array,
    write_common_header,
    write_float_array_header,
    write_wrapper_header,
)


@dataclass(frozen=True)
class ConvCaseFlt:
    dataset: str
    dtype_name: str
    layout: str
    use_wrapper: bool
    batches: int
    input_h: int
    input_w: int
    input_channels: int
    output_channels: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    padding_h: int
    padding_w: int
    dilation_h: int
    dilation_w: int
    activation_min: float
    activation_max: float
    seed: int
    input_min: float = -1.0
    input_max: float = 1.0
    weight_min: float = -0.75
    weight_max: float = 0.75
    bias_min: float = -0.5
    bias_max: float = 0.5


class ConvSettingsFlt:
    def __init__(self, case: ConvCaseFlt):
        self.case = case
        self.outdir = Path("TestCases/TestData")
        self.pregenerated_dir = Path("PregeneratedData")

        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset, self.outdir, self.pregenerated_dir
        )

        self.input_table_file = self.pregenerated_data_dir / "input.txt"
        self.weights_table_file = self.pregenerated_data_dir / "weights.txt"
        self.bias_table_file = self.pregenerated_data_dir / "bias.txt"

        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_array(self, filepath: Path, array: np.ndarray) -> None:
        save_pregenerated_array(filepath, array)

    def _load_array(self, filepath: Path, shape: tuple[int, ...]) -> np.ndarray:
        return load_pregenerated_array(filepath, shape)

    def _get_raw_tensors(self, regenerate_input: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_shape = (self.case.batches, self.case.input_channels, self.case.input_h, self.case.input_w)
        weight_shape = (self.case.output_channels, self.case.input_channels, self.case.kernel_h, self.case.kernel_w)
        bias_shape = (self.case.output_channels,)

        if (
            self.input_table_file.exists()
            and self.weights_table_file.exists()
            and self.bias_table_file.exists()
            and not regenerate_input
        ):
            print(f"Loading data from {self.pregenerated_data_dir}")
            return (
                self._load_array(self.input_table_file, input_shape),
                self._load_array(self.weights_table_file, weight_shape),
                self._load_array(self.bias_table_file, bias_shape),
            )

        rng = np.random.default_rng(self.case.seed)
        input_data = rng.uniform(self.case.input_min, self.case.input_max, size=input_shape).astype(np.float32)
        weights = rng.uniform(self.case.weight_min, self.case.weight_max, size=weight_shape).astype(np.float32)
        bias = rng.uniform(self.case.bias_min, self.case.bias_max, size=bias_shape).astype(np.float32)

        print(f"Saving data to {self.pregenerated_data_dir}")
        self._save_array(self.input_table_file, input_data)
        self._save_array(self.weights_table_file, weights)
        self._save_array(self.bias_table_file, bias)
        return input_data, weights, bias

    def _reorder_input(self, input_nchw: np.ndarray) -> np.ndarray:
        return np.transpose(input_nchw, (0, 2, 3, 1))

    def _reorder_output(self, output_nchw: np.ndarray) -> np.ndarray:
        return np.transpose(output_nchw, (0, 2, 3, 1))

    def _reorder_weights(self, weights_oihw: np.ndarray) -> np.ndarray:
        return np.transpose(weights_oihw, (0, 2, 3, 1))

    def _compute_reference(self, input_nchw: np.ndarray, weights_oihw: np.ndarray, bias: np.ndarray) -> np.ndarray:
        input_tensor = torch.tensor(input_nchw, dtype=torch.float32)
        weight_tensor = torch.tensor(weights_oihw, dtype=torch.float32)
        bias_tensor = torch.tensor(bias, dtype=torch.float32)
        output = F.conv2d(
            input_tensor,
            weight_tensor,
            bias_tensor,
            stride=(self.case.stride_h, self.case.stride_w),
            padding=(self.case.padding_h, self.case.padding_w),
            dilation=(self.case.dilation_h, self.case.dilation_w),
        )
        output = torch.clamp(output, min=self.case.activation_min, max=self.case.activation_max)
        return output.cpu().numpy().astype(np.float32)

    def _write_config_header(self, output_h: int, output_w: int) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")

        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16
        layout_macro = "ARM_NN_LAYOUT_NHWC"

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_INPUT_BATCHES {self.case.batches}\n")
            fh.write(f"#define {prefix}_INPUT_H {self.case.input_h}\n")
            fh.write(f"#define {prefix}_INPUT_W {self.case.input_w}\n")
            fh.write(f"#define {prefix}_IN_CH {self.case.input_channels}\n")
            fh.write(f"#define {prefix}_FILTER_H {self.case.kernel_h}\n")
            fh.write(f"#define {prefix}_FILTER_W {self.case.kernel_w}\n")
            fh.write(f"#define {prefix}_OUT_CH {self.case.output_channels}\n")
            fh.write(f"#define {prefix}_STRIDE_H {self.case.stride_h}\n")
            fh.write(f"#define {prefix}_STRIDE_W {self.case.stride_w}\n")
            fh.write(f"#define {prefix}_PADDING_H {self.case.padding_h}\n")
            fh.write(f"#define {prefix}_PADDING_W {self.case.padding_w}\n")
            fh.write(f"#define {prefix}_DILATION_H {self.case.dilation_h}\n")
            fh.write(f"#define {prefix}_DILATION_W {self.case.dilation_w}\n")
            fh.write(f"#define {prefix}_OUTPUT_H {output_h}\n")
            fh.write(f"#define {prefix}_OUTPUT_W {output_w}\n")
            fh.write(f"#define {prefix}_OUTPUT_C {self.case.output_channels}\n")
            fh.write(f"#define {prefix}_DST_SIZE {self.case.batches * output_h * output_w * self.case.output_channels}\n")
            fh.write(f"#define {prefix}_LAYOUT {layout_macro}\n")
            fh.write(f"#define {prefix}_USE_WRAPPER {1 if self.case.use_wrapper else 0}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MIN {formatter(self.case.activation_min)}\n")
            fh.write(f"#define {prefix}_OUT_ACTIVATION_MAX {formatter(self.case.activation_max)}\n")
        format_generated_file(filepath)

    def _write_array_header(self, suffix: str, values: np.ndarray) -> None:
        filename = suffix
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        symbol_suffix = Path(suffix).stem
        write_float_array_header(
            filepath,
            self.script_name,
            f"{self.case.dataset}_{symbol_suffix}",
            values,
            self.case.dtype_name,
        )

    def _write_wrapper_header(self) -> None:
        filepath = self.headers_dir / "test_data.h"
        write_wrapper_header(filepath, self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        input_nchw, weights_oihw, bias = self._get_raw_tensors(regenerate_input)
        output_nchw = self._compute_reference(input_nchw, weights_oihw, bias)
        output_h = output_nchw.shape[2]
        output_w = output_nchw.shape[3]

        input_export = self._reorder_input(input_nchw)
        weights_export = self._reorder_weights(weights_oihw)
        output_export = self._reorder_output(output_nchw)

        if self.case.dtype_name == "f16":
            input_export = input_export.astype(np.float16)
            weights_export = weights_export.astype(np.float16)
            bias = bias.astype(np.float16)
            output_export = output_export.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header(output_h, output_w)
        self._write_array_header("input_data.h", input_export)
        self._write_array_header("weights_data.h", weights_export)
        self._write_array_header("biases_data.h", bias)
        self._write_array_header("output_ref_data.h", output_export)
        self._write_wrapper_header()


def _build_cases() -> list[ConvCaseFlt]:
    base_cases = [
        ConvCaseFlt(
            dataset="conv_basic_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=9,
            input_w=11,
            input_channels=6,
            output_channels=10,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2001,
        ),
        ConvCaseFlt(
            dataset="conv_basic_nhwc_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=9,
            input_w=11,
            input_channels=6,
            output_channels=10,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2001,
        ),
        ConvCaseFlt(
            dataset="conv_k3_opt_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=1,
            input_w=21,
            input_channels=8,
            output_channels=12,
            kernel_h=1,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2002,
        ),
        ConvCaseFlt(
            dataset="conv_k5_opt_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=1,
            input_w=21,
            input_channels=8,
            output_channels=12,
            kernel_h=1,
            kernel_w=5,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2003,
        ),
        ConvCaseFlt(
            dataset="conv_k3_opt_nhwc_tuned_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=1,
            input_w=21,
            input_channels=16,
            output_channels=16,
            kernel_h=1,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2004,
        ),
        ConvCaseFlt(
            dataset="conv_k5_opt_nhwc_tuned_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=1,
            input_w=21,
            input_channels=16,
            output_channels=16,
            kernel_h=1,
            kernel_w=5,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2005,
        ),
        ConvCaseFlt(
            dataset="conv_1x1_stride2_nhwc_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=3,
            input_w=17,
            input_channels=11,
            output_channels=13,
            kernel_h=1,
            kernel_w=1,
            stride_h=1,
            stride_w=2,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-0.75,
            activation_max=0.75,
            seed=2008,
        ),
        ConvCaseFlt(
            dataset="conv_kernel_2x2_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=6,
            input_w=7,
            input_channels=4,
            output_channels=5,
            kernel_h=2,
            kernel_w=2,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2006,
        ),
        ConvCaseFlt(
            dataset="conv_kernel_3x3_pad1_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=3,
            input_w=6,
            input_channels=2,
            output_channels=4,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=1,
            padding_w=1,
            dilation_h=1,
            dilation_w=1,
            activation_min=-1000.0,
            activation_max=1000.0,
            seed=2007,
        ),
        ConvCaseFlt(
            dataset="conv_match_basic_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=8,
            input_w=5,
            input_channels=1,
            output_channels=1,
            kernel_h=4,
            kernel_w=2,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-128.0,
            activation_max=127.0,
            seed=2101,
        ),
        ConvCaseFlt(
            dataset="conv_match_stride2pad1_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=7,
            input_w=7,
            input_channels=1,
            output_channels=1,
            kernel_h=3,
            kernel_w=3,
            stride_h=2,
            stride_w=2,
            padding_h=1,
            padding_w=1,
            dilation_h=1,
            dilation_w=1,
            activation_min=-128.0,
            activation_max=127.0,
            seed=2102,
        ),
        ConvCaseFlt(
            dataset="conv_match_conv_2_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=3,
            input_w=6,
            input_channels=2,
            output_channels=4,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=1,
            padding_w=1,
            dilation_h=1,
            dilation_w=1,
            activation_min=-101.0,
            activation_max=127.0,
            seed=2103,
        ),
        ConvCaseFlt(
            dataset="conv_match_conv_3_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=49,
            input_w=10,
            input_channels=3,
            output_channels=1,
            kernel_h=10,
            kernel_w=4,
            stride_h=2,
            stride_w=1,
            padding_h=4,
            padding_w=1,
            dilation_h=1,
            dilation_w=1,
            activation_min=-127.0,
            activation_max=127.0,
            seed=2104,
        ),
        ConvCaseFlt(
            dataset="conv_match_conv_4_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=3,
            input_h=5,
            input_w=5,
            input_channels=3,
            output_channels=3,
            kernel_h=3,
            kernel_w=2,
            stride_h=2,
            stride_w=2,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-109.0,
            activation_max=127.0,
            seed=2105,
        ),
        ConvCaseFlt(
            dataset="conv_match_conv_5_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=1,
            input_w=128,
            input_channels=128,
            output_channels=1,
            kernel_h=3,
            kernel_w=3,
            stride_h=4,
            stride_w=4,
            padding_h=1,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-88.0,
            activation_max=127.0,
            seed=2106,
        ),
        ConvCaseFlt(
            dataset="conv_match_out_activation_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=3,
            input_w=3,
            input_channels=4,
            output_channels=2,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=1,
            padding_w=1,
            dilation_h=1,
            dilation_w=1,
            activation_min=-61.0,
            activation_max=107.0,
            seed=2107,
            input_min=-8.0,
            input_max=8.0,
            weight_min=-4.0,
            weight_max=4.0,
            bias_min=-20.0,
            bias_max=20.0,
        ),
        ConvCaseFlt(
            dataset="conv_match_dilation_golden_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=2,
            input_h=4,
            input_w=6,
            input_channels=1,
            output_channels=3,
            kernel_h=2,
            kernel_w=2,
            stride_h=1,
            stride_w=1,
            padding_h=1,
            padding_w=1,
            dilation_h=2,
            dilation_w=3,
            activation_min=-128.0,
            activation_max=127.0,
            seed=2108,
        ),
        ConvCaseFlt(
            dataset="conv_match_2x2_dilation_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=10,
            input_w=10,
            input_channels=2,
            output_channels=2,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=2,
            dilation_w=2,
            activation_min=-61.0,
            activation_max=107.0,
            seed=2109,
        ),
        ConvCaseFlt(
            dataset="conv_match_2x3_dilation_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=3,
            input_w=3,
            input_channels=2,
            output_channels=2,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=2,
            padding_w=2,
            dilation_h=2,
            dilation_w=2,
            activation_min=-61.0,
            activation_max=107.0,
            seed=2110,
        ),
        ConvCaseFlt(
            dataset="conv_match_3x2_dilation_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=3,
            input_w=3,
            input_channels=2,
            output_channels=2,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=2,
            padding_w=3,
            dilation_h=2,
            dilation_w=3,
            activation_min=-61.0,
            activation_max=107.0,
            seed=2111,
        ),
        ConvCaseFlt(
            dataset="conv_match_3x3_dilation_5x5_input_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=11,
            input_w=9,
            input_channels=2,
            output_channels=2,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=2,
            padding_w=2,
            dilation_h=2,
            dilation_w=2,
            activation_min=-61.0,
            activation_max=107.0,
            seed=2112,
        ),
        ConvCaseFlt(
            dataset="conv_match_2x2_dilation_5x5_input_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=True,
            batches=1,
            input_h=5,
            input_w=5,
            input_channels=2,
            output_channels=2,
            kernel_h=3,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=2,
            padding_w=2,
            dilation_h=2,
            dilation_w=2,
            activation_min=-61.0,
            activation_max=107.0,
            seed=2113,
        ),
        ConvCaseFlt(
            dataset="conv_match_1x1_basic_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=2,
            input_h=5,
            input_w=7,
            input_channels=19,
            output_channels=7,
            kernel_h=1,
            kernel_w=1,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-126.0,
            activation_max=127.0,
            seed=2114,
        ),
        ConvCaseFlt(
            dataset="conv_match_1x1_stride_x_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=2,
            input_h=4,
            input_w=7,
            input_channels=9,
            output_channels=5,
            kernel_h=1,
            kernel_w=1,
            stride_h=1,
            stride_w=3,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-126.0,
            activation_max=127.0,
            seed=2115,
        ),
        ConvCaseFlt(
            dataset="conv_match_1x1_stride_x_y_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=3,
            input_h=6,
            input_w=7,
            input_channels=23,
            output_channels=15,
            kernel_h=1,
            kernel_w=1,
            stride_h=2,
            stride_w=2,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-6.0,
            activation_max=127.0,
            seed=2116,
        ),
        ConvCaseFlt(
            dataset="conv_match_1x1_stride_x_y_1_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=2,
            input_h=4,
            input_w=4,
            input_channels=5,
            output_channels=5,
            kernel_h=1,
            kernel_w=1,
            stride_h=2,
            stride_w=2,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-126.0,
            activation_max=127.0,
            seed=2117,
        ),
        ConvCaseFlt(
            dataset="conv_match_1x1_stride_x_y_2_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=2,
            input_h=4,
            input_w=4,
            input_channels=5,
            output_channels=5,
            kernel_h=1,
            kernel_w=1,
            stride_h=3,
            stride_w=3,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-126.0,
            activation_max=127.0,
            seed=2118,
        ),
        ConvCaseFlt(
            dataset="conv_match_1xn_1_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=2,
            input_h=1,
            input_w=2,
            input_channels=4,
            output_channels=3,
            kernel_h=1,
            kernel_w=3,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=1,
            dilation_h=1,
            dilation_w=1,
            activation_min=-127.0,
            activation_max=127.0,
            seed=2119,
        ),
        ConvCaseFlt(
            dataset="conv_match_1xn_2_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=1,
            input_w=296,
            input_channels=4,
            output_channels=3,
            kernel_h=1,
            kernel_w=48,
            stride_h=1,
            stride_w=2,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-111.0,
            activation_max=127.0,
            seed=2120,
        ),
        ConvCaseFlt(
            dataset="conv_match_1xn_3_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=1,
            input_w=296,
            input_channels=4,
            output_channels=1,
            kernel_h=1,
            kernel_w=48,
            stride_h=1,
            stride_w=2,
            padding_h=0,
            padding_w=23,
            dilation_h=1,
            dilation_w=1,
            activation_min=-111.0,
            activation_max=127.0,
            seed=2121,
        ),
        ConvCaseFlt(
            dataset="conv_match_1xn_4_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=1,
            input_w=16,
            input_channels=4,
            output_channels=4,
            kernel_h=1,
            kernel_w=3,
            stride_h=1,
            stride_w=2,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-88.0,
            activation_max=127.0,
            seed=2122,
        ),
        ConvCaseFlt(
            dataset="conv_match_1xn_5_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=1,
            input_w=17,
            input_channels=4,
            output_channels=1,
            kernel_h=1,
            kernel_w=3,
            stride_h=1,
            stride_w=3,
            padding_h=0,
            padding_w=0,
            dilation_h=1,
            dilation_w=1,
            activation_min=-88.0,
            activation_max=127.0,
            seed=2123,
        ),
        ConvCaseFlt(
            dataset="conv_match_1xn_6_generic_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=1,
            input_h=1,
            input_w=4,
            input_channels=1,
            output_channels=16,
            kernel_h=1,
            kernel_w=8,
            stride_h=1,
            stride_w=4,
            padding_h=0,
            padding_w=2,
            dilation_h=1,
            dilation_w=1,
            activation_min=-125.0,
            activation_max=126.0,
            seed=2124,
        ),
        ConvCaseFlt(
            dataset="conv_match_1xn_7_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=2,
            input_h=1,
            input_w=148,
            input_channels=20,
            output_channels=15,
            kernel_h=1,
            kernel_w=32,
            stride_h=1,
            stride_w=1,
            padding_h=0,
            padding_w=15,
            dilation_h=1,
            dilation_w=1,
            activation_min=-127.0,
            activation_max=127.0,
            seed=2125,
        ),
        ConvCaseFlt(
            dataset="conv_match_1xn_8_f32",
            dtype_name="f32",
            layout="nhwc",
            use_wrapper=False,
            batches=2,
            input_h=1,
            input_w=148,
            input_channels=12,
            output_channels=10,
            kernel_h=1,
            kernel_w=48,
            stride_h=1,
            stride_w=2,
            padding_h=0,
            padding_w=23,
            dilation_h=1,
            dilation_w=1,
            activation_min=-127.0,
            activation_max=127.0,
            seed=2126,
        ),
    ]

    f16_cases = [
        ConvCaseFlt(**{**case.__dict__, "dataset": case.dataset.replace("_f32", "_f16"), "dtype_name": "f16"})
        for case in base_cases
    ]
    return base_cases + f16_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float convolution unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, conv_f32_family, "
            "conv_f16_family, or a specific dataset name such as conv_basic_nhwc_f16."
        ),
    )
    parser.add_argument(
        "--regenerate-input",
        action="store_true",
        help="Regenerate the pregenerated raw input/weight/bias tensors instead of reusing them.",
    )
    return parser.parse_args()


def _select_cases(all_cases: list[ConvCaseFlt], selector: str) -> list[ConvCaseFlt]:
    if selector == "all":
        return all_cases
    if selector == "conv_f32_family":
        return [case for case in all_cases if case.dtype_name == "f32"]
    if selector == "conv_f16_family":
        return [case for case in all_cases if case.dtype_name == "f16"]
    return [case for case in all_cases if case.dataset == selector]


def main() -> None:
    args = parse_args()
    selected = _select_cases(_build_cases(), args.dataset)
    if not selected:
        raise SystemExit(f"Unknown dataset selector: {args.dataset}")

    for case in selected:
        print(f"Generating {case.dataset}...")
        ConvSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
