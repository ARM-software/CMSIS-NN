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
class DsCnnSBodyCaseFlt:
    dataset: str
    dtype_name: str
    seed: int
    activation_max: float
    input_batches: int = 1
    input_h: int = 25
    input_w: int = 5
    channels: int = 64
    num_blocks: int = 4
    num_classes: int = 12
    input_min: float = -1.0
    input_max: float = 1.0
    depthwise_weight_min: float = -0.25
    depthwise_weight_max: float = 0.25
    pointwise_weight_min: float = -0.12
    pointwise_weight_max: float = 0.12
    fc_weight_min: float = -0.2
    fc_weight_max: float = 0.2
    bias_min: float = -0.05
    bias_max: float = 0.05

    @property
    def input_shape_nchw(self) -> tuple[int, int, int, int]:
        return (self.input_batches, self.channels, self.input_h, self.input_w)

    @property
    def feature_map_size(self) -> int:
        return self.input_batches * self.input_h * self.input_w * self.channels


class DsCnnSBodySettingsFlt:
    def __init__(self, case: DsCnnSBodyCaseFlt):
        self.case = case
        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.case.dataset,
            Path("TestCases/TestData"),
            Path("PregeneratedData"),
        )
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _pregenerated_path(self, stem: str) -> Path:
        return self.pregenerated_data_dir / f"{stem}.txt"

    def _load_or_create_array(
        self,
        stem: str,
        shape: tuple[int, ...],
        low: float,
        high: float,
        rng: np.random.Generator,
        regenerate_input: bool,
    ) -> np.ndarray:
        filepath = self._pregenerated_path(stem)
        if filepath.exists() and not regenerate_input:
            return load_pregenerated_array(filepath, shape)

        values = rng.uniform(low, high, size=shape).astype(np.float32)
        save_pregenerated_array(filepath, values)
        return values

    def _load_or_create_tensors(self, regenerate_input: bool) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.case.seed)
        arrays: dict[str, np.ndarray] = {}

        arrays["input"] = self._load_or_create_array(
            "input",
            self.case.input_shape_nchw,
            self.case.input_min,
            self.case.input_max,
            rng,
            regenerate_input,
        )

        for block_idx in range(self.case.num_blocks):
            arrays[f"dwconv{block_idx}_weights"] = self._load_or_create_array(
                f"dwconv{block_idx}_weights",
                (self.case.channels, 1, 3, 3),
                self.case.depthwise_weight_min,
                self.case.depthwise_weight_max,
                rng,
                regenerate_input,
            )
            arrays[f"dwconv{block_idx}_biases"] = self._load_or_create_array(
                f"dwconv{block_idx}_biases",
                (self.case.channels,),
                self.case.bias_min,
                self.case.bias_max,
                rng,
                regenerate_input,
            )
            arrays[f"pwconv{block_idx}_weights"] = self._load_or_create_array(
                f"pwconv{block_idx}_weights",
                (self.case.channels, self.case.channels, 1, 1),
                self.case.pointwise_weight_min,
                self.case.pointwise_weight_max,
                rng,
                regenerate_input,
            )
            arrays[f"pwconv{block_idx}_biases"] = self._load_or_create_array(
                f"pwconv{block_idx}_biases",
                (self.case.channels,),
                self.case.bias_min,
                self.case.bias_max,
                rng,
                regenerate_input,
            )

        arrays["fc_weights"] = self._load_or_create_array(
            "fc_weights",
            (self.case.num_classes, self.case.channels),
            self.case.fc_weight_min,
            self.case.fc_weight_max,
            rng,
            regenerate_input,
        )
        arrays["fc_biases"] = self._load_or_create_array(
            "fc_biases",
            (self.case.num_classes,),
            self.case.bias_min,
            self.case.bias_max,
            rng,
            regenerate_input,
        )
        return arrays

    def _run_reference(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        # Mirror the ds_cnn_s body topology used by the float manifest tests:
        # 4x [depthwise 3x3 + relu, pointwise 1x1 + relu], avgpool, fc, softmax.
        x = torch.tensor(arrays["input"], dtype=torch.float32)
        for block_idx in range(self.case.num_blocks):
            x = F.conv2d(
                x,
                torch.tensor(arrays[f"dwconv{block_idx}_weights"], dtype=torch.float32),
                torch.tensor(arrays[f"dwconv{block_idx}_biases"], dtype=torch.float32),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=self.case.channels,
            )
            x = F.relu(x)
            x = F.conv2d(
                x,
                torch.tensor(arrays[f"pwconv{block_idx}_weights"], dtype=torch.float32),
                torch.tensor(arrays[f"pwconv{block_idx}_biases"], dtype=torch.float32),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
            )
            x = F.relu(x)

        x = F.avg_pool2d(
            x,
            kernel_size=(self.case.input_h, self.case.input_w),
            stride=(1, 1),
            padding=(0, 0),
            count_include_pad=False,
        )
        x = x.reshape(self.case.input_batches, self.case.channels)
        x = F.linear(
            x,
            torch.tensor(arrays["fc_weights"], dtype=torch.float32),
            torch.tensor(arrays["fc_biases"], dtype=torch.float32),
        )
        x = torch.softmax(x, dim=-1)
        return x.cpu().numpy().astype(np.float32).reshape(-1)

    def _export_input(self, input_nchw: np.ndarray) -> np.ndarray:
        return np.transpose(input_nchw, (0, 2, 3, 1))

    def _export_depthwise_weights(self, weights_oc1hw: np.ndarray) -> np.ndarray:
        return np.transpose(np.squeeze(weights_oc1hw, axis=1), (1, 2, 0))

    def _export_pointwise_weights(self, weights_oihw: np.ndarray) -> np.ndarray:
        return np.transpose(weights_oihw, (0, 2, 3, 1))

    def _write_config_header(self) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")
        formatter = format_float32 if self.case.dtype_name == "f32" else format_float16
        identity_min = -self.case.activation_max

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_INPUT_BATCHES {self.case.input_batches}\n")
            fh.write(f"#define {prefix}_INPUT_H {self.case.input_h}\n")
            fh.write(f"#define {prefix}_INPUT_W {self.case.input_w}\n")
            fh.write(f"#define {prefix}_INPUT_C {self.case.channels}\n")
            fh.write(f"#define {prefix}_CHANNELS {self.case.channels}\n")
            fh.write(f"#define {prefix}_NUM_BLOCKS {self.case.num_blocks}\n")
            fh.write(f"#define {prefix}_NUM_CLASSES {self.case.num_classes}\n")
            fh.write(f"#define {prefix}_FEATURE_MAP_SIZE {self.case.feature_map_size}\n")
            fh.write(f"#define {prefix}_POOLED_SIZE {self.case.channels}\n")
            fh.write(f"#define {prefix}_DST_SIZE {self.case.num_classes}\n")
            fh.write(f"#define {prefix}_SOFTMAX_ROWS 1\n")
            fh.write(f"#define {prefix}_SOFTMAX_COLS {self.case.num_classes}\n")
            fh.write(f"#define {prefix}_LAYOUT ARM_NN_LAYOUT_NHWC\n")
            fh.write(f"#define {prefix}_DEPTHWISE_CH_MULT 1\n")
            fh.write(f"#define {prefix}_DEPTHWISE_FILTER_H 3\n")
            fh.write(f"#define {prefix}_DEPTHWISE_FILTER_W 3\n")
            fh.write(f"#define {prefix}_DEPTHWISE_STRIDE_H 1\n")
            fh.write(f"#define {prefix}_DEPTHWISE_STRIDE_W 1\n")
            fh.write(f"#define {prefix}_DEPTHWISE_PADDING_H 1\n")
            fh.write(f"#define {prefix}_DEPTHWISE_PADDING_W 1\n")
            fh.write(f"#define {prefix}_DEPTHWISE_DILATION_H 1\n")
            fh.write(f"#define {prefix}_DEPTHWISE_DILATION_W 1\n")
            fh.write(f"#define {prefix}_POINTWISE_FILTER_H 1\n")
            fh.write(f"#define {prefix}_POINTWISE_FILTER_W 1\n")
            fh.write(f"#define {prefix}_POINTWISE_STRIDE_H 1\n")
            fh.write(f"#define {prefix}_POINTWISE_STRIDE_W 1\n")
            fh.write(f"#define {prefix}_POINTWISE_PADDING_H 0\n")
            fh.write(f"#define {prefix}_POINTWISE_PADDING_W 0\n")
            fh.write(f"#define {prefix}_POINTWISE_DILATION_H 1\n")
            fh.write(f"#define {prefix}_POINTWISE_DILATION_W 1\n")
            fh.write(f"#define {prefix}_AVGPOOL_FILTER_H {self.case.input_h}\n")
            fh.write(f"#define {prefix}_AVGPOOL_FILTER_W {self.case.input_w}\n")
            fh.write(f"#define {prefix}_AVGPOOL_OUTPUT_H 1\n")
            fh.write(f"#define {prefix}_AVGPOOL_OUTPUT_W 1\n")
            fh.write(f"#define {prefix}_RELU_MIN {formatter(0.0)}\n")
            fh.write(f"#define {prefix}_RELU_MAX {formatter(self.case.activation_max)}\n")
            fh.write(f"#define {prefix}_IDENTITY_MIN {formatter(identity_min)}\n")
            fh.write(f"#define {prefix}_IDENTITY_MAX {formatter(self.case.activation_max)}\n")
        format_generated_file(filepath)

    def _write_array_header(self, filename: str, symbol_name: str, values: np.ndarray) -> None:
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, symbol_name, values, self.case.dtype_name)

    def _write_headers(self, arrays: dict[str, np.ndarray], output_ref: np.ndarray) -> None:
        exported_input = self._export_input(arrays["input"])
        if self.case.dtype_name == "f16":
            exported_input = exported_input.astype(np.float16)
            output_ref = output_ref.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header()
        self._write_array_header("input_data.h", f"{self.case.dataset}_input_data", exported_input)

        for block_idx in range(self.case.num_blocks):
            dw_weights = self._export_depthwise_weights(arrays[f"dwconv{block_idx}_weights"])
            pw_weights = self._export_pointwise_weights(arrays[f"pwconv{block_idx}_weights"])
            dw_biases = arrays[f"dwconv{block_idx}_biases"]
            pw_biases = arrays[f"pwconv{block_idx}_biases"]
            if self.case.dtype_name == "f16":
                dw_weights = dw_weights.astype(np.float16)
                pw_weights = pw_weights.astype(np.float16)
                dw_biases = dw_biases.astype(np.float16)
                pw_biases = pw_biases.astype(np.float16)

            self._write_array_header(
                f"dwconv{block_idx}_weights_data.h",
                f"{self.case.dataset}_dwconv{block_idx}_weights_data",
                dw_weights,
            )
            self._write_array_header(
                f"dwconv{block_idx}_biases_data.h",
                f"{self.case.dataset}_dwconv{block_idx}_biases_data",
                dw_biases,
            )
            self._write_array_header(
                f"pwconv{block_idx}_weights_data.h",
                f"{self.case.dataset}_pwconv{block_idx}_weights_data",
                pw_weights,
            )
            self._write_array_header(
                f"pwconv{block_idx}_biases_data.h",
                f"{self.case.dataset}_pwconv{block_idx}_biases_data",
                pw_biases,
            )

        fc_weights = arrays["fc_weights"]
        fc_biases = arrays["fc_biases"]
        if self.case.dtype_name == "f16":
            fc_weights = fc_weights.astype(np.float16)
            fc_biases = fc_biases.astype(np.float16)

        self._write_array_header("fc_weights_data.h", f"{self.case.dataset}_fc_weights_data", fc_weights)
        self._write_array_header("fc_biases_data.h", f"{self.case.dataset}_fc_biases_data", fc_biases)
        self._write_array_header("output_ref_data.h", f"{self.case.dataset}_output_ref_data", output_ref)
        write_wrapper_header(self.headers_dir / "test_data.h", self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        arrays = self._load_or_create_tensors(regenerate_input)
        output_ref = self._run_reference(arrays)
        self._write_headers(arrays, output_ref)


def _build_cases() -> list[DsCnnSBodyCaseFlt]:
    return [
        DsCnnSBodyCaseFlt(
            dataset="ds_cnn_s_body_f32",
            dtype_name="f32",
            seed=9101,
            activation_max=3.4028234663852886e38,
        ),
        DsCnnSBodyCaseFlt(
            dataset="ds_cnn_s_body_f16",
            dtype_name="f16",
            seed=9101,
            activation_max=65504.0,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DS-CNN-S body float references using PyTorch.")
    parser.add_argument(
        "--dataset",
        choices=[
            "ds_cnn_s_body_f32",
            "ds_cnn_s_body_f16",
            "ds_cnn_s_body_f32_family",
            "ds_cnn_s_body_f16_family",
            "all",
        ],
        default="all",
        help="Dataset or family selector to generate.",
    )
    parser.add_argument(
        "--regenerate-input",
        action="store_true",
        help="Regenerate the pregenerated DS-CNN-S body tensors instead of reusing them.",
    )
    return parser.parse_args()


def _matches_selector(case: DsCnnSBodyCaseFlt, selector: str) -> bool:
    if selector == "all":
        return True
    if selector == case.dataset:
        return True
    if selector == "ds_cnn_s_body_f32_family":
        return case.dtype_name == "f32"
    if selector == "ds_cnn_s_body_f16_family":
        return case.dtype_name == "f16"
    return False


def main() -> None:
    args = parse_args()
    for case in _build_cases():
        if not _matches_selector(case, args.dataset):
            continue
        print(f"Generating {case.dataset}...")
        DsCnnSBodySettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
