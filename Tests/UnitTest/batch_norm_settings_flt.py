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
import torch.nn as nn
from float_test_gen_utils import (
    format_generated_file,
    load_pregenerated_array,
    prepare_dataset_dirs,
    save_pregenerated_array,
    write_common_header,
    write_float_array_header,
    write_wrapper_header,
)


LAYOUT_VALUE = {"nhwc": "ARM_NN_LAYOUT_NHWC"}


@dataclass(frozen=True)
class BatchNormCaseFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    layout: str
    channels: int
    input_shape_nchw: tuple[int, int, int, int]
    seed: int
    input_min: float = -4.0
    input_max: float = 4.0


class BatchNormModel(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=True, track_running_stats=True)
        with torch.no_grad():
            self.bn.weight.copy_(torch.linspace(0.8, 1.2, channels))
            self.bn.bias.copy_(torch.linspace(-0.15, 0.15, channels))
            self.bn.running_mean.copy_(torch.linspace(-0.2, 0.2, channels))
            self.bn.running_var.copy_(torch.linspace(0.5, 1.5, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class BatchNormSettingsFlt:
    def __init__(self, case: BatchNormCaseFlt):
        self.case = case
        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(case.dataset)
        self.input_table_file = self.pregenerated_data_dir / "input_tensor.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_input(self, values: np.ndarray) -> None:
        save_pregenerated_array(self.input_table_file, values)

    def _load_input(self) -> np.ndarray:
        return load_pregenerated_array(self.input_table_file, self.case.input_shape_nchw)

    def get_input_data(self, regenerate_input: bool) -> np.ndarray:
        if self.input_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input_table_file}")
            return self._load_input()

        rng = np.random.default_rng(self.case.seed)
        values = rng.uniform(
            self.case.input_min,
            self.case.input_max,
            size=self.case.input_shape_nchw,
        ).astype(np.float32)
        print(f"Saving data to {self.input_table_file}")
        self._save_input(values)
        return values

    def _reorder(self, values_nchw: np.ndarray) -> np.ndarray:
        return np.transpose(values_nchw, (0, 2, 3, 1))

    def _fold_affine(self, model: BatchNormModel) -> tuple[np.ndarray, np.ndarray]:
        bn = model.bn
        weight = bn.weight.detach().cpu().numpy().astype(np.float32)
        bias = bn.bias.detach().cpu().numpy().astype(np.float32)
        mean = bn.running_mean.detach().cpu().numpy().astype(np.float32)
        var = bn.running_var.detach().cpu().numpy().astype(np.float32)
        scale = weight / np.sqrt(var + float(bn.eps))
        folded_bias = bias - mean * scale
        return scale, folded_bias

    def _write_config_header(self) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")
        n, c, h, w = self.case.input_shape_nchw
        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_INPUT_N {n}\n")
            fh.write(f"#define {prefix}_INPUT_H {h}\n")
            fh.write(f"#define {prefix}_INPUT_W {w}\n")
            fh.write(f"#define {prefix}_INPUT_C {c}\n")
            fh.write(f"#define {prefix}_SIZE {n * c * h * w}\n")
            fh.write(f"#define {prefix}_LAYOUT {LAYOUT_VALUE[self.case.layout]}\n")
        format_generated_file(filepath)

    def _write_array_header(self, suffix: str, values: np.ndarray) -> None:
        filename = f"{suffix}_data.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.case.dataset}_{suffix}", values, self.case.dtype_name)

    def _write_wrapper_header(self) -> None:
        write_wrapper_header(self.headers_dir / "test_data.h", self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        input_nchw = self.get_input_data(regenerate_input)
        model = BatchNormModel(self.case.channels).eval()
        scale, bias = self._fold_affine(model)
        output_nchw = model(torch.tensor(input_nchw, dtype=torch.float32)).detach().cpu().numpy().astype(np.float32)

        input_export = self._reorder(input_nchw)
        output_export = self._reorder(output_nchw)
        if self.case.dtype_name == "f16":
            input_export = input_export.astype(np.float16)
            scale = scale.astype(np.float16)
            bias = bias.astype(np.float16)
            output_export = output_export.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header()
        self._write_array_header("input", input_export)
        self._write_array_header("scale", scale)
        self._write_array_header("bias", bias)
        self._write_array_header("output_ref", output_export)
        self._write_wrapper_header()


def build_cases() -> list[BatchNormCaseFlt]:
    base_cases = [
        ("bn_basic", 6, (1, 6, 8, 8), "nhwc", 8101),
        ("bn_basic_nhwc", 6, (1, 6, 8, 8), "nhwc", 8102),
        ("bn_op", 7, (1, 7, 5, 6), "nhwc", 8103),
        ("bn_op_nhwc", 7, (1, 7, 5, 6), "nhwc", 8104),
    ]
    cases: list[BatchNormCaseFlt] = []
    for dtype_name, torch_dtype, seed_offset in (("f32", torch.float32, 0), ("f16", torch.float16, 100)):
        for name, channels, shape, layout, seed in base_cases:
            cases.append(
                BatchNormCaseFlt(
                    dataset=f"{name}_{dtype_name}",
                    dtype_name=dtype_name,
                    torch_dtype=torch_dtype,
                    layout=layout,
                    channels=channels,
                    input_shape_nchw=shape,
                    seed=seed + seed_offset,
                )
            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float batch-norm unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, batch_norm_f32_family, "
            "batch_norm_f16_family, or a specific dataset name."
        ),
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated input samples.")
    return parser.parse_args()


def matches(dataset_filter: str, case: BatchNormCaseFlt) -> bool:
    if dataset_filter == "all":
        return True
    if dataset_filter == "batch_norm_f32_family":
        return case.dtype_name == "f32"
    if dataset_filter == "batch_norm_f16_family":
        return case.dtype_name == "f16"
    return case.dataset == dataset_filter


def main() -> None:
    args = parse_args()
    selected = [case for case in build_cases() if matches(args.dataset, case)]
    if not selected:
        raise RuntimeError(f"No batch_norm dataset matched '{args.dataset}'")
    for case in selected:
        print(f"Generating {case.dataset}...")
        BatchNormSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
