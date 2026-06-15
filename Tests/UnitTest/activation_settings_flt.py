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


@dataclass
class ActivationSettingsFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    size: int = 9
    leaky_relu_param: float = 0.125

    outdir: Path = Path("TestCases/TestData")
    pregenerated_dir: Path = Path("PregeneratedData")

    def __post_init__(self) -> None:
        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(
            self.dataset, self.outdir, self.pregenerated_dir
        )

        self.input_table_file = self.pregenerated_data_dir / "input.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _save_input(self, array: np.ndarray) -> None:
        save_pregenerated_array(self.input_table_file, array)

    def _load_input(self) -> np.ndarray:
        return load_pregenerated_array(self.input_table_file, (self.size,))

    def get_input_data(self, regenerate_input: bool) -> np.ndarray:
        if self.input_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input_table_file}")
            return self._load_input()

        input_data = np.array([-5.0, -3.0, -1.25, -0.1, 0.0, 0.5, 1.5, 3.0, 6.0], dtype=np.float32)
        print(f"Saving data to {self.input_table_file}")
        self._save_input(input_data)
        return input_data

    def _write_config_header(self) -> None:
        prefix = self.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_SIZE {self.size}\n")
            if self.dtype_name == "f32":
                fh.write(f"#define {prefix}_LEAKY_RELU_PARAM {format_float32(self.leaky_relu_param)}\n")
            else:
                fh.write(f"#define {prefix}_LEAKY_RELU_PARAM {format_float16(self.leaky_relu_param)}\n")
        format_generated_file(filepath)

    def _write_array_header(self, name: str, values: np.ndarray) -> None:
        filename = f"{name}_data.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.dataset}_{name}", values, self.dtype_name)

    def _write_wrapper_header(self) -> None:
        filepath = self.headers_dir / "test_data.h"
        write_wrapper_header(filepath, self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        input_data = self.get_input_data(regenerate_input)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        outputs = {
            "input": input_data.astype(np.float32 if self.dtype_name == "f32" else np.float16),
            "output_ref_sigmoid": torch.sigmoid(input_tensor).to(dtype=self.torch_dtype).cpu().numpy(),
            "output_ref_tanh": torch.tanh(input_tensor).to(dtype=self.torch_dtype).cpu().numpy(),
            "output_ref_hardswish": F.hardswish(input_tensor).to(dtype=self.torch_dtype).cpu().numpy(),
            "output_ref_leaky_relu": F.leaky_relu(input_tensor, negative_slope=self.leaky_relu_param)
            .to(dtype=self.torch_dtype)
            .cpu()
            .numpy(),
        }

        if self.dtype_name == "f16":
            outputs = {name: value.astype(np.float16) for name, value in outputs.items()}

        self.generated_header_files = []
        self._write_config_header()
        for name, values in outputs.items():
            self._write_array_header(name, values)
        self._write_wrapper_header()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float activation unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        choices=["activation_f32", "activation_f16", "all"],
        default="all",
        help="Dataset to generate.",
    )
    parser.add_argument(
        "--regenerate-input",
        action="store_true",
        help="Regenerate the pregenerated input sample instead of reusing input.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    jobs: list[ActivationSettingsFlt] = []
    if args.dataset in ("activation_f32", "all"):
        jobs.append(ActivationSettingsFlt(dataset="activation_f32", dtype_name="f32", torch_dtype=torch.float32))
    if args.dataset in ("activation_f16", "all"):
        jobs.append(ActivationSettingsFlt(dataset="activation_f16", dtype_name="f16", torch_dtype=torch.float16))

    for job in jobs:
        print(f"Generating {job.dataset}...")
        job.generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
