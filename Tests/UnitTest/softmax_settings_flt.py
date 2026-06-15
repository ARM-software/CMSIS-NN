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
from float_test_gen_utils import (
    format_generated_file,
    load_pregenerated_array,
    prepare_dataset_dirs,
    save_pregenerated_array,
    write_common_header,
    write_float_array_header,
    write_wrapper_header,
)


@dataclass
class SoftmaxSettingsFlt:
    dataset: str
    dtype_name: str
    torch_dtype: torch.dtype
    num_rows: int = 2
    row_size: int = 5

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
        return load_pregenerated_array(self.input_table_file, (self.num_rows, self.row_size))

    def get_input_data(self, regenerate_input: bool) -> np.ndarray:
        if self.input_table_file.exists() and not regenerate_input:
            print(f"Loading data from {self.input_table_file}")
            return self._load_input()

        input_data = np.array(
            [
                [1.0, 0.25, -0.5, 2.0, -1.25],
                [-2.0, -0.125, 0.75, 3.0, 1.5],
            ],
            dtype=np.float32,
        )
        print(f"Saving data to {self.input_table_file}")
        self._save_input(input_data)
        return input_data

    def _write_config_header(self) -> None:
        prefix = self.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")

        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_NUM_ROWS {self.num_rows}\n")
            fh.write(f"#define {prefix}_ROW_SIZE {self.row_size}\n")
            fh.write(f"#define {prefix}_DST_SIZE {self.num_rows * self.row_size}\n")
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
        output_tensor = torch.softmax(input_tensor, dim=1).to(dtype=self.torch_dtype)

        output_data = output_tensor.cpu().numpy().astype(np.float32)
        if self.dtype_name == "f16":
            output_data = output_data.astype(np.float16)

        self.generated_header_files = []
        self._write_config_header()
        self._write_array_header("input", input_data.astype(np.float32 if self.dtype_name == "f32" else np.float16))
        self._write_array_header("output_ref", output_data)
        self._write_wrapper_header()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float softmax unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        choices=["softmax_f32", "softmax_f16", "all"],
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

    jobs: list[SoftmaxSettingsFlt] = []
    if args.dataset in ("softmax_f32", "all"):
        jobs.append(SoftmaxSettingsFlt(dataset="softmax_f32", dtype_name="f32", torch_dtype=torch.float32))
    if args.dataset in ("softmax_f16", "all"):
        jobs.append(SoftmaxSettingsFlt(dataset="softmax_f16", dtype_name="f16", torch_dtype=torch.float16))

    for job in jobs:
        print(f"Generating {job.dataset}...")
        job.generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
