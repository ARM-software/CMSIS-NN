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


GATE_ORDER = {
    "input": 0,
    "forget": 1,
    "cell": 2,
    "output": 3,
}


@dataclass(frozen=True)
class LstmCaseFlt:
    dataset: str
    dtype_name: str
    time_major: int
    time_steps: int
    batch_size: int
    input_size: int
    hidden_size: int
    seed: int
    input_min: float = -1.0
    input_max: float = 1.0
    weight_min: float = -0.35
    weight_max: float = 0.35
    bias_min: float = -0.25
    bias_max: float = 0.25


class LstmSettingsFlt:
    def __init__(self, case: LstmCaseFlt):
        self.case = case
        self.headers_dir, self.pregenerated_data_dir = prepare_dataset_dirs(case.dataset)
        self.input_table_file = self.pregenerated_data_dir / "input.txt"
        self.weight_ih_table_file = self.pregenerated_data_dir / "weight_ih.txt"
        self.weight_hh_table_file = self.pregenerated_data_dir / "weight_hh.txt"
        self.bias_ih_table_file = self.pregenerated_data_dir / "bias_ih.txt"
        self.bias_hh_table_file = self.pregenerated_data_dir / "bias_hh.txt"
        self.generated_header_files: list[str] = []
        self.script_name = Path(__file__).name

    def _input_shape(self) -> tuple[int, int, int]:
        if self.case.time_major:
            return (self.case.time_steps, self.case.batch_size, self.case.input_size)
        return (self.case.batch_size, self.case.time_steps, self.case.input_size)

    def _save_array(self, filepath: Path, values: np.ndarray) -> None:
        save_pregenerated_array(filepath, values)

    def _load_array(self, filepath: Path, shape: tuple[int, ...]) -> np.ndarray:
        return load_pregenerated_array(filepath, shape)

    def get_raw_tensors(self, regenerate_input: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        input_shape = self._input_shape()
        weight_ih_shape = (4 * self.case.hidden_size, self.case.input_size)
        weight_hh_shape = (4 * self.case.hidden_size, self.case.hidden_size)
        bias_shape = (4 * self.case.hidden_size,)
        if (
            self.input_table_file.exists()
            and self.weight_ih_table_file.exists()
            and self.weight_hh_table_file.exists()
            and self.bias_ih_table_file.exists()
            and self.bias_hh_table_file.exists()
            and not regenerate_input
        ):
            print(f"Loading data from {self.pregenerated_data_dir}")
            return (
                self._load_array(self.input_table_file, input_shape),
                self._load_array(self.weight_ih_table_file, weight_ih_shape),
                self._load_array(self.weight_hh_table_file, weight_hh_shape),
                self._load_array(self.bias_ih_table_file, bias_shape),
                self._load_array(self.bias_hh_table_file, bias_shape),
            )

        rng = np.random.default_rng(self.case.seed)
        input_data = rng.uniform(self.case.input_min, self.case.input_max, size=input_shape).astype(np.float32)
        weight_ih = rng.uniform(self.case.weight_min, self.case.weight_max, size=weight_ih_shape).astype(np.float32)
        weight_hh = rng.uniform(self.case.weight_min, self.case.weight_max, size=weight_hh_shape).astype(np.float32)
        bias_ih = rng.uniform(self.case.bias_min, self.case.bias_max, size=bias_shape).astype(np.float32)
        bias_hh = rng.uniform(self.case.bias_min, self.case.bias_max, size=bias_shape).astype(np.float32)
        print(f"Saving data to {self.pregenerated_data_dir}")
        self._save_array(self.input_table_file, input_data)
        self._save_array(self.weight_ih_table_file, weight_ih)
        self._save_array(self.weight_hh_table_file, weight_hh)
        self._save_array(self.bias_ih_table_file, bias_ih)
        self._save_array(self.bias_hh_table_file, bias_hh)
        return input_data, weight_ih, weight_hh, bias_ih, bias_hh

    def _run_reference(
        self,
        input_data: np.ndarray,
        weight_ih: np.ndarray,
        weight_hh: np.ndarray,
        bias_ih: np.ndarray,
        bias_hh: np.ndarray,
    ) -> np.ndarray:
        batch_first = not bool(self.case.time_major)
        lstm = nn.LSTM(
            input_size=self.case.input_size,
            hidden_size=self.case.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=batch_first,
            bidirectional=False,
        ).eval()
        with torch.no_grad():
            lstm.weight_ih_l0.copy_(torch.tensor(weight_ih))
            lstm.weight_hh_l0.copy_(torch.tensor(weight_hh))
            lstm.bias_ih_l0.copy_(torch.tensor(bias_ih))
            lstm.bias_hh_l0.copy_(torch.tensor(bias_hh))
            output, _ = lstm(torch.tensor(input_data, dtype=torch.float32))
        return output.detach().cpu().numpy().astype(np.float32)

    def _gate_slice(self, values: np.ndarray, gate_name: str) -> np.ndarray:
        gate_idx = GATE_ORDER[gate_name]
        hidden_size = self.case.hidden_size
        return values[gate_idx * hidden_size:(gate_idx + 1) * hidden_size].astype(np.float32, copy=False)

    def _write_config_header(self) -> None:
        prefix = self.case.dataset.upper()
        filepath = self.headers_dir / "config_data.h"
        self.generated_header_files.append("config_data.h")
        with filepath.open("w", encoding="utf-8") as fh:
            write_common_header(fh, self.script_name)
            fh.write(f"#define {prefix}_TIME_MAJOR {self.case.time_major}\n")
            fh.write(f"#define {prefix}_TIME_STEPS {self.case.time_steps}\n")
            fh.write(f"#define {prefix}_BATCH_SIZE {self.case.batch_size}\n")
            fh.write(f"#define {prefix}_INPUT_SIZE {self.case.input_size}\n")
            fh.write(f"#define {prefix}_HIDDEN_SIZE {self.case.hidden_size}\n")
            fh.write(f"#define {prefix}_CELL_CLIP 0.0f\n")
            fh.write(f"#define {prefix}_CELL_STATE_SIZE {self.case.batch_size * self.case.hidden_size}\n")
            fh.write(f"#define {prefix}_DST_SIZE {self.case.time_steps * self.case.batch_size * self.case.hidden_size}\n")
        format_generated_file(filepath)

    def _write_array_header(self, suffix: str, values: np.ndarray) -> None:
        filename = f"{suffix}_data.h"
        filepath = self.headers_dir / filename
        self.generated_header_files.append(filename)
        write_float_array_header(filepath, self.script_name, f"{self.case.dataset}_{suffix}", values, self.case.dtype_name)

    def _write_wrapper_header(self) -> None:
        write_wrapper_header(self.headers_dir / "test_data.h", self.script_name, self.generated_header_files)

    def generate_data(self, regenerate_input: bool = False) -> None:
        input_data, weight_ih, weight_hh, bias_ih, bias_hh = self.get_raw_tensors(regenerate_input)
        output_ref = self._run_reference(input_data, weight_ih, weight_hh, bias_ih, bias_hh)
        dtype = np.float16 if self.case.dtype_name == "f16" else np.float32

        self.generated_header_files = []
        self._write_config_header()
        self._write_array_header("input", input_data.astype(dtype))
        self._write_array_header("output_ref", output_ref.astype(dtype))
        for gate_name in ("forget", "input", "cell", "output"):
            self._write_array_header(f"{gate_name}_input_weights", self._gate_slice(weight_ih, gate_name).astype(dtype))
            self._write_array_header(f"{gate_name}_hidden_weights", self._gate_slice(weight_hh, gate_name).astype(dtype))
            combined_bias = self._gate_slice(bias_ih, gate_name) + self._gate_slice(bias_hh, gate_name)
            self._write_array_header(f"{gate_name}_bias", combined_bias.astype(dtype))
        self._write_wrapper_header()


def build_cases() -> list[LstmCaseFlt]:
    base_cases = [
        ("lstm_small", 1, 5, 2, 8, 8, 8401),
        ("lstm_medium", 1, 8, 3, 12, 16, 8402),
        ("lstm_large", 1, 12, 4, 16, 24, 8403),
        # Match legacy s8 lstm_1 geometry.
        ("lstm_match_1", 1, 10, 1, 22, 11, 8404),
        # Match legacy s8 lstm_2 geometry.
        ("lstm_match_2", 0, 9, 1, 6, 7, 8405),
        # Match legacy s8 lstm_one_time_step geometry.
        ("lstm_match_one_time_step", 0, 1, 3, 22, 3, 8406),
    ]
    cases: list[LstmCaseFlt] = []
    for dtype_name, seed_offset in (("f32", 0), ("f16", 100)):
        for name, time_major, time_steps, batch_size, input_size, hidden_size, seed in base_cases:
            cases.append(
                LstmCaseFlt(
                    dataset=f"{name}_{dtype_name}",
                    dtype_name=dtype_name,
                    time_major=time_major,
                    time_steps=time_steps,
                    batch_size=batch_size,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    seed=seed + seed_offset,
                )
            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate float LSTM unit-test references using PyTorch.")
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset or family to generate. Supported values: all, lstm_f32_family, "
            "lstm_f16_family, or a specific dataset name."
        ),
    )
    parser.add_argument("--regenerate-input", action="store_true", help="Regenerate pregenerated input samples.")
    return parser.parse_args()


def matches(dataset_filter: str, case: LstmCaseFlt) -> bool:
    if dataset_filter == "all":
        return True
    if dataset_filter == "lstm_f32_family":
        return case.dtype_name == "f32"
    if dataset_filter == "lstm_f16_family":
        return case.dtype_name == "f16"
    return case.dataset == dataset_filter


def main() -> None:
    args = parse_args()
    selected = [case for case in build_cases() if matches(args.dataset, case)]
    if not selected:
        raise RuntimeError(f"No lstm dataset matched '{args.dataset}'")
    for case in selected:
        print(f"Generating {case.dataset}...")
        LstmSettingsFlt(case).generate_data(regenerate_input=args.regenerate_input)


if __name__ == "__main__":
    main()
