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
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


UNIT_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = UNIT_TEST_ROOT.parents[1]

INTEGER_TEST_DIR = UNIT_TEST_ROOT / "TestCases"


@dataclass(frozen=True)
class Toolchain:
    requested: str
    family: str
    cc: str
    cxx: str
    c_flags: str
    cxx_flags: str


@dataclass(frozen=True)
class StepResult:
    stage: str
    test: str
    toolchain: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class IntegerTestSpec:
    name: str
    cmake_target: str


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_integer_tests() -> list[IntegerTestSpec]:
    specs: list[IntegerTestSpec] = []
    pattern = re.compile(r"add_cmsis_nn_unit_test_executable\(([^)]+)\)")

    for test_dir in sorted(INTEGER_TEST_DIR.iterdir()):
        if not test_dir.is_dir():
            continue
        if not test_dir.name.startswith("test_arm_"):
            continue
        if "f16" in test_dir.name or "f32" in test_dir.name:
            continue

        cmake_lists = test_dir / "CMakeLists.txt"
        if not cmake_lists.exists():
            continue

        match = pattern.search(cmake_lists.read_text(encoding="utf-8"))
        if not match:
            raise RuntimeError(f"Unable to find test target in {cmake_lists}")

        specs.append(IntegerTestSpec(name=test_dir.name, cmake_target=match.group(1).strip()))

    if not specs:
        raise RuntimeError(f"No integer unit tests found under {INTEGER_TEST_DIR}")

    return specs


INTEGER_TEST_SPECS = discover_integer_tests()
INTEGER_TEST_NAMES = [spec.name for spec in INTEGER_TEST_SPECS]
INTEGER_TEST_MAP = {spec.name: spec for spec in INTEGER_TEST_SPECS}


def parse_version(value: str) -> tuple[int | str, ...]:
    parts: list[int | str] = []
    for part in re.split(r"[.-]", value):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return tuple(parts)


def env_name_for_toolchain(toolchain: str) -> str | None:
    match = re.match(r"^([A-Za-z0-9]+)@([0-9][A-Za-z0-9_.-]*)$", toolchain)
    if not match:
        return None
    family = match.group(1).upper()
    version = re.sub(r"[^A-Za-z0-9]", "_", match.group(2))
    return f"{family}_TOOLCHAIN_{version}"


def resolve_tool(bin_dir: Path | None, name: str) -> str | None:
    if bin_dir is not None:
        candidate = bin_dir / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def toolchain_dirs_from_env(prefix: str) -> list[Path]:
    matches: list[Path] = []
    generic_name = f"{prefix}_TOOLCHAIN"
    if generic_name in os.environ:
        matches.append(Path(os.environ[generic_name]).resolve())

    versioned = sorted(
        (name for name in os.environ if name.startswith(f"{prefix}_TOOLCHAIN_")),
        key=lambda name: parse_version(name.split(f"{prefix}_TOOLCHAIN_", 1)[1]),
        reverse=True,
    )
    for name in versioned:
        matches.append(Path(os.environ[name]).resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in matches:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def resolve_toolchain(requested: str) -> Toolchain:
    family = requested.split("@", 1)[0].upper()
    exact_env = env_name_for_toolchain(requested)
    search_dirs: list[Path | None] = []
    if exact_env and exact_env in os.environ:
        search_dirs.append(Path(os.environ[exact_env]).resolve())
    if family == "GCC":
        search_dirs.extend(toolchain_dirs_from_env("GCC"))
    elif family == "AC6":
        search_dirs.extend(toolchain_dirs_from_env("AC6"))
    search_dirs.append(None)

    if family == "GCC":
        cc = None
        cxx = None
        for search_dir in search_dirs:
            cc = resolve_tool(search_dir, "arm-none-eabi-gcc")
            cxx = resolve_tool(search_dir, "arm-none-eabi-g++")
            if cc and cxx:
                break
        if not cc or not cxx:
            raise RuntimeError(
                "Unable to resolve arm-none-eabi-gcc/g++. Export GCC_TOOLCHAIN "
                "(or a versioned GCC_TOOLCHAIN_<version>) or add the compiler to PATH."
            )
        flags = "-mcpu=cortex-m55 -mthumb -mfloat-abi=hard"
        return Toolchain(requested=requested, family=family, cc=cc, cxx=cxx, c_flags=flags, cxx_flags=flags)

    if family == "AC6":
        cc = None
        for search_dir in search_dirs:
            cc = resolve_tool(search_dir, "armclang")
            if cc:
                break
        if not cc:
            raise RuntimeError(
                "Unable to resolve armclang. Export AC6_TOOLCHAIN "
                "(or a versioned AC6_TOOLCHAIN_<version>) or add armclang to PATH."
            )
        flags = "--target=arm-arm-none-eabi -mcpu=Cortex-M55"
        return Toolchain(requested=requested, family=family, cc=cc, cxx=cc, c_flags=flags, cxx_flags=flags)

    raise RuntimeError(f"Unsupported toolchain family '{family}'. Supported families: GCC, AC6.")


def resolve_pack_dir(pack_root: Path, vendor: str, name: str, preferred_version: str) -> Path:
    base = pack_root / vendor / name
    if not base.exists():
        raise RuntimeError(f"Required pack directory not found: {base}")

    preferred = base / preferred_version
    if preferred.exists():
        return preferred

    versions = sorted((path for path in base.iterdir() if path.is_dir()), key=lambda path: parse_version(path.name))
    if not versions:
        raise RuntimeError(f"No installed versions found for pack {vendor}::{name} under {base}")
    return versions[-1]


def create_cmsis_overlay(overlay_root: Path, pack_root: Path, cmsis_version: str, cortex_dfp_version: str) -> Path:
    cmsis_dir = resolve_pack_dir(pack_root, "ARM", "CMSIS", cmsis_version)
    cortex_dfp_dir = resolve_pack_dir(pack_root, "ARM", "Cortex_DFP", cortex_dfp_version)

    if overlay_root.exists():
        shutil.rmtree(overlay_root)

    (overlay_root / "Device" / "ARM").mkdir(parents=True, exist_ok=True)

    os.symlink(cmsis_dir / "CMSIS", overlay_root / "CMSIS", target_is_directory=True)
    os.symlink(cortex_dfp_dir / "Device" / "ARMCM55", overlay_root / "Device" / "ARM" / "ARMCM55", target_is_directory=True)

    return overlay_root


def run_command(cmd: list[str], log_path: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if exc.stderr:
            output += exc.stderr
        log_path.write_text(output, encoding="utf-8")
        raise

    log_path.write_text(completed.stdout, encoding="utf-8")
    return completed


def summarize_unity_ticks(output: str) -> str:
    tick_matches = [int(match) for match in re.findall(r":PASS\s+\((\d+)\s+ticks\)", output)]
    if not tick_matches:
        return ""
    return f"sum {sum(tick_matches)} ticks ({len(tick_matches)} tests)"


def _decode_timeout_output(stream: bytes | str | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream


def configure_build(
    build_dir: Path,
    overlay_root: Path,
    toolchain: Toolchain,
    optimization_level: str,
    log_dir: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "cmake",
        "-S",
        str(UNIT_TEST_ROOT),
        "-B",
        str(build_dir),
        f"-DCMSIS_PATH={overlay_root}",
        "-DCMAKE_SYSTEM_NAME=Generic",
        "-DCMAKE_SYSTEM_PROCESSOR=cortex-m55",
        "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
        f"-DCMAKE_C_COMPILER={toolchain.cc}",
        f"-DCMAKE_CXX_COMPILER={toolchain.cxx}",
        f"-DCMAKE_C_FLAGS={toolchain.c_flags}",
        f"-DCMAKE_CXX_FLAGS={toolchain.cxx_flags}",
        f"-DCMAKE_ASM_FLAGS={toolchain.c_flags}",
        f"-DCMSIS_OPTIMIZATION_LEVEL={optimization_level}",
        "-Wno-dev",
    ]
    return run_command(cmd, log_dir / "configure.log")


def build_targets(
    build_dir: Path,
    toolchain: Toolchain,
    targets: list[str],
    jobs: int,
    log_dir: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = ["cmake", "--build", str(build_dir), f"-j{jobs}", "--target", *targets]
    return run_command(cmd, log_dir / "build.log")


def target_image(build_dir: Path, test_name: str, cmake_target: str) -> Path:
    direct = build_dir / "TestCases" / test_name / f"{cmake_target}.elf"
    if direct.exists():
        return direct

    test_dir = build_dir / "TestCases" / test_name
    if test_dir.exists():
        matches = sorted(test_dir.glob("*.elf"))
        if len(matches) == 1:
            return matches[0]

    raise FileNotFoundError(f"Expected test image not found for {test_name} (target {cmake_target})")


def run_fvp_target(
    build_dir: Path,
    test_name: str,
    cmake_target: str,
    fvp_bin: str,
    fvp_image_arg: str,
    timeout: int,
    log_dir: Path,
) -> subprocess.CompletedProcess[str]:
    image = target_image(build_dir, test_name, cmake_target)

    cmd = [fvp_bin]
    if fvp_image_arg:
        cmd.append(fvp_image_arg)
    cmd.append(str(image))
    cmd.extend(
        [
            "-C",
            "mps3_board.visualisation.disable-visualisation=1",
            "-C",
            "mps3_board.telnetterminal0.start_telnet=0",
            "-C",
            "mps3_board.uart0.out_file=-",
            "-C",
            "mps3_board.uart0.unbuffered_output=1",
            "-C",
            "mps3_board.uart0.shutdown_on_eot=1",
        ]
    )
    return run_command(cmd, log_dir / f"{test_name}.log", timeout=timeout)


def print_summary(results: list[StepResult]) -> None:
    headers = ["Stage", "Test", "Toolchain", "Status", "Detail"]
    rows = [[result.stage, result.test, result.toolchain, result.status, result.detail] for result in results]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def fmt(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)

    print("\nInteger Unit-Test Summary")
    print(fmt(headers))
    print(separator)
    for row in rows:
        print(fmt(row))

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(separator)
    print(f"PASS={counts['PASS']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run legacy CMSIS-NN integer unit tests on Corstone-300 FVP.")
    parser.add_argument("--tests", default="all", help="Comma-separated integer test targets to run, or all.")
    parser.add_argument("--toolchains", default="GCC,AC6", help="Comma-separated toolchains to use. Supported: GCC, AC6.")
    parser.add_argument("--list", action="store_true", help="List supported integer test targets and exit.")
    parser.add_argument("--build-fvp", action="store_true", help="Configure and build integer FVP tests.")
    parser.add_argument("--run-fvp", action="store_true", help="Run built integer FVP tests.")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel jobs for CMake builds.")
    parser.add_argument("--clean", action="store_true", help="Delete existing build directories before configuring.")
    parser.add_argument("--build-root", default="/tmp/cmsis-nn-integer-fvp", help="Root build directory.")
    parser.add_argument("--cmsis-pack-root", default=os.environ.get("CMSIS_PACK_ROOT", "/home/runner/.cache/arm/packs"), help="CMSIS pack root.")
    parser.add_argument("--cmsis-version", default="6.3.0", help="Preferred ARM::CMSIS pack version.")
    parser.add_argument("--cortex-dfp-version", default="1.1.0", help="Preferred ARM::Cortex_DFP pack version.")
    parser.add_argument("--optimization-level", default="-O3", help="CMSIS optimization level for the integer FVP build.")
    parser.add_argument("--fvp-bin", default="FVP_Corstone_SSE-300", help="FVP executable.")
    parser.add_argument("--fvp-image-arg", default="", help="Optional image argument flag for the FVP binary, for example -a.")
    parser.add_argument("--fvp-timeout", type=int, default=90, help="FVP timeout in seconds per test image.")
    return parser.parse_args()


def resolve_tests(selected: str) -> list[str]:
    requested = parse_csv(selected)
    if not requested or requested == ["all"]:
        return list(INTEGER_TEST_NAMES)

    supported = set(INTEGER_TEST_NAMES)
    unknown = [item for item in requested if item not in supported]
    if unknown:
        raise RuntimeError(f"Unsupported integer test target(s): {', '.join(unknown)}")

    return requested


def main() -> int:
    args = parse_args()

    if args.list:
        for target in INTEGER_TEST_NAMES:
            print(target)
        return 0

    selected_tests = resolve_tests(args.tests)
    requested_toolchains = parse_csv(args.toolchains)
    if not requested_toolchains:
        raise RuntimeError("No toolchains selected.")

    if not args.build_fvp and not args.run_fvp:
        args.build_fvp = True
        args.run_fvp = True

    build_root = Path(args.build_root).resolve()
    pack_root = Path(args.cmsis_pack_root).resolve()
    overlay_root = build_root / "cmsis-overlay"
    overlay_root = create_cmsis_overlay(overlay_root, pack_root, args.cmsis_version, args.cortex_dfp_version)

    results: list[StepResult] = []
    failed = False

    for requested_toolchain in requested_toolchains:
        toolchain = resolve_toolchain(requested_toolchain)
        build_dir = build_root / toolchain.family.lower()
        log_dir = build_dir / "logs"

        if args.clean and build_dir.exists():
            shutil.rmtree(build_dir)

        if args.build_fvp:
            configure = configure_build(
                build_dir,
                overlay_root,
                toolchain,
                args.optimization_level,
                log_dir,
            )
            if configure.returncode != 0:
                for target in selected_tests:
                    results.append(StepResult("build-fvp", target, toolchain.requested, "FAIL", "cmake configure failed"))
                failed = True
                continue

            cmake_targets = [INTEGER_TEST_MAP[test_name].cmake_target for test_name in selected_tests]
            build = build_targets(build_dir, toolchain, cmake_targets, args.jobs, log_dir)
            build_status = "PASS" if build.returncode == 0 else "FAIL"
            build_detail = "" if build.returncode == 0 else "cmake build failed"
            for test_name in selected_tests:
                results.append(StepResult("build-fvp", test_name, toolchain.requested, build_status, build_detail))
            if build.returncode != 0:
                failed = True
                continue

        if args.run_fvp:
            for test_name in selected_tests:
                spec = INTEGER_TEST_MAP[test_name]
                try:
                    completed = run_fvp_target(
                        build_dir,
                        test_name,
                        spec.cmake_target,
                        args.fvp_bin,
                        args.fvp_image_arg,
                        args.fvp_timeout,
                        log_dir / "fvp",
                    )
                    output = completed.stdout
                    passed = completed.returncode == 0 and "0 Failures" in output and "OK" in output
                    if passed:
                        results.append(
                            StepResult(
                                "run-fvp",
                                test_name,
                                toolchain.requested,
                                "PASS",
                                summarize_unity_ticks(output),
                            )
                        )
                    else:
                        detail = "fvp run failed"
                        fail_line = next((line.strip() for line in output.splitlines() if ":FAIL:" in line), "")
                        if fail_line:
                            detail = fail_line
                        results.append(StepResult("run-fvp", test_name, toolchain.requested, "FAIL", detail))
                        failed = True
                except subprocess.TimeoutExpired:
                    timeout_log = (log_dir / "fvp" / f"{test_name}.log")
                    output = timeout_log.read_text(encoding="utf-8") if timeout_log.exists() else ""
                    timed_out_after_success = (
                        "0 Failures" in output
                        and "\nOK" in output
                        and "FAIL" not in output
                    )
                    if timed_out_after_success:
                        tick_summary = summarize_unity_ticks(output)
                        timeout_detail = f"timeout {args.fvp_timeout}s after PASS output"
                        if tick_summary:
                            timeout_detail = f"{tick_summary}; {timeout_detail}"
                        results.append(
                            StepResult(
                                "run-fvp",
                                test_name,
                                toolchain.requested,
                                "PASS",
                                timeout_detail,
                            )
                        )
                    else:
                        results.append(StepResult("run-fvp", test_name, toolchain.requested, "FAIL", "fvp timeout"))
                        failed = True
                except FileNotFoundError as exc:
                    results.append(StepResult("run-fvp", test_name, toolchain.requested, "FAIL", str(exc)))
                    failed = True

    print_summary(results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
