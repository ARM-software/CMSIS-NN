#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import dataclasses
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_ROOT = Path("/tmp/cmsis-nn-lib-variants")


@dataclasses.dataclass(frozen=True)
class Variant:
    key: str
    label: str
    enable_f16: bool
    enable_f32: bool


@dataclasses.dataclass(frozen=True)
class Toolchain:
    requested: str
    family: str
    cc: str
    cxx: str
    c_flags: str
    cxx_flags: str
    ar: str | None = None
    ranlib: str | None = None
    size_mode: str = "section"
    size_tool: str | None = None


@dataclasses.dataclass
class VariantResult:
    variant: Variant
    status: str
    archive_bytes: int = 0
    code_bytes: int = 0
    rodata_bytes: int = 0
    data_bytes: int = 0
    bss_bytes: int = 0
    other_bytes: int = 0
    detail: str = ""


VARIANTS: list[Variant] = [
    Variant("baseline", "baseline (integer)", False, False),
    Variant("f16", "baseline + float16", True, False),
    Variant("f32", "baseline + float32", False, True),
    Variant("f16_f32", "baseline + float16 + float32", True, True),
]


SECTION_LINE_RE = re.compile(r"^(\S+)\s+(\d+)\s+\d+$")
FROMELF_LINE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+.+$")


def run_command(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return completed


def format_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return str(value)


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


def resolve_size_tool(family: str, gcc_bin_dir: Path | None, ac6_bin_dir: Path | None) -> tuple[str, str]:
    if family == "GCC":
        size_tool = resolve_tool(gcc_bin_dir, "arm-none-eabi-size") or shutil.which("llvm-size")
        if size_tool:
            return ("section", size_tool)
        raise RuntimeError("Unable to find arm-none-eabi-size or llvm-size for GCC builds.")

    if family == "AC6":
        size_tool = shutil.which("llvm-size")
        if size_tool:
            return ("section", size_tool)
        size_tool = resolve_tool(ac6_bin_dir, "fromelf")
        if size_tool:
            return ("fromelf", size_tool)
        size_tool = resolve_tool(gcc_bin_dir, "arm-none-eabi-size")
        if size_tool:
            return ("section", size_tool)
        raise RuntimeError("Unable to find llvm-size, fromelf, or arm-none-eabi-size for AC6 builds.")

    if family == "CLANG":
        size_tool = shutil.which("llvm-size") or resolve_tool(gcc_bin_dir, "arm-none-eabi-size")
        if size_tool:
            return ("section", size_tool)
        raise RuntimeError("Unable to find llvm-size or arm-none-eabi-size for CLANG builds.")

    raise RuntimeError(f"Unsupported toolchain family: {family}")


def clang_runtime_root(bin_dir: Path | None) -> Path | None:
    if bin_dir is None:
        return None

    candidate = bin_dir.parent / "lib" / "clang-runtimes" / "arm-none-eabi"
    if candidate.exists():
        return candidate

    lib_dir = bin_dir.parent / "lib"
    if lib_dir.exists():
        matches = sorted(lib_dir.glob("**/clang-runtimes/arm-none-eabi"))
        if matches:
            return matches[0]

    return None


def parse_version(value: str) -> tuple[int | str, ...]:
    parts: list[int | str] = []
    for part in re.split(r"[.-]", value):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return tuple(parts)


def resolve_cmsis_path(cmsis_path: str | None, cmsis_pack_root: str | None, cmsis_version: str) -> Path:
    if cmsis_path:
        path = Path(cmsis_path).resolve()
        if (path / "CMSIS" / "Core" / "Include").exists():
            return path
        raise RuntimeError(f"CMSIS_PATH does not contain CMSIS/Core/Include: {path}")

    if cmsis_pack_root:
        pack_root = Path(cmsis_pack_root).resolve()
        pack_base = pack_root / "ARM" / "CMSIS"
        if not pack_base.exists():
            raise RuntimeError(f"CMSIS pack root does not contain ARM/CMSIS: {pack_base}")

        preferred = pack_base / cmsis_version
        if preferred.exists():
            return preferred

        versions = sorted((path for path in pack_base.iterdir() if path.is_dir()), key=lambda path: parse_version(path.name))
        if versions:
            return versions[-1]
        raise RuntimeError(f"No ARM::CMSIS pack versions found under {pack_base}")

    raise RuntimeError("Provide either --cmsis-path or --cmsis-pack-root/CMSIS_PACK_ROOT for library builds.")


def resolve_toolchain(requested: str, cpu: str, toolchain_bin: str | None) -> Toolchain:
    family = requested.split("@", 1)[0].upper()
    exact_env = env_name_for_toolchain(requested)

    explicit_bin_dir = Path(toolchain_bin).resolve() if toolchain_bin else None
    gcc_env_dir = Path(os.environ[exact_env]).resolve() if exact_env and exact_env in os.environ else None
    generic_env_dir = None
    if family == "GCC" and "GCC_TOOLCHAIN" in os.environ:
        generic_env_dir = Path(os.environ["GCC_TOOLCHAIN"]).resolve()
    elif family == "AC6" and "AC6_TOOLCHAIN" in os.environ:
        generic_env_dir = Path(os.environ["AC6_TOOLCHAIN"]).resolve()
    elif family == "CLANG" and "CLANG_TOOLCHAIN" in os.environ:
        generic_env_dir = Path(os.environ["CLANG_TOOLCHAIN"]).resolve()

    bin_dir = explicit_bin_dir or gcc_env_dir or generic_env_dir

    gcc_bin_dir = None
    if "GCC_TOOLCHAIN_15_2_1" in os.environ:
        gcc_bin_dir = Path(os.environ["GCC_TOOLCHAIN_15_2_1"]).resolve()
    elif "GCC_TOOLCHAIN" in os.environ:
        gcc_bin_dir = Path(os.environ["GCC_TOOLCHAIN"]).resolve()

    ac6_bin_dir = None
    if "AC6_TOOLCHAIN_6_24_0" in os.environ:
        ac6_bin_dir = Path(os.environ["AC6_TOOLCHAIN_6_24_0"]).resolve()
    elif "AC6_TOOLCHAIN" in os.environ:
        ac6_bin_dir = Path(os.environ["AC6_TOOLCHAIN"]).resolve()

    size_mode, size_tool = resolve_size_tool(family, gcc_bin_dir, ac6_bin_dir)

    if family == "GCC":
        cc = resolve_tool(bin_dir, "arm-none-eabi-gcc")
        cxx = resolve_tool(bin_dir, "arm-none-eabi-g++")
        ar = resolve_tool(bin_dir, "arm-none-eabi-ar")
        ranlib = resolve_tool(bin_dir, "arm-none-eabi-ranlib")
        if not cc or not cxx:
            raise RuntimeError("Unable to resolve arm-none-eabi-gcc/g++ for GCC toolchain.")
        flags = (
            f"-mcpu={cpu} -mthumb -mfloat-abi=hard "
            "-fdisable-rtl-ce1 -fdisable-rtl-ce2"
        )
        return Toolchain(requested, family, cc, cxx, flags, flags, ar=ar, ranlib=ranlib, size_mode=size_mode, size_tool=size_tool)

    if family == "AC6":
        cc = resolve_tool(bin_dir, "armclang")
        ar = resolve_tool(bin_dir, "armar")
        if not cc:
            raise RuntimeError("Unable to resolve armclang for AC6 toolchain.")
        flags = f"--target=arm-arm-none-eabi -mcpu={cpu} -mfloat-abi=hard"
        return Toolchain(requested, family, cc, cc, flags, flags, ar=ar, size_mode=size_mode, size_tool=size_tool)

    if family == "CLANG":
        cc = resolve_tool(bin_dir, "clang")
        cxx = resolve_tool(bin_dir, "clang++")
        ar = resolve_tool(bin_dir, "llvm-ar")
        ranlib = resolve_tool(bin_dir, "llvm-ranlib")
        if not cc or not cxx:
            raise RuntimeError("Unable to resolve clang/clang++ for CLANG toolchain.")
        flags = f"--target=arm-arm-none-eabi -mcpu={cpu} -mthumb -mfloat-abi=hard"
        runtime_root = clang_runtime_root(bin_dir)
        if runtime_root:
            flags = f"{flags} --sysroot {runtime_root}"
        return Toolchain(requested, family, cc, cxx, flags, flags, ar=ar, ranlib=ranlib, size_mode=size_mode, size_tool=size_tool)

    raise RuntimeError(f"Unsupported toolchain family '{family}'. Supported families: GCC, AC6, CLANG.")


def should_count_as_code(section: str) -> bool:
    return section.startswith((".text", ".init", ".fini", ".ARM.exidx", ".ARM.extab"))


def should_count_as_rodata(section: str) -> bool:
    return section.startswith((".rodata", ".rdata", ".constdata", ".srodata"))


def should_count_as_data(section: str) -> bool:
    return section.startswith((".data", ".sdata", ".tdata"))


def should_count_as_bss(section: str) -> bool:
    return section.startswith((".bss", ".sbss", ".tbss"))


def should_ignore_section(section: str) -> bool:
    return section.startswith(
        (
            ".debug",
            ".comment",
            ".note",
            ".ARM.attributes",
            ".eh_frame",
            ".symtab",
            ".strtab",
            ".shstrtab",
            ".rela",
            ".rel",
            ".llvm",
        )
    )


def parse_section_size_output(output: str) -> tuple[int, int, int, int, int]:
    code_bytes = 0
    rodata_bytes = 0
    data_bytes = 0
    bss_bytes = 0
    other_bytes = 0

    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = SECTION_LINE_RE.match(line)
        if not match:
            continue
        section = match.group(1)
        size = int(match.group(2))

        if should_ignore_section(section):
            continue
        if should_count_as_code(section):
            code_bytes += size
        elif should_count_as_rodata(section):
            rodata_bytes += size
        elif should_count_as_data(section):
            data_bytes += size
        elif should_count_as_bss(section):
            bss_bytes += size
        else:
            other_bytes += size

    return code_bytes, rodata_bytes, data_bytes, bss_bytes, other_bytes


def parse_fromelf_size_output(output: str) -> tuple[int, int, int, int, int]:
    code_bytes = 0
    rodata_bytes = 0
    data_bytes = 0
    bss_bytes = 0
    other_bytes = 0

    for raw_line in output.splitlines():
        match = FROMELF_LINE_RE.match(raw_line)
        if not match:
            continue
        code_bytes += int(match.group(1))
        rodata_bytes += int(match.group(2))
        data_bytes += int(match.group(3))
        bss_bytes += int(match.group(4))

    return code_bytes, rodata_bytes, data_bytes, bss_bytes, other_bytes


def summarize_archive(archive_path: Path, toolchain: Toolchain) -> tuple[int, int, int, int, int]:
    if not toolchain.size_tool:
        raise RuntimeError("No size tool is configured.")

    if toolchain.size_mode == "fromelf":
        cmd = [toolchain.size_tool, "-z", str(archive_path)]
    else:
        cmd = [toolchain.size_tool, "-A", "-t", str(archive_path)]

    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Size tool failed for {archive_path}:\n{completed.stdout}")

    if toolchain.size_mode == "fromelf":
        return parse_fromelf_size_output(completed.stdout)

    return parse_section_size_output(completed.stdout)


def variant_total_bytes(result: VariantResult) -> int:
    return result.code_bytes + result.rodata_bytes + result.data_bytes + result.bss_bytes


def cmake_generator_args() -> list[str]:
    if shutil.which("ninja"):
        return ["-G", "Ninja"]
    return []


def build_variant(
    repo_root: Path,
    build_root: Path,
    cmsis_path: Path,
    toolchain: Toolchain,
    variant: Variant,
    cpu: str,
    optimization: str,
    jobs: int,
) -> VariantResult:
    build_dir = build_root / toolchain.family.lower() / variant.key
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    configure_log = build_dir / "configure.log"
    build_log = build_dir / "build.log"

    cmake_cmd = [
        "cmake",
        "-S",
        str(repo_root),
        "-B",
        str(build_dir),
        *cmake_generator_args(),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMSIS_PATH={cmsis_path}",
        "-DCMAKE_SYSTEM_NAME=Generic",
        f"-DCMAKE_SYSTEM_PROCESSOR={cpu}",
        "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
        f"-DCMAKE_C_COMPILER={toolchain.cc}",
        f"-DCMAKE_CXX_COMPILER={toolchain.cxx}",
        f"-DCMAKE_C_FLAGS={toolchain.c_flags}",
        f"-DCMAKE_CXX_FLAGS={toolchain.cxx_flags}",
        f"-DCMSIS_OPTIMIZATION_LEVEL={optimization}",
        f"-DARM_NN_ENABLE_F32={'ON' if variant.enable_f32 else 'OFF'}",
        f"-DARM_NN_ENABLE_F16={'ON' if variant.enable_f16 else 'OFF'}",
    ]
    if toolchain.ar:
        cmake_cmd.append(f"-DCMAKE_AR={toolchain.ar}")
    if toolchain.ranlib:
        cmake_cmd.append(f"-DCMAKE_RANLIB={toolchain.ranlib}")

    configured = run_command(cmake_cmd, configure_log)
    if configured.returncode != 0:
        tail = "\n".join(configured.stdout.splitlines()[-40:])
        return VariantResult(variant=variant, status="FAIL", detail=f"configure failed\n{tail}")

    build_cmd = ["cmake", "--build", str(build_dir), "--target", "cmsis-nn", "-j", str(jobs)]
    built = run_command(build_cmd, build_log)
    if built.returncode != 0:
        tail = "\n".join(built.stdout.splitlines()[-40:])
        return VariantResult(variant=variant, status="FAIL", detail=f"build failed\n{tail}")

    archive_path = build_dir / "libcmsis-nn.a"
    if not archive_path.exists():
        return VariantResult(variant=variant, status="FAIL", detail="libcmsis-nn.a was not produced")

    code_bytes, rodata_bytes, data_bytes, bss_bytes, other_bytes = summarize_archive(archive_path, toolchain)
    return VariantResult(
        variant=variant,
        status="PASS",
        archive_bytes=archive_path.stat().st_size,
        code_bytes=code_bytes,
        rodata_bytes=rodata_bytes,
        data_bytes=data_bytes,
        bss_bytes=bss_bytes,
        other_bytes=other_bytes,
        detail=str(build_dir),
    )


def print_summary(toolchain: Toolchain, results: Iterable[VariantResult]) -> None:
    rows = []
    for result in results:
        rows.append(
            [
                result.variant.key,
                result.status,
                format_bytes(result.archive_bytes),
                format_bytes(result.code_bytes),
                format_bytes(result.rodata_bytes),
                format_bytes(result.data_bytes),
                format_bytes(result.bss_bytes),
                format_bytes(variant_total_bytes(result)),
                format_bytes(result.other_bytes),
            ]
        )

    headers = ["Variant", "Status", "Archive", "Code", "RO", "RW", "ZI", "Total", "Other"]
    widths = [len(h) for h in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print("")
    print(f"CMSIS-NN library variant summary for {toolchain.requested}")
    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator_line = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    failures = [result for result in results if result.status != "PASS"]
    if failures:
        print("")
        print("Failures")
        print("--------")
        for result in failures:
            print(f"{result.variant.key}: {result.detail}")


def parse_variants(selected: str) -> list[Variant]:
    if selected.strip().lower() == "all":
        return list(VARIANTS)
    wanted = {token.strip() for token in selected.split(",") if token.strip()}
    chosen = [variant for variant in VARIANTS if variant.key in wanted]
    missing = sorted(wanted - {variant.key for variant in chosen})
    if missing:
        raise RuntimeError(f"Unknown variant keys: {', '.join(missing)}")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and summarize CMSIS-NN library configuration variants.")
    parser.add_argument(
        "--toolchain",
        required=True,
        help="Toolchain selector, for example GCC@15.2.1 or AC6@6.24.0.",
    )
    parser.add_argument(
        "--toolchain-bin",
        help="Optional toolchain bin directory. If omitted, the script falls back to toolchain-specific environment variables or PATH.",
    )
    parser.add_argument(
        "--cmsis-path",
        default=os.environ.get("CMSIS_PATH", ""),
        help="Optional CMSIS root containing CMSIS/Core/Include. Overrides CMSIS pack resolution.",
    )
    parser.add_argument(
        "--cmsis-pack-root",
        default=os.environ.get("CMSIS_PACK_ROOT", ""),
        help="CMSIS pack root used to resolve ARM::CMSIS when --cmsis-path is not provided.",
    )
    parser.add_argument(
        "--cmsis-version",
        default="6.3.0",
        help="Preferred ARM::CMSIS pack version.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant keys to build. Defaults to 'all'.",
    )
    parser.add_argument(
        "--cpu",
        default="cortex-m55",
        help="Target CPU passed to the toolchain. Defaults to cortex-m55.",
    )
    parser.add_argument(
        "--optimization",
        default="-Ofast",
        help="Value forwarded to CMSIS_OPTIMIZATION_LEVEL. Defaults to -Ofast.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel build jobs. Defaults to the local CPU count.",
    )
    parser.add_argument(
        "--build-root",
        default=str(DEFAULT_BUILD_ROOT),
        help=f"Root directory for per-variant build trees. Defaults to {DEFAULT_BUILD_ROOT}.",
    )
    args = parser.parse_args()

    toolchain = resolve_toolchain(args.toolchain, args.cpu, args.toolchain_bin)
    build_root = Path(args.build_root).resolve()
    cmsis_path = resolve_cmsis_path(args.cmsis_path or None, args.cmsis_pack_root or None, args.cmsis_version)
    variants = parse_variants(args.variants)

    results: list[VariantResult] = []
    failed = False

    for variant in variants:
        print(f"==> Building {variant.label} [{variant.key}] with {toolchain.requested}")
        result = build_variant(REPO_ROOT, build_root, cmsis_path, toolchain, variant, args.cpu, args.optimization, args.jobs)
        results.append(result)
        if result.status != "PASS":
            failed = True
            print(f"    FAIL: {variant.key}")
        else:
            print(f"    PASS: {variant.key}")

    print_summary(toolchain, results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
