#!/usr/bin/env python3
"""Deploy the research LoRA adapter as an Ollama model.

Relative ``--adapter`` / ``--merged-dir`` / ``--staging`` paths resolve under
``modules/llm`` (the parent of ``scripts/``), not the shell cwd.

Requires local tools only:
- torch, peft, transformers, sentencepiece
- Ollama on PATH
- llama.cpp via ``--llama-cpp`` or ``LLAMA_CPP_ROOT``

This script intentionally does not auto-clone llama.cpp. The project guardrails require
local-first, reproducible runtime setup with no silent model-family or cloud fallback.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_BASE_OLLAMA_TAG = "qwen3:8b"
DEFAULT_TAG = "qwen3-research-lora:latest"
DEVICE_ENV_VAR = "RESEARCH_LORA_MERGE_DEVICE"
REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "split_eval_metrics.json",
)


def _workstream_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_project_path(path: Path, root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _abort(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _convert_hf_to_gguf_script(llama_cpp: Path) -> Path | None:
    for candidate in (
        llama_cpp / "convert_hf_to_gguf.py",
        llama_cpp / "tools" / "convert_hf_to_gguf.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _convert_lora_to_gguf_script(llama_cpp: Path) -> Path | None:
    candidate = llama_cpp / "convert_lora_to_gguf.py"
    return candidate if candidate.is_file() else None


def _resolve_llama_cpp(cli_path: Path | None) -> Path:
    candidates: list[Path] = []
    if cli_path is not None:
        candidates.append(cli_path)
    env = os.environ.get("LLAMA_CPP_ROOT", "").strip()
    if env:
        candidates.append(Path(env))

    for candidate in candidates:
        expanded = candidate.expanduser().resolve()
        if _convert_hf_to_gguf_script(expanded) is not None or _convert_lora_to_gguf_script(expanded) is not None:
            return expanded
        raise FileNotFoundError(f"No GGUF conversion scripts found under {expanded}")

    raise FileNotFoundError(
        "llama.cpp is required for GGUF conversion. Set LLAMA_CPP_ROOT or pass --llama-cpp "
        "to a local llama.cpp checkout containing convert_hf_to_gguf.py or convert_lora_to_gguf.py."
    )


def _same_python(python_exe: str) -> bool:
    try:
        return os.path.samefile(python_exe, sys.executable)
    except OSError:
        return False


def _check_convert_python(python_exe: str) -> list[str]:
    problems: list[str] = []
    if _same_python(python_exe):
        if not _module_available("sentencepiece"):
            problems.append(f"{python_exe!r} needs sentencepiece for GGUF conversion.")
        return problems

    result = subprocess.run(
        [python_exe, "-c", "import sentencepiece"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        problems.append(
            f"{python_exe!r} needs sentencepiece for GGUF conversion. "
            f"Install with: {python_exe} -m pip install sentencepiece"
        )
    return problems


def _validate_adapter(adapter_dir: Path, expected_base: str) -> list[str]:
    problems = []
    if not adapter_dir.is_dir():
        return [f"Adapter directory does not exist: {adapter_dir}"]

    for filename in REQUIRED_ADAPTER_FILES:
        if not (adapter_dir / filename).is_file():
            problems.append(f"Missing adapter file: {adapter_dir / filename}")

    config_path = adapter_dir / "adapter_config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"Invalid adapter_config.json: {exc}")
        else:
            base_name = config.get("base_model_name_or_path")
            if base_name != expected_base:
                problems.append(
                    "Adapter base mismatch: "
                    f"adapter_config.json has {base_name!r}, expected {expected_base!r}."
                )

    return problems


def _merge_device() -> str:
    import torch

    env = os.environ.get(DEVICE_ENV_VAR, "").strip().lower()
    if env in ("cpu", "cuda", "mps"):
        return env
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _merge_lora(*, base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problems = _validate_adapter(adapter_dir, base_model)
    if problems:
        raise RuntimeError("\n".join(problems))

    merged_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    device = _merge_device()

    print(f"Loading {base_model} on {device} for PEFT merge (no device_map='auto') ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    base = base.to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload(safe_merge=True)

    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = model.cpu()
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Saved merged HF model to {merged_dir}", flush=True)


def _convert_gguf(*, python_exe: str, llama_cpp: Path, merged_dir: Path, gguf_path: Path) -> None:
    convert_script = _convert_hf_to_gguf_script(llama_cpp)
    if convert_script is None:
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found under {llama_cpp}")

    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(convert_script),
        str(merged_dir),
        "--outfile",
        str(gguf_path),
        "--outtype",
        "f16",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(llama_cpp))


def _convert_lora_gguf(
    *,
    python_exe: str,
    llama_cpp: Path,
    adapter_dir: Path,
    adapter_gguf_path: Path,
    base_model: str,
    adapter_outtype: str,
) -> None:
    convert_script = _convert_lora_to_gguf_script(llama_cpp)
    if convert_script is None:
        raise FileNotFoundError(f"convert_lora_to_gguf.py not found under {llama_cpp}")

    adapter_gguf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(convert_script),
        "--base-model-id",
        base_model,
        "--outfile",
        str(adapter_gguf_path),
        "--outtype",
        adapter_outtype,
        str(adapter_dir),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(llama_cpp))


def _qwen3_ollama_template() -> str:
    im_s = "<|" + "im_start" + "|>"
    im_e = "<|" + "im_end" + "|>"
    return (
        'TEMPLATE """\n'
        "{{- if .System }}"
        + im_s
        + "system\n{{ .System }}"
        + im_e
        + "\n{{- end }}\n"
        "{{- range .Messages }}\n"
        '{{- if eq .Role "user" }}'
        + im_s
        + "user\n{{ .Content }}"
        + im_e
        + "\n"
        '{{- else if eq .Role "assistant" }}'
        + im_s
        + "assistant\n{{ .Content }}"
        + im_e
        + "\n"
        "{{- end }}\n"
        "{{- end }}\n"
        + im_s
        + "assistant\n"
        '"""\n'
    )


def _write_modelfile(staging: Path, gguf_name: str) -> None:
    im_start = "<|" + "im_start" + "|>"
    im_end = "<|" + "im_end" + "|>"
    body = (
        f"FROM ./{gguf_name}\n"
        + _qwen3_ollama_template()
        + """PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER top_k 20
"""
        + f'PARAMETER stop "{im_end}"\n'
        + 'PARAMETER stop "<|endoftext|>"\n'
        + f'PARAMETER stop "{im_start}"\n'
    )
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "Modelfile").write_text(body, encoding="utf-8")


def _write_adapter_modelfile(staging: Path, base_tag: str, adapter_gguf_name: str) -> None:
    im_start = "<|" + "im_start" + "|>"
    im_end = "<|" + "im_end" + "|>"
    body = (
        f"FROM {base_tag}\n"
        f"ADAPTER ./{adapter_gguf_name}\n"
        + _qwen3_ollama_template()
        + """PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER top_k 20
"""
        + f'PARAMETER stop "{im_end}"\n'
        + 'PARAMETER stop "<|endoftext|>"\n'
        + f'PARAMETER stop "{im_start}"\n'
    )
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "Modelfile").write_text(body, encoding="utf-8")


def _ollama_bin() -> str:
    ollama = shutil.which("ollama")
    if not ollama:
        raise FileNotFoundError("ollama not found on PATH; install/start Ollama and retry.")
    return ollama


def _ollama_create(*, staging: Path, model_tag: str) -> None:
    modelfile = (staging / "Modelfile").resolve()
    cmd = [_ollama_bin(), "create", model_tag, "-f", str(modelfile)]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(staging))


def _has_merged_weights(merged_dir: Path) -> bool:
    return (merged_dir / "model.safetensors").is_file() or (merged_dir / "model.safetensors.index.json").is_file()


def _ollama_model_exists(model_tag: str) -> bool:
    if shutil.which("ollama") is None:
        return False
    result = subprocess.run(
        [_ollama_bin(), "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and any(line.split()[0] == model_tag for line in result.stdout.splitlines()[1:])


def _run_check_only(
    *,
    adapter: Path,
    base_model: str,
    python_exe: str,
    llama_cpp_cli: Path | None,
    merge_mode: str,
    base_ollama_tag: str,
) -> int:
    problems = _validate_adapter(adapter, base_model)
    required_modules = ("torch", "transformers") if merge_mode == "adapter" else ("torch", "peft", "transformers")
    for module in required_modules:
        if not _module_available(module):
            problems.append(f"Missing Python package: {module}")
    problems.extend(_check_convert_python(python_exe))
    if shutil.which("ollama") is None:
        problems.append("ollama not found on PATH")

    llama_status = "not configured"
    try:
        llama_cpp = _resolve_llama_cpp(llama_cpp_cli)
    except FileNotFoundError as exc:
        llama_status = str(exc)
    else:
        llama_status = str(llama_cpp)

    print("Research LoRA deployment check")
    print(f"Mode: {merge_mode}")
    print(f"Adapter: {adapter}")
    print(f"Base model: {base_model}")
    print(f"Base Ollama tag: {base_ollama_tag}")
    print(f"Ollama: {shutil.which('ollama') or 'not found'}")
    print(f"llama.cpp: {llama_status}")
    base_tag_exists = True
    if merge_mode == "adapter" and shutil.which("ollama") is not None:
        base_tag_exists = _ollama_model_exists(base_ollama_tag)
        print(f"Base tag in Ollama: {'yes' if base_tag_exists else 'no'}")
        if not base_tag_exists:
            problems.append(f"Ollama base tag is missing or unreachable: {base_ollama_tag}. Run: ollama pull {base_ollama_tag}")
    if problems:
        print("Blocking problems:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    if "required" in llama_status or "not configured" in llama_status:
        print("Not ready for full deployment until llama.cpp is configured.")
        return 0
    print("Ready for full deployment.")
    return 0


def main() -> int:
    """Merge or convert the research LoRA adapter and register an Ollama model tag."""
    root = _workstream_root()
    parser = argparse.ArgumentParser(
        description="Deploy Qwen3 research LoRA to Ollama."
    )
    parser.add_argument(
        "--merge-mode",
        choices=("adapter", "full"),
        default="adapter",
        help="adapter keeps qwen3:8b as the Ollama base; full creates a large merged F16 GGUF",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HF id for base weights")
    parser.add_argument("--base-ollama-tag", default=DEFAULT_BASE_OLLAMA_TAG, help="Ollama base tag for adapter mode")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("models/adapters/research_lora_adapter"),
        help="PEFT adapter directory",
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=Path("models/merged/qwen3-research-lora-merged"),
        help="Merged HF output directory",
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=Path("models/ollama/qwen3-research-lora"),
        help="GGUF and Modelfile staging directory",
    )
    parser.add_argument("--gguf-name", default="qwen3-research-lora-f16.gguf", help="GGUF filename in staging")
    parser.add_argument(
        "--adapter-gguf-name",
        default="qwen3-research-lora-adapter.gguf",
        help="LoRA adapter GGUF filename in staging",
    )
    parser.add_argument(
        "--adapter-outtype",
        choices=("f32", "f16", "bf16", "q8_0", "auto"),
        default="f16",
        help="output type for convert_lora_to_gguf.py in adapter mode",
    )
    parser.add_argument(
        "--tag",
        default=os.environ.get("OLLAMA_MODEL_NAME", DEFAULT_TAG),
        help=f"Ollama tag (default: OLLAMA_MODEL_NAME or {DEFAULT_TAG})",
    )
    parser.add_argument("--llama-cpp", type=Path, default=None, help="local llama.cpp checkout")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for convert_hf_to_gguf.py; must have sentencepiece",
    )
    parser.add_argument("--skip-merge", action="store_true", help="Skip PEFT merge; require existing merged HF model")
    parser.add_argument("--check-only", action="store_true", help="Validate local inputs/prerequisites without merging")
    args = parser.parse_args()

    adapter = _resolve_project_path(args.adapter, root)
    merged_dir = _resolve_project_path(args.merged_dir, root)
    staging = _resolve_project_path(args.staging, root)
    gguf_path = staging / args.gguf_name
    adapter_gguf_path = staging / args.adapter_gguf_name

    if args.check_only:
        return _run_check_only(
            adapter=adapter,
            base_model=args.base_model,
            python_exe=args.python,
            llama_cpp_cli=args.llama_cpp,
            merge_mode=args.merge_mode,
            base_ollama_tag=args.base_ollama_tag,
        )

    problems = _validate_adapter(adapter, args.base_model)
    problems.extend(_check_convert_python(args.python))
    if problems:
        return _abort("\n".join(problems))

    try:
        llama_cpp = _resolve_llama_cpp(args.llama_cpp)
    except FileNotFoundError as exc:
        return _abort(str(exc))

    print(f"Using llama.cpp at {llama_cpp}", flush=True)
    if args.merge_mode == "adapter":
        if not _ollama_model_exists(args.base_ollama_tag):
            return _abort(
                f"Ollama base tag {args.base_ollama_tag!r} is not available. "
                f"Run: ollama pull {args.base_ollama_tag}"
            )
        print("Converting PEFT LoRA adapter to GGUF. This avoids the large merged F16 model.", flush=True)
        try:
            _convert_lora_gguf(
                python_exe=args.python,
                llama_cpp=llama_cpp,
                adapter_dir=adapter,
                adapter_gguf_path=adapter_gguf_path,
                base_model=args.base_model,
                adapter_outtype=args.adapter_outtype,
            )
        except subprocess.CalledProcessError as exc:
            return _abort(f"LoRA adapter GGUF conversion failed with exit code {exc.returncode}.")
        print(f"Wrote {adapter_gguf_path}", flush=True)
        _write_adapter_modelfile(staging, args.base_ollama_tag, args.adapter_gguf_name)
    else:
        print("Full merge mode selected. This can create a 15+ GB GGUF and use substantial RAM.", flush=True)
        if not args.skip_merge:
            try:
                _merge_lora(base_model=args.base_model, adapter_dir=adapter, merged_dir=merged_dir)
            except Exception as exc:
                return _abort(f"PEFT merge failed: {type(exc).__name__}: {exc}")
        elif not _has_merged_weights(merged_dir):
            return _abort(f"Merged model not found under {merged_dir}. Run without --skip-merge first.")

        print("Converting merged HF model to GGUF. This may take several minutes.", flush=True)
        try:
            _convert_gguf(python_exe=args.python, llama_cpp=llama_cpp, merged_dir=merged_dir, gguf_path=gguf_path)
        except subprocess.CalledProcessError as exc:
            return _abort(f"GGUF conversion failed with exit code {exc.returncode}.")
        print(f"Wrote {gguf_path}", flush=True)
        _write_modelfile(staging, args.gguf_name)

    print(f"Wrote {staging / 'Modelfile'}", flush=True)

    try:
        _ollama_create(staging=staging, model_tag=args.tag)
    except FileNotFoundError as exc:
        return _abort(str(exc))
    except subprocess.CalledProcessError as exc:
        return _abort(f"ollama create failed with exit code {exc.returncode}.")

    print("Done. Listing Ollama models:", flush=True)
    subprocess.run([_ollama_bin(), "list"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
