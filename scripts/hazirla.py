"""Fill a missing local runtime, then report the device the app should use.

Kraken's dependency install can replace a CUDA torch with a CPU build.
Torch is therefore checked after requirements and pinned again when a GPU
is present. An existing matching install is left untouched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CUDA_TORCH = ("torch==2.11.0+cu128", "torchvision==0.26.0+cu128")
CPU_TORCH = ("torch==2.14.0", "torchvision==0.29.1")
MODULES = (
    "torch",
    "torchvision",
    "kraken",
    "fastapi",
    "uvicorn",
    "gradio",
    "yaml",
    "dotenv",
    "PIL",
    "pypdfium2",
    "httpx",
    "pytest",
)


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    return env


def run(command: list[str]) -> int:
    print("+", " ".join(command))
    return subprocess.run(command, cwd=ROOT, env=runtime_env()).returncode


def python_probe() -> tuple[list[str], bool, str]:
    script = (
        "import importlib,sys\n"
        f"mods={MODULES!r}\n"
        "bad=[]\n"
        "for name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        bad.append(name+':'+type(exc).__name__)\n"
        "cuda=False\n"
        "version=''\n"
        "try:\n"
        "    import torch\n"
        "    cuda=bool(torch.cuda.is_available())\n"
        "    version=torch.__version__\n"
        "except Exception:\n"
        "    pass\n"
        "print('---')\n"
        "print('cuda' if cuda else 'cpu')\n"
        "print(version)\n"
        "print('|'.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=runtime_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = [line.strip() for line in (result.stdout or "").splitlines()]
    if "---" in lines:
        lines = lines[len(lines) - lines[::-1].index("---") :]
    while len(lines) < 3:
        lines.append("")
    missing = [part for part in lines[2].split("|") if part]
    if result.returncode not in (0, None) and not missing:
        detail = (result.stderr or "python kontrolu kirildi").strip().splitlines()
        missing.append(detail[-1] if detail else "python kontrolu kirildi")
    return missing, lines[0] == "cuda", lines[1]


def nvidia_name() -> str:
    candidates = []
    found = shutil.which("nvidia-smi")
    if found:
        candidates.append(found)
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates.append(str(Path(windir) / "System32" / "nvidia-smi.exe"))
    for path in candidates:
        if not Path(path).is_file():
            continue
        result = subprocess.run(
            [path, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
    return ""


def install_requirements() -> None:
    code = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(ROOT / "requirements.txt"),
        ]
    )
    if code != 0:
        raise SystemExit("Paket kurulumu tamamlanamadi.")


def install_torch(index: str, specs: tuple[str, str]) -> bool:
    code = run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--force-reinstall",
            "--no-deps",
            specs[0],
            specs[1],
            "--index-url",
            index,
        ]
    )
    return code == 0


def openrouter_configured() -> bool:
    path = ROOT / ".env"
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return bool(line.split("=", 1)[1].strip())
    return False


def ensure_env() -> None:
    env = ROOT / ".env"
    example = ROOT / ".env.example"
    if env.is_file():
        return
    if not example.is_file():
        raise SystemExit(".env.example yok.")
    env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    print(".env olusturuldu.")


def model_gaps() -> list[str]:
    script = (
        "from app.config import get_settings\n"
        "s=get_settings()\n"
        "pairs=(('tanima', s.recognition_model), ('satir', s.segmentation_model), ('muharaf', s.muharaf_model))\n"
        "print('|'.join(name for name, path in pairs if not path.is_file()))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=runtime_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "ayarlar okunamadi").strip().splitlines()
        return [detail[-1] if detail else "ayarlar okunamadi"]
    return [part for part in (result.stdout or "").strip().split("|") if part]


def download_models() -> None:
    code = run([sys.executable, str(ROOT / "scripts" / "download_models.py")])
    if code != 0:
        raise SystemExit("Model indirme tamamlanamadi.")


def write_device(device: str) -> None:
    target = Path(sys.executable).resolve().parents[1] / ".device"
    target.write_text(device, encoding="utf-8")
    print(f"TARIHHTR_DEVICE={device}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksik calisma ortami ve modelleri tamamla.")
    parser.add_argument("--denetle", action="store_true", help="Kurulum yapmadan durumu yaz.")
    parser.add_argument("--cpu", action="store_true", help="Ekran karti olsa da CPU torch kullan.")
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        raise SystemExit("64-bit Python 3.11 gerekli.")
    for name in ("config.yaml", "requirements.txt", ".env.example"):
        if not (ROOT / name).is_file():
            raise SystemExit(f"Eksik dosya: {name}")

    gpu = nvidia_name()
    want_cuda = bool(gpu) and not args.cpu
    missing, cuda, version = python_probe()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {version or 'yok'}")
    print(f"Ekran karti: {gpu or 'yok'}")

    if args.denetle:
        gaps = list(missing)
        if not missing:
            gaps.extend(model_gaps())
        if not (ROOT / ".env").is_file():
            print(".env yok; baslat olusturur.")
        elif not openrouter_configured():
            print("OpenRouter anahtari yok. Okuma yerel modelle surer.")
        if gpu and not cuda and not args.cpu:
            print("Kart var, torch CPU. Hizli okuma icin baslat.cmd CUDA kurar.")
        if gaps:
            print("Eksik: " + ", ".join(gaps))
            raise SystemExit(1)
        device = "cuda" if cuda and not args.cpu else "cpu"
        print(f"Calisir. Okuma cihazi: {device}")
        return

    if missing:
        print("Eksik paketler kuruluyor.")
        install_requirements()
        missing, cuda, version = python_probe()
        if any(not item.startswith("torch") for item in missing):
            raise SystemExit("Paketler kuruldu ama ice aktarilamadi: " + ", ".join(missing))

    if want_cuda and not cuda:
        print("CUDA torch kuruluyor. Bu indirme yalnizca ilk seferde olur.")
        installed = install_torch(CUDA_INDEX, CUDA_TORCH)
        missing, cuda, version = python_probe()
        if installed and (missing or not cuda):
            print("CUDA acilmadi. CPU torch kuruluyor.")
            if not install_torch(CPU_INDEX, CPU_TORCH):
                raise SystemExit("Torch kurulamadi.")
            missing, cuda, version = python_probe()
        elif not installed:
            print("CUDA kurulamadi. Okuma CPU'da surer.")
            print("Acik arayuz dosyayi kilitliyor olabilir. Kapatip baslat.cmd yeniden calistirin.")
            missing, cuda, version = python_probe()
    elif any(item.startswith("torch") for item in missing):
        print("CPU torch kuruluyor.")
        if not install_torch(CPU_INDEX, CPU_TORCH):
            raise SystemExit("Torch kurulamadi.")
        missing, cuda, version = python_probe()

    if missing:
        raise SystemExit("Kurulumdan sonra eksik: " + ", ".join(missing))

    ensure_env()
    if not openrouter_configured():
        print("OpenRouter anahtari yok. Okuma yerel modelle surer.")
    print("Modeller kontrol ediliyor.")
    download_models()
    remaining = model_gaps()
    if remaining:
        raise SystemExit("Model eksik: " + ", ".join(remaining))

    device = "cuda" if want_cuda and cuda else "cpu"
    print(f"Torch: {version}")
    print("Azra, YOLO ve veri arsivleri bu adimda inmez.")
    write_device(device)


if __name__ == "__main__":
    main()
