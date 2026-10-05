"""Download the line recognizer, the Muharaf weights, and the BLLA segmenter."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import urllib.request

from app.config import get_settings
from app.utils import configure_stdio


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"İndiriliyor: {url}")
    urllib.request.urlretrieve(url, temporary)
    actual = md5_file(temporary)
    if actual != expected_md5:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"MD5 uyuşmadı: {destination.name} beklenen {expected_md5}, gelen {actual}")
    temporary.replace(destination)
    print(f"Tamam: {destination} ({actual})")


def ensure(path: Path, url: str, expected_md5: str) -> None:
    if path.is_file() and md5_file(path) == expected_md5:
        print(f"Mevcut: {path}")
        return
    download(url, path, expected_md5)


def copy_packaged_blla(destination: Path) -> bool:
    try:
        from importlib.resources import files
    except ImportError:
        return False
    source = files("kraken").joinpath("blla.mlmodel")
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    print(f"Paket içinden kopyalandı: {destination}")
    return True


def main() -> None:
    configure_stdio()
    settings = get_settings()
    ensure(settings.recognition_model, settings.recognition_url, settings.recognition_md5)
    ensure(settings.muharaf_model, settings.muharaf_url, settings.muharaf_md5)
    if not settings.segmentation_model.is_file():
        if not copy_packaged_blla(settings.segmentation_model):
            ensure(settings.segmentation_model, settings.blla_url, settings.blla_md5)
    else:
        print(f"Mevcut: {settings.segmentation_model}")
    print("Modeller hazır.")


if __name__ == "__main__":
    main()
