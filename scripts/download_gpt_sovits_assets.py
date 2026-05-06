from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = WORKSPACE_ROOT / "GPT-SoVITS"
DOWNLOAD_DIR = WORKSPACE_ROOT / "runtime" / "downloads"
PRETRAINED_DIR = REPO_DIR / "GPT_SoVITS" / "pretrained_models"

SOURCES = {
    "hf": {
        "pretrained": "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip",
        "g2pw": "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip",
        "uvr5": "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip",
        "nltk": "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip",
        "open_jtalk": "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz",
    },
    "hf-mirror": {
        "pretrained": "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip",
        "g2pw": "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip",
        "uvr5": "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip",
        "nltk": "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip",
        "open_jtalk": "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz",
    },
    "modelscope": {
        "pretrained": "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip",
        "g2pw": "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip",
        "uvr5": "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/uvr5_weights.zip",
        "nltk": "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip",
        "open_jtalk": "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz",
    },
}

HF_PRETRAINED_PREFIX = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/"

V2PRO_PLUS_FILES = {
    "s1v3.ckpt": PRETRAINED_DIR / "s1v3.ckpt",
    "v2Pro/s2Gv2ProPlus.pth": PRETRAINED_DIR / "v2Pro" / "s2Gv2ProPlus.pth",
    "v2Pro/s2Dv2ProPlus.pth": PRETRAINED_DIR / "v2Pro" / "s2Dv2ProPlus.pth",
    "sv/pretrained_eres2netv2w24s4ep4.ckpt": PRETRAINED_DIR
    / "sv"
    / "pretrained_eres2netv2w24s4ep4.ckpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GPT-SoVITS pretrained assets into the submodule."
    )
    parser.add_argument("--source", choices=sorted(SOURCES), default="modelscope")
    parser.add_argument("--download-uvr5", action="store_true")
    parser.add_argument(
        "--v2pro-plus",
        action="store_true",
        help="Download official v2ProPlus weights from Hugging Face into GPT_SoVITS/pretrained_models.",
    )
    parser.add_argument(
        "--skip-base-assets",
        action="store_true",
        help="Skip the common pretrained/G2PW/NLTK/OpenJTalk asset downloads.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download(url: str, target: Path, *, force: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith(HF_PRETRAINED_PREFIX):
        try:
            from huggingface_hub import hf_hub_download

            filename = url.removeprefix(HF_PRETRAINED_PREFIX).split("?", 1)[0]
            downloaded = Path(
                hf_hub_download(
                    repo_id="XXXXRT/GPT-SoVITS-Pretrained",
                    filename=filename,
                    local_dir=str(target.parent),
                    force_download=force,
                )
            )
            if downloaded != target:
                if target.exists() and force:
                    target.unlink()
                if not target.exists():
                    shutil.copyfile(downloaded, target)
            return target
        except ImportError:
            print("huggingface_hub is unavailable; falling back to direct URL download.")

    if target.exists() and not force:
        print(f"Using cached download: {target}")
        return target
    if target.exists() and force:
        target.unlink()
    print(f"Downloading: {url}")
    print(f"Target: {target}")
    with urllib.request.urlopen(url) as response, target.open("wb") as file:
        shutil.copyfileobj(response, file)
    return target


def download_huggingface_file(repo_id: str, repo_path: str, target: Path, *, force: bool) -> None:
    if target.exists() and not force:
        print(f"Exists, skipping: {target}")
        return
    url = f"https://huggingface.co/{repo_id}/resolve/main/{repo_path}?download=true"
    archive = download(url, DOWNLOAD_DIR / repo_path.replace("/", "__"), force=force)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {archive} -> {target}")
    shutil.copyfile(archive, target)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    print(f"Extracting {zip_path.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)


def extract_tar(tar_path: Path, target_dir: Path) -> None:
    print(f"Extracting {tar_path.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as archive:
        archive.extractall(target_dir)


def maybe_download_pretrained(urls: dict[str, str], *, force: bool) -> None:
    markers = [
        PRETRAINED_DIR / "chinese-roberta-wwm-ext-large",
        PRETRAINED_DIR / "chinese-hubert-base",
    ]
    if all(marker.exists() for marker in markers) and not force:
        print("Pretrained models already look present; skipping.")
        return
    archive = download(urls["pretrained"], DOWNLOAD_DIR / "pretrained_models.zip", force=force)
    extract_zip(archive, REPO_DIR / "GPT_SoVITS")


def maybe_download_g2pw(urls: dict[str, str], *, force: bool) -> None:
    marker = REPO_DIR / "GPT_SoVITS" / "text" / "G2PWModel"
    if marker.exists() and not force:
        print("G2PWModel already present; skipping.")
        return
    archive = download(urls["g2pw"], DOWNLOAD_DIR / "G2PWModel.zip", force=force)
    extract_zip(archive, REPO_DIR / "GPT_SoVITS" / "text")


def maybe_download_nltk(urls: dict[str, str], *, force: bool) -> None:
    marker = Path(sys.prefix) / "nltk_data"
    if marker.exists() and not force:
        print("NLTK data already present in this environment; skipping.")
        return
    archive = download(urls["nltk"], DOWNLOAD_DIR / "nltk_data.zip", force=force)
    extract_zip(archive, Path(sys.prefix))


def maybe_download_open_jtalk(urls: dict[str, str], *, force: bool) -> None:
    try:
        import pyopenjtalk
    except ImportError:
        print("pyopenjtalk is not installed yet; run install-deps first, then rerun assets.")
        return

    target = Path(pyopenjtalk.__file__).resolve().parent
    marker = target / "open_jtalk_dic_utf_8-1.11"
    if marker.exists() and not force:
        print("Open JTalk dictionary already present; skipping.")
        return
    archive = download(urls["open_jtalk"], DOWNLOAD_DIR / "open_jtalk_dic_utf_8-1.11.tar.gz", force=force)
    extract_tar(archive, target)


def maybe_download_uvr5(urls: dict[str, str], *, force: bool) -> None:
    marker = REPO_DIR / "tools" / "uvr5" / "uvr5_weights"
    if marker.exists() and not force:
        print("UVR5 weights already present; skipping.")
        return
    archive = download(urls["uvr5"], DOWNLOAD_DIR / "uvr5_weights.zip", force=force)
    extract_zip(archive, REPO_DIR / "tools" / "uvr5")


def maybe_download_v2pro_plus(*, force: bool) -> None:
    print("Downloading GPT-SoVITS v2ProPlus weights from Hugging Face.")
    for repo_path, target in V2PRO_PLUS_FILES.items():
        download_huggingface_file("lj1995/GPT-SoVITS", repo_path, target, force=force)
    print("")
    print("v2ProPlus local paths")
    print(f"GPT weights : {PRETRAINED_DIR / 's1v3.ckpt'}")
    print(f"SoVITS G    : {PRETRAINED_DIR / 'v2Pro' / 's2Gv2ProPlus.pth'}")
    print(f"SoVITS D    : {PRETRAINED_DIR / 'v2Pro' / 's2Dv2ProPlus.pth'}")
    print(f"SV model    : {PRETRAINED_DIR / 'sv' / 'pretrained_eres2netv2w24s4ep4.ckpt'}")


def main() -> None:
    args = parse_args()
    if not REPO_DIR.exists():
        raise SystemExit(
            "GPT-SoVITS submodule is missing. Run `pixi run submodule-init` first."
        )
    urls = SOURCES[args.source]
    if not args.skip_base_assets:
        maybe_download_pretrained(urls, force=args.force)
        maybe_download_g2pw(urls, force=args.force)
        maybe_download_nltk(urls, force=args.force)
        maybe_download_open_jtalk(urls, force=args.force)
        if args.download_uvr5:
            maybe_download_uvr5(urls, force=args.force)
    if args.v2pro_plus:
        maybe_download_v2pro_plus(force=args.force)
    print("Asset download step finished.")


if __name__ == "__main__":
    main()
