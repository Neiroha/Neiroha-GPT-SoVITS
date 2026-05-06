from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = WORKSPACE_ROOT / "GPT-SoVITS"
MODELS_ROOT = WORKSPACE_ROOT / "models"
DOWNLOAD_DIR = MODELS_ROOT / "downloads"
PRETRAINED_PACKAGE_DIR = MODELS_ROOT / "pretrained" / "GPT-SoVITS" / "GPT_SoVITS"
PRETRAINED_DIR = PRETRAINED_PACKAGE_DIR / "pretrained_models"
COMPAT_PRETRAINED_DIR = REPO_DIR / "GPT_SoVITS" / "pretrained_models"
COMPAT_G2PW_DIR = REPO_DIR / "GPT_SoVITS" / "text" / "G2PWModel"
ACTIVE_PROFILE_PATH = WORKSPACE_ROOT / "profiles" / "voices.json"
GENSHIN_PROFILE_EXAMPLE_PATH = WORKSPACE_ROOT / "profiles" / "voices.genshin.example.json"

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

GENSHIN_REPO_ID = "UnlimitedBurst/GPT-SoVITS"
GENSHIN_ROOT = "原神（已更新4.8）"
GENSHIN_MODEL_DIR = MODELS_ROOT / "voices" / "hf" / "UnlimitedBurst__GPT-SoVITS" / GENSHIN_ROOT
GENSHIN_DEFAULT_SPEAKERS = ["派蒙", "刻晴", "可莉"]
GENSHIN_VOICE_IDS = {
    "派蒙": "genshin-paimon",
    "刻晴": "genshin-keqing",
    "可莉": "genshin-klee",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GPT-SoVITS model assets into the top-level models directory."
    )
    parser.add_argument("--source", choices=sorted(SOURCES), default="modelscope")
    parser.add_argument("--download-uvr5", action="store_true")
    parser.add_argument(
        "--v2pro-plus",
        action="store_true",
        help="Download official v2ProPlus weights from Hugging Face into models/pretrained.",
    )
    parser.add_argument(
        "--skip-base-assets",
        action="store_true",
        help="Skip the common pretrained/G2PW/NLTK/OpenJTalk asset downloads.",
    )
    parser.add_argument(
        "--genshin-demo",
        action="store_true",
        help="Download a small ready-to-switch multi-speaker Genshin GPT-SoVITS demo set.",
    )
    parser.add_argument(
        "--genshin-speakers",
        default=",".join(GENSHIN_DEFAULT_SPEAKERS),
        help="Comma-separated speaker names from UnlimitedBurst/GPT-SoVITS.",
    )
    parser.add_argument(
        "--genshin-repo-id",
        default=GENSHIN_REPO_ID,
        help="Hugging Face repo id for the Genshin speaker models.",
    )
    parser.add_argument(
        "--activate-voices",
        action="store_true",
        help="Write the downloaded demo profiles to profiles/voices.json.",
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


def download_huggingface_model_file(repo_id: str, repo_path: str, target: Path, *, force: bool) -> None:
    if target.exists() and not force:
        print(f"Exists, skipping: {target}")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Run `pixi run install-deps` first.") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and force:
        target.unlink()
    print(f"Downloading HF model file: {repo_id}/{repo_path}")
    downloaded = None
    for attempt in range(1, 4):
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_path,
                    repo_type="model",
                    force_download=force,
                )
            )
            break
        except Exception as exc:
            if attempt >= 3:
                raise
            print(
                f"Download failed on attempt {attempt}/3: {exc}. "
                f"Retrying in {attempt * 5}s..."
            )
            time.sleep(attempt * 5)
    if downloaded is None:
        raise RuntimeError(f"Failed to download {repo_id}/{repo_path}")
    print(f"Copying {downloaded} -> {target}")
    shutil.copyfile(downloaded, target)


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


def sync_compat_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def maybe_download_pretrained(urls: dict[str, str], *, force: bool) -> None:
    markers = [
        PRETRAINED_DIR / "chinese-roberta-wwm-ext-large",
        PRETRAINED_DIR / "chinese-hubert-base",
    ]
    if all(marker.exists() for marker in markers) and not force:
        print("Pretrained models already look present; skipping.")
        if not COMPAT_PRETRAINED_DIR.exists():
            sync_compat_tree(PRETRAINED_DIR, COMPAT_PRETRAINED_DIR)
        return
    archive = download(urls["pretrained"], DOWNLOAD_DIR / "pretrained_models.zip", force=force)
    extract_zip(archive, PRETRAINED_PACKAGE_DIR)
    sync_compat_tree(PRETRAINED_DIR, COMPAT_PRETRAINED_DIR)


def maybe_download_g2pw(urls: dict[str, str], *, force: bool) -> None:
    marker = PRETRAINED_PACKAGE_DIR / "text" / "G2PWModel"
    if marker.exists() and not force:
        print("G2PWModel already present; skipping.")
        if not COMPAT_G2PW_DIR.exists():
            sync_compat_tree(marker, COMPAT_G2PW_DIR)
        return
    archive = download(urls["g2pw"], DOWNLOAD_DIR / "G2PWModel.zip", force=force)
    extract_zip(archive, PRETRAINED_PACKAGE_DIR / "text")
    sync_compat_tree(PRETRAINED_PACKAGE_DIR / "text" / "G2PWModel", COMPAT_G2PW_DIR)


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
    marker = MODELS_ROOT / "tools" / "uvr5" / "uvr5_weights"
    if marker.exists() and not force:
        print("UVR5 weights already present; skipping.")
        return
    archive = download(urls["uvr5"], DOWNLOAD_DIR / "uvr5_weights.zip", force=force)
    extract_zip(archive, MODELS_ROOT / "tools" / "uvr5")


def maybe_download_v2pro_plus(*, force: bool) -> None:
    print("Downloading GPT-SoVITS v2ProPlus weights from Hugging Face.")
    for repo_path, target in V2PRO_PLUS_FILES.items():
        download_huggingface_file("lj1995/GPT-SoVITS", repo_path, target, force=force)
        compat_target = COMPAT_PRETRAINED_DIR / repo_path
        compat_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, compat_target)
    print("")
    print("v2ProPlus local paths")
    print(f"GPT weights : {PRETRAINED_DIR / 's1v3.ckpt'}")
    print(f"SoVITS G    : {PRETRAINED_DIR / 'v2Pro' / 's2Gv2ProPlus.pth'}")
    print(f"SoVITS D    : {PRETRAINED_DIR / 'v2Pro' / 's2Dv2ProPlus.pth'}")
    print(f"SV model    : {PRETRAINED_DIR / 'sv' / 'pretrained_eres2netv2w24s4ep4.ckpt'}")


def split_speakers(raw_speakers: str) -> list[str]:
    speakers = [speaker.strip() for speaker in raw_speakers.split(",") if speaker.strip()]
    if not speakers:
        raise SystemExit("--genshin-speakers did not contain any speaker names.")
    return speakers


def choose_genshin_file(files: list[str], speaker: str, suffix: str) -> str:
    prefix = f"{GENSHIN_ROOT}/{speaker}/"
    matches = [
        item
        for item in files
        if item.startswith(prefix) and item.lower().endswith(suffix) and not item.endswith("train.log")
    ]
    if not matches:
        raise SystemExit(f"No {suffix} file found for Genshin speaker: {speaker}")
    return sorted(matches)[0]


def choose_genshin_ref(files: list[str], speaker: str) -> str:
    prefix = f"{GENSHIN_ROOT}/{speaker}/"
    wavs = [item for item in files if item.startswith(prefix) and item.lower().endswith(".wav")]
    if not wavs:
        raise SystemExit(f"No reference wav file found for Genshin speaker: {speaker}")
    calm_wavs = [item for item in wavs if "平静" in Path(item).stem]
    return sorted(calm_wavs or wavs)[0]


def prompt_from_reference(repo_path: str) -> str:
    prompt = Path(repo_path).stem.strip()
    if "-" in prompt:
        prompt = prompt.split("-", 1)[1].strip()
    return prompt


def workspace_relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()


def write_genshin_profiles(profiles: list[dict[str, str]], *, activate: bool) -> None:
    payload = {"voices": profiles}
    GENSHIN_PROFILE_EXAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    GENSHIN_PROFILE_EXAMPLE_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote profile example: {GENSHIN_PROFILE_EXAMPLE_PATH}")
    if activate:
        ACTIVE_PROFILE_PATH.write_text(text, encoding="utf-8")
        print(f"Activated profiles: {ACTIVE_PROFILE_PATH}")


def maybe_download_genshin_demo(repo_id: str, speakers: list[str], *, activate: bool, force: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Run `pixi run install-deps` first.") from exc

    print(f"Listing Genshin speaker files from Hugging Face: {repo_id}")
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    profiles = []
    for speaker in speakers:
        ckpt_repo_path = choose_genshin_file(files, speaker, ".ckpt")
        pth_repo_path = choose_genshin_file(files, speaker, ".pth")
        ref_repo_path = choose_genshin_ref(files, speaker)

        speaker_dir = GENSHIN_MODEL_DIR / speaker
        ckpt_target = speaker_dir / Path(ckpt_repo_path).name
        pth_target = speaker_dir / Path(pth_repo_path).name
        ref_target = speaker_dir / Path(ref_repo_path).name

        download_huggingface_model_file(repo_id, ckpt_repo_path, ckpt_target, force=force)
        download_huggingface_model_file(repo_id, pth_repo_path, pth_target, force=force)
        download_huggingface_model_file(repo_id, ref_repo_path, ref_target, force=force)

        voice_id = GENSHIN_VOICE_IDS.get(speaker, f"genshin-{speaker}")
        profiles.append(
            {
                "id": voice_id,
                "name": speaker,
                "description": f"{speaker} from {repo_id} {GENSHIN_ROOT}.",
                "model_id": "genshin-4.8",
                "model_name": "Genshin 4.8 (UnlimitedBurst/GPT-SoVITS)",
                "model_type": "trained",
                "ref_audio_path": workspace_relative(ref_target),
                "prompt_text": prompt_from_reference(ref_repo_path),
                "prompt_lang": "zh",
                "text_lang": "zh",
                "aux_ref_audio_paths": [],
                "gpt_weights_path": workspace_relative(ckpt_target),
                "sovits_weights_path": workspace_relative(pth_target),
            }
        )

    write_genshin_profiles(profiles, activate=activate)
    print("")
    print("Genshin demo voices")
    for profile in profiles:
        print(f"- {profile['id']}: {profile['name']} ({profile['prompt_text']})")


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
    if args.genshin_demo:
        maybe_download_genshin_demo(
            args.genshin_repo_id,
            split_speakers(args.genshin_speakers),
            activate=args.activate_voices,
            force=args.force,
        )
    print("Asset download step finished.")


if __name__ == "__main__":
    main()
