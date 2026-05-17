from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
import tomllib
import urllib.request
import wave
import zipfile
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_DIR = WORKSPACE_ROOT / "GPT-SoVITS"
MODELS_ROOT = WORKSPACE_ROOT / "models"
CONFIG_ROOT = WORKSPACE_ROOT / "configs"
VOICE_SETS_DIR = CONFIG_ROOT / "voice-sets"
RUNTIME_ROOT = WORKSPACE_ROOT / "runtime"
RUNTIME_VOICES_ROOT = RUNTIME_ROOT / "voices"
DOWNLOAD_DIR = MODELS_ROOT / "downloads"
PRETRAINED_PACKAGE_DIR = MODELS_ROOT / "pretrained" / "GPT-SoVITS" / "GPT_SoVITS"
PRETRAINED_DIR = PRETRAINED_PACKAGE_DIR / "pretrained_models"
COMPAT_PRETRAINED_DIR = REPO_DIR / "GPT_SoVITS" / "pretrained_models"
COMPAT_G2PW_DIR = REPO_DIR / "GPT_SoVITS" / "text" / "G2PWModel"
ACTIVE_PROFILE_PATH = WORKSPACE_ROOT / "profiles" / "voices.json"
GENSHIN_PROFILE_EXAMPLE_PATH = WORKSPACE_ROOT / "profiles" / "voices.genshin.example.json"
SHARED_PROFILE_EXAMPLE_PATH = WORKSPACE_ROOT / "profiles" / "voices.shared-genshin.example.json"
SAMPLE_VOICE_ID = "genshin-keqing"
SAMPLE_VOICE_NAME = "Keqing Sample"
SAMPLE_VOICE_SET_ID = "default"
SAMPLE_REFERENCE_SPEAKER = "刻晴"


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if value is None:
        value = ""
    return json.dumps(str(value), ensure_ascii=False)


def write_toml_mapping(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    nested: list[tuple[str, dict[str, Any]]] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            nested.append((key, value))
        else:
            lines.append(f"{key} = {toml_value(value)}")
    for table, values in nested:
        lines.append("")
        lines.append(f"[{table}]")
        if values:
            for key, value in values.items():
                lines.append(f"{key} = {toml_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def read_mapping_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".toml":
        with path.open("rb") as file:
            payload = tomllib.load(file)
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain an object/table: {path}")
    return payload

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
    "sv/pretrained_eres2netv2w24s4ep4.ckpt": PRETRAINED_DIR
    / "sv"
    / "pretrained_eres2netv2w24s4ep4.ckpt",
}

UNUSED_PRETRAINED_PATHS = [
    "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    "s2G488k.pth",
    "s2D488k.pth",
    "s2Gv3.pth",
    "gsv-v2final-pretrained",
    "gsv-v4-pretrained",
    "models--nvidia--bigvgan_v2_24khz_100band_256x",
    "v2Pro/s2Gv2Pro.pth",
    "v2Pro/s2Dv2Pro.pth",
    "v2Pro/s2Dv2ProPlus.pth",
]

GENSHIN_REPO_ID = "UnlimitedBurst/GPT-SoVITS"
GENSHIN_ROOT = "原神（已更新4.8）"
GENSHIN_MODEL_DIR = MODELS_ROOT / "voices" / "hf" / "UnlimitedBurst__GPT-SoVITS" / GENSHIN_ROOT
GENSHIN_DEFAULT_SPEAKERS = ["派蒙", "刻晴", "可莉"]
GENSHIN_EXTENDED_SPEAKERS = [
    "派蒙",
    "刻晴",
    "可莉",
    "胡桃",
    "甘雨",
    "雷电将军",
    "纳西妲",
    "神里绫华",
    "八重神子",
    "钟离",
]
GENSHIN_VOICE_IDS = {
    "派蒙": "genshin-paimon",
    "刻晴": "genshin-keqing",
    "可莉": "genshin-klee",
    "胡桃": "genshin-hutao",
    "甘雨": "genshin-ganyu",
    "雷电将军": "genshin-raiden-shogun",
    "纳西妲": "genshin-nahida",
    "神里绫华": "genshin-kamisato-ayaka",
    "八重神子": "genshin-yae-miko",
    "钟离": "genshin-zhongli",
}

SHARED_REPO_ID = "AI-Hobbyist/GPT-SoVits-V2-models"
SHARED_MODEL_DIR = MODELS_ROOT / "voices" / "hf" / "AI-Hobbyist__GPT-SoVits-V2-models"
SHARED_MODEL_INDEX_PATH = SHARED_MODEL_DIR / "shared_models.json"
SHARED_REFERENCE_REPO_ID = "AquaV/genshin-voices-separated"
SHARED_REFERENCE_DIR = MODELS_ROOT / "reference-audio" / "hf" / "AquaV__genshin-voices-separated"
SHARED_REFERENCE_INDEX_PATH = SHARED_REFERENCE_DIR / "reference_audios.json"
SHARED_REFERENCE_DEFAULT_CHARACTERS = ["Furina", "Keqing", "Klee", "Zhongli", "Nahida"]
SHARED_REFERENCE_DEFAULT_LANGUAGES = ["English(US)", "Japanese"]
SHARED_REFERENCE_LANGUAGE_PRESETS = {
    "English(US)": {
        "preset": "genshin-en",
        "lang": "en",
        "suffix": "en",
    },
    "Japanese": {
        "preset": "genshin-ja",
        "lang": "ja",
        "suffix": "ja",
    },
}
SHARED_REFERENCE_CHARACTER_SLUGS = {
    "Furina": "furina",
    "Keqing": "keqing",
    "Klee": "klee",
    "Zhongli": "zhongli",
    "Nahida": "nahida",
    "Raiden Shogun": "raiden-shogun",
    "Hu Tao": "hutao",
    "Ganyu": "ganyu",
    "Kamisato Ayaka": "kamisato-ayaka",
    "Yae Miko": "yae-miko",
}
SHARED_PRESETS = {
    "genshin-en": {
        "name": "Genshin Impact EN 5.1",
        "language": "en",
        "model_id": "ai-hobbyist-genshin-en-5.1",
        "gpt": "Genshin_Impact/EN/GPT_GenshinImpact_EN_5.1.ckpt",
        "sovits": "Genshin_Impact/EN/SV_GenshinImpact_EN_5.1.pth",
        "notes": "Shared GPT-SoVITS v2 weights. Add reference audio and prompt text before exposing voices.",
    },
    "genshin-ja": {
        "name": "Genshin Impact JA 5.1",
        "language": "ja",
        "model_id": "ai-hobbyist-genshin-ja-5.1",
        "gpt": "Genshin_Impact/JA/GPT_GenshinImpact_JA_5.1.ckpt",
        "sovits": "Genshin_Impact/JA/SV_GenshinImpact_JA_5.1.pth",
        "notes": "Shared GPT-SoVITS v2 weights. Add reference audio and prompt text before exposing voices.",
    },
    "wuthering-cn": {
        "name": "Wuthering Waves CN 1.3",
        "language": "zh",
        "model_id": "ai-hobbyist-wuthering-cn-1.3",
        "gpt": "Wuthering_Waves/CN/GPT_WutheringWaves_CN_1.3.ckpt",
        "sovits": "Wuthering_Waves/CN/SV_WutheringWaves_CN_1.3.pth",
        "notes": "Shared GPT-SoVITS v2 weights. Add reference audio and prompt text before exposing voices.",
    },
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
        "--sample-reference",
        action="store_true",
        help="Download only the single bundled reference voice used by the default voice set.",
    )
    parser.add_argument(
        "--activate-voices",
        action="store_true",
        help="Update configs/voice-sets/default.toml and runtime/voices/<id>/voice.toml.",
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


def prune_unused_pretrained_assets() -> None:
    for root in (PRETRAINED_DIR, COMPAT_PRETRAINED_DIR):
        for relative_path in UNUSED_PRETRAINED_PATHS:
            target = root / relative_path
            if not target.exists():
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            print(f"Pruned unused pretrained asset: {target}")


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
    print(f"SV model    : {PRETRAINED_DIR / 'sv' / 'pretrained_eres2netv2w24s4ep4.ckpt'}")


def split_speakers(raw_speakers: str) -> list[str]:
    speakers = [speaker.strip() for speaker in raw_speakers.split(",") if speaker.strip()]
    if not speakers:
        raise SystemExit("--genshin-speakers did not contain any speaker names.")
    return speakers


def split_presets(raw_presets: str) -> list[str]:
    presets = [preset.strip() for preset in raw_presets.split(",") if preset.strip()]
    if not presets:
        raise SystemExit("--shared-presets did not contain any preset names.")
    unknown = [preset for preset in presets if preset not in SHARED_PRESETS]
    if unknown:
        raise SystemExit(f"Unknown shared presets: {', '.join(unknown)}. Supported: {', '.join(SHARED_PRESETS)}")
    return presets


def split_csv(raw_value: str, *, field_name: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise SystemExit(f"{field_name} did not contain any values.")
    return values


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


def read_profiles(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("voices", [])
    if not isinstance(data, list):
        raise SystemExit(f"Profile file must contain a list or voices object: {path}")
    return [item for item in data if isinstance(item, dict)]


def merge_active_profiles(profiles: list[dict[str, Any]]) -> None:
    existing = read_profiles(ACTIVE_PROFILE_PATH)
    merged: dict[str, dict[str, Any]] = {}
    for profile in [*existing, *profiles]:
        profile_id = str(profile.get("id") or profile.get("name") or "").strip()
        if profile_id:
            merged[profile_id] = profile
    ACTIVE_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_PROFILE_PATH.write_text(
        json.dumps({"voices": list(merged.values())}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Activated/merged profiles: {ACTIVE_PROFILE_PATH}")


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
        try:
            ckpt_repo_path = choose_genshin_file(files, speaker, ".ckpt")
            pth_repo_path = choose_genshin_file(files, speaker, ".pth")
            ref_repo_path = choose_genshin_ref(files, speaker)
        except SystemExit as exc:
            print(f"WARNING: skipping Genshin speaker {speaker}: {exc}")
            continue

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

    if not profiles:
        raise SystemExit("No Genshin voices were downloaded or available for profile generation.")

    write_genshin_profiles(profiles, activate=activate)
    print("")
    print("Genshin demo voices")
    for profile in profiles:
        print(f"- {profile['id']}: {profile['name']} ({profile['prompt_text']})")


def write_single_sample_voice(*, reference_audio: Path, prompt_text: str, activate: bool) -> None:
    voice_dir = RUNTIME_VOICES_ROOT / SAMPLE_VOICE_ID
    voice_dir.mkdir(parents=True, exist_ok=True)
    voice_payload = {
        "schema_version": 1,
        "id": SAMPLE_VOICE_ID,
        "name": SAMPLE_VOICE_NAME,
        "mode": "prompt_clone",
        "model_preset": "v2proplus-clone",
        "reference_audio": workspace_relative(reference_audio),
        "prompt_audio": "",
        "prompt_text": prompt_text,
        "text_lang": "zh",
        "prompt_lang": "zh",
        "instruction": "",
        "speed": 1.0,
        "engine_options": {},
    }
    voice_profile_path = voice_dir / "voice.toml"
    write_toml_mapping(voice_profile_path, voice_payload)
    (voice_dir / "voice.json").unlink(missing_ok=True)
    print(f"Wrote sample voice: {voice_profile_path}")

    if activate:
        VOICE_SETS_DIR.mkdir(parents=True, exist_ok=True)
        voice_set_path = VOICE_SETS_DIR / f"{SAMPLE_VOICE_SET_ID}.toml"
        legacy_voice_set_path = VOICE_SETS_DIR / f"{SAMPLE_VOICE_SET_ID}.json"
        if voice_set_path.exists():
            voice_set = read_mapping_file(voice_set_path)
        elif legacy_voice_set_path.exists():
            voice_set = read_mapping_file(legacy_voice_set_path)
        else:
            voice_set = {
                "schema_version": 1,
                "id": SAMPLE_VOICE_SET_ID,
                "name": "Default",
                "description": "Default voices exposed as OpenAI TTS models.",
                "voices": [],
            }
        voices = [item for item in voice_set.get("voices", []) if item != SAMPLE_VOICE_ID]
        voice_set["voices"] = [*voices, SAMPLE_VOICE_ID]
        write_toml_mapping(voice_set_path, voice_set)
        legacy_voice_set_path.unlink(missing_ok=True)
        print(f"Activated sample voice in: {voice_set_path}")


def maybe_download_sample_reference(*, activate: bool, force: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Run `pixi run install-deps` first.") from exc

    print(f"Listing sample reference files from Hugging Face: {GENSHIN_REPO_ID}")
    files = HfApi().list_repo_files(repo_id=GENSHIN_REPO_ID, repo_type="model")
    ref_repo_path = choose_genshin_ref(files, SAMPLE_REFERENCE_SPEAKER)
    target = RUNTIME_VOICES_ROOT / SAMPLE_VOICE_ID / "reference.wav"
    download_huggingface_model_file(GENSHIN_REPO_ID, ref_repo_path, target, force=force)
    write_single_sample_voice(reference_audio=target, prompt_text=prompt_from_reference(ref_repo_path), activate=activate)
    print("")
    print("Single sample reference voice")
    print(f"- {SAMPLE_VOICE_ID}: {SAMPLE_VOICE_NAME} ({prompt_from_reference(ref_repo_path)})")


def maybe_download_shared_multispeaker(repo_id: str, presets: list[str], *, force: bool) -> None:
    print(f"Downloading shared multi-speaker GPT-SoVITS weights from Hugging Face: {repo_id}")
    records = []
    for preset_name in presets:
        preset = SHARED_PRESETS[preset_name]
        preset_dir = SHARED_MODEL_DIR / preset_name
        gpt_target = preset_dir / Path(preset["gpt"]).name
        sovits_target = preset_dir / Path(preset["sovits"]).name

        download_huggingface_model_file(repo_id, preset["gpt"], gpt_target, force=force)
        download_huggingface_model_file(repo_id, preset["sovits"], sovits_target, force=force)

        records.append(
            {
                "preset": preset_name,
                "repo_id": repo_id,
                "name": preset["name"],
                "model_id": preset["model_id"],
                "model_type": "shared-trained",
                "language": preset["language"],
                "gpt_weights_path": workspace_relative(gpt_target),
                "sovits_weights_path": workspace_relative(sovits_target),
                "requires_reference_audio": True,
                "notes": preset["notes"],
            }
        )

    SHARED_MODEL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"models": records}
    SHARED_MODEL_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote shared model index: {SHARED_MODEL_INDEX_PATH}")
    print("")
    print("Shared multi-speaker model presets")
    for record in records:
        print(f"- {record['preset']}: {record['name']} ({record['language']})")


def download_huggingface_dataset_file(repo_id: str, repo_path: str, target: Path, *, force: bool) -> Path:
    if target.exists() and not force:
        print(f"Exists, skipping: {target}")
        return target
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Run `pixi run install-deps` first.") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and force:
        target.unlink()
    print(f"Downloading HF dataset file: {repo_id}/{repo_path}")
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
            force_download=force,
        )
    )
    print(f"Copying {downloaded} -> {target}")
    shutil.copyfile(downloaded, target)
    return target


def audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    with wave.open(str(path), "rb") as wav_file:
        return float(wav_file.getnframes()) / float(wav_file.getframerate())


def choose_shared_reference(
    *,
    repo_id: str,
    character: str,
    language: str,
    force: bool,
    min_seconds: float = 3.05,
    max_seconds: float = 9.95,
    scan_limit: int = 80,
) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Run `pixi run install-deps` first.") from exc

    repo_prefix = f"{character}/{language}"
    print(f"Searching reference audio: {repo_id}/{repo_prefix}")
    files = [item.path for item in HfApi().list_repo_tree(repo_id, repo_type="dataset", path_in_repo=repo_prefix)]
    metadata_files = sorted(path for path in files if path.endswith("_metadata.json"))
    if not metadata_files:
        raise SystemExit(f"No metadata files found for {repo_prefix}")

    language_dir = language.replace("(", "_").replace(")", "").replace("/", "_")
    target_dir = SHARED_REFERENCE_DIR / character / language_dir

    fallback: dict[str, Any] | None = None
    for metadata_repo_path in metadata_files[:scan_limit]:
        stem = metadata_repo_path.removesuffix("_metadata.json")
        audio_repo_path = f"{stem}_audio.wav"
        if audio_repo_path not in files:
            continue
        metadata_target = target_dir / Path(metadata_repo_path).name
        audio_target = target_dir / Path(audio_repo_path).name
        download_huggingface_dataset_file(repo_id, metadata_repo_path, metadata_target, force=force)
        metadata = json.loads(metadata_target.read_text(encoding="utf-8-sig"))
        prompt_text = str(metadata.get("transcription") or "").strip()
        if not prompt_text:
            continue
        if any(marker in prompt_text for marker in ("{", "}", "#")):
            print(f"Skipping templated prompt text: {prompt_text}")
            continue
        download_huggingface_dataset_file(repo_id, audio_repo_path, audio_target, force=force)
        duration = audio_duration_seconds(audio_target)
        record = {
            "repo_id": repo_id,
            "character": character,
            "language": language,
            "prompt_text": prompt_text,
            "duration_seconds": duration,
            "ref_audio_path": workspace_relative(audio_target),
            "metadata_path": workspace_relative(metadata_target),
            "source_audio_path": audio_repo_path,
            "source_metadata_path": metadata_repo_path,
        }
        if fallback is None:
            fallback = record
        if min_seconds <= duration <= max_seconds:
            return record

    if fallback is not None:
        print(
            "WARNING: no 3-10s sample found in scan window; "
            f"using first readable sample at {fallback['duration_seconds']:.3f}s."
        )
        return fallback
    raise SystemExit(f"No usable reference audio found for {repo_prefix}")


def write_shared_reference_profiles(profiles: list[dict[str, Any]], *, activate: bool) -> None:
    payload = {"voices": profiles}
    SHARED_PROFILE_EXAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHARED_PROFILE_EXAMPLE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote shared profile example: {SHARED_PROFILE_EXAMPLE_PATH}")
    if activate:
        merge_active_profiles(profiles)


def maybe_download_shared_reference_demo(
    *,
    weight_repo_id: str,
    reference_repo_id: str,
    characters: list[str],
    languages: list[str],
    activate: bool,
    force: bool,
) -> None:
    unknown_languages = [language for language in languages if language not in SHARED_REFERENCE_LANGUAGE_PRESETS]
    if unknown_languages:
        supported = ", ".join(SHARED_REFERENCE_LANGUAGE_PRESETS)
        raise SystemExit(f"Unsupported shared reference languages: {', '.join(unknown_languages)}. Supported: {supported}")

    needed_presets = sorted(
        {SHARED_REFERENCE_LANGUAGE_PRESETS[language]["preset"] for language in languages}
    )
    maybe_download_shared_multispeaker(weight_repo_id, needed_presets, force=force)

    reference_records = []
    profiles = []
    for language in languages:
        language_info = SHARED_REFERENCE_LANGUAGE_PRESETS[language]
        preset = SHARED_PRESETS[language_info["preset"]]
        preset_dir = SHARED_MODEL_DIR / language_info["preset"]
        gpt_target = preset_dir / Path(preset["gpt"]).name
        sovits_target = preset_dir / Path(preset["sovits"]).name
        for character in characters:
            reference = choose_shared_reference(
                repo_id=reference_repo_id,
                character=character,
                language=language,
                force=force,
            )
            reference_records.append(reference)
            slug = SHARED_REFERENCE_CHARACTER_SLUGS.get(character, character.lower().replace(" ", "-"))
            voice_id = f"shared-genshin-{language_info['suffix']}-{slug}"
            profiles.append(
                {
                    "id": voice_id,
                    "name": f"{character} ({language_info['suffix'].upper()} shared)",
                    "description": (
                        f"{character} reference from {reference_repo_id}; "
                        f"shared weights from {weight_repo_id}."
                    ),
                    "model_id": preset["model_id"],
                    "model_name": preset["name"],
                    "model_type": "shared-trained",
                    "ref_audio_path": reference["ref_audio_path"],
                    "prompt_text": reference["prompt_text"],
                    "prompt_lang": language_info["lang"],
                    "text_lang": language_info["lang"],
                    "aux_ref_audio_paths": [],
                    "gpt_weights_path": workspace_relative(gpt_target),
                    "sovits_weights_path": workspace_relative(sovits_target),
                }
            )

    SHARED_REFERENCE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHARED_REFERENCE_INDEX_PATH.write_text(
        json.dumps({"references": reference_records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote shared reference index: {SHARED_REFERENCE_INDEX_PATH}")
    write_shared_reference_profiles(profiles, activate=activate)
    print("")
    print("Shared reference voices")
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
    if args.sample_reference:
        maybe_download_sample_reference(activate=args.activate_voices, force=args.force)
    prune_unused_pretrained_assets()
    print("Asset download step finished.")


if __name__ == "__main__":
    main()
