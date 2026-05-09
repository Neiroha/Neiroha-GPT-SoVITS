# Public Trained Model Candidates

Snapshot date: 2026-05-07. These are community model packs for local testing
only. Character voice models often carry game/audio copyright risk even when the
repository itself has an open-source license field.

## Best Fit

### UnlimitedBurst/GPT-SoVITS

- URL: https://huggingface.co/UnlimitedBurst/GPT-SoVITS
- Shape: many per-character folders across Genshin, Honkai, Star Rail, ZZZ and
  Blue Archive style collections.
- Assets: `.ckpt`, `.pth`, and reference `.wav` are available per character in
  the tested Genshin folders.
- Compatibility: v2-style GPT-SoVITS trained voices.
- License/risk: Hugging Face shows MIT, but the model card says the uploader is
  sharing another author's files and asks rights holders to contact them for
  removal. Treat as local testing only.
- Neiroha fit: best current target. `scripts/download_gpt_sovits_assets.py`
  already supports downloading selected Genshin speakers and writing
  `profiles/voices.json`.

Default speakers used by this repo:

```text
派蒙,刻晴,可莉
```

## Secondary Candidates

### AI-Hobbyist/GPT-SoVits-V2-models

- URL: https://huggingface.co/AI-Hobbyist/GPT-SoVits-V2-models
- Shape: shared language/work weights with many listed game characters.
- Assets: model card says `GPT*` files are GPT weights and `SV*` files are
  SoVITS weights.
- Local downloader presets: `genshin-en`, `genshin-ja`, `wuthering-cn`.
- Compatibility: v2.
- Caveat: the model card says it does not provide reference audio, so profiles
  need separate `ref_audio_path` and `prompt_text`.
- Local reference helper: `scripts/download_gpt_sovits_assets.py
  --shared-reference-demo` can pair the Genshin EN/JA presets with short
  reference clips from `AquaV/genshin-voices-separated` and write
  `profiles/voices.shared-genshin.example.json`.
- License/risk: Hugging Face shows AGPL-3.0 and the model card also forbids
  redistribution and commercial use. Treat as local testing only.
- Neiroha fit: good for testing a shared-weight multi-role profile layout after
  reference audio is supplied.

### AquaV/genshin-voices-separated

- URL: https://huggingface.co/datasets/AquaV/genshin-voices-separated
- Shape: separated Genshin voice lines by character and language.
- Assets: wav files plus metadata JSON containing `transcription`, `language`
  and `speaker`.
- Compatibility: useful as reference audio for GPT-SoVITS profiles when the
  selected line is around 3-10 seconds.
- Caveat: the dataset is large, so the downloader only pulls a few small
  metadata/audio pairs for selected characters and languages.
- Neiroha fit: pairs well with AI-Hobbyist shared Genshin EN/JA weights because
  it provides the missing `ref_audio_path` + `prompt_text`.

### baicai1145/GPT-SoVITS-STAR

- URL: https://huggingface.co/baicai1145/GPT-SoVITS-STAR
- Shape: Star Rail character model zips, about 52 characters according to the
  model card.
- Assets: likely per-character GPT/SoVITS pairs inside zips, but each zip should
  be inspected before writing profiles.
- Compatibility: model card says version 2.0.
- Caveat: model card says the models have not been tested and asks for reference
  audio contributions.
- License/risk: Hugging Face shows MIT, but character/audio rights remain a
  practical risk. Treat as local testing only.
- Neiroha fit: useful for testing bulk import later, not as turnkey as
  `UnlimitedBurst/GPT-SoVITS`.

## Current Recommendation

Use `UnlimitedBurst/GPT-SoVITS` first because it has the three pieces this
launcher needs per voice profile:

- `gpt_weights_path`
- `sovits_weights_path`
- `ref_audio_path` plus matching `prompt_text`

Then use `AI-Hobbyist/GPT-SoVits-V2-models` if you want to test one shared model
with many voices, after adding reference audio manually.
