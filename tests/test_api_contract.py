from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    import scripts.launch_gpt_sovits as launcher
except ModuleNotFoundError as exc:  # pragma: no cover - local Python may lack Pixi deps.
    TestClient = None
    launcher = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[1]


class FakeRuntime:
    def status(self) -> dict[str, object]:
        return {
            "provider": "gpt-sovits",
            "loaded": False,
            "device": "config",
            "is_half": False,
        }


class FakeRegistry:
    profile_path = ROOT / "profiles" / "voices.json"
    server_config_path = ROOT / "configs" / "server.toml"
    voice_sets_dir = ROOT / "configs" / "voice-sets"
    model_presets_dir = ROOT / "configs" / "model-presets"
    runtime_voices_dir = ROOT / "runtime" / "voices"

    def active_voice_set_id(self) -> str:
        return "default"

    def active_model_preset_id(self) -> str:
        return "v2proplus-clone"

    def default_voice_id(self) -> str:
        return "genshin-keqing"

    def list_voice_sets(self) -> list[object]:
        return []

    def list_profiles(self, model_id: str = "") -> list[object]:
        return []

    def has_voice_set(self, model_id: str = "") -> bool:
        return model_id in {"", "default"}


@unittest.skipIf(launcher is None, f"Pixi API dependencies unavailable: {IMPORT_ERROR}")
class ApiContractBehaviorTest(unittest.TestCase):
    def build_client(self) -> TestClient:
        app = launcher.create_api_app(
            FakeRuntime(),
            FakeRegistry(),
            default_voice_id="genshin-keqing",
        )
        return TestClient(app)

    def test_capabilities_use_standard_native_prefix(self) -> None:
        client = self.build_client()
        response = client.get("/api/gpt-sovits/capabilities")

        self.assertEqual(response.status_code, 200)
        routes = response.json()["routes"]
        self.assertEqual(routes["clone_upload"], "/api/gpt-sovits/clone/upload")
        self.assertEqual(routes["trained_models"], "/api/gpt-sovits/models")

    def test_legacy_capabilities_route_still_works(self) -> None:
        client = self.build_client()
        response = client.get("/gpt-sovits/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["routes"]["clone_upload"], "/api/gpt-sovits/clone/upload")

    def test_external_admin_ready_accepts_new_and_legacy_clone_upload_routes(self) -> None:
        for clone_upload in ("/api/gpt-sovits/clone/upload", "/gpt-sovits/clone/upload"):
            with self.subTest(clone_upload=clone_upload):
                self.assertTrue(self.external_admin_ready_with_clone_route(clone_upload))

    def test_external_admin_ready_falls_back_to_legacy_capabilities_route(self) -> None:
        self.assertTrue(
            self.external_admin_ready_with_clone_route(
                "/gpt-sovits/clone/upload",
                fail_new_route=True,
            )
        )

    def test_external_admin_ready_rejects_missing_clone_upload_route(self) -> None:
        self.assertFalse(self.external_admin_ready_with_clone_route("/wrong/clone/upload"))

    def test_openai_error_has_stable_contract_shape(self) -> None:
        response = launcher.openai_error("response_format must be wav", status_code=400)
        payload = json.loads(response.body.decode("utf-8"))

        self.assertEqual(payload["error"]["code"], "unsupported_format")
        self.assertEqual(payload["error"]["message"], "response_format must be wav")
        self.assertEqual(payload["error"]["details"], {})

    def external_admin_ready_with_clone_route(self, clone_upload: str, *, fail_new_route: bool = False) -> bool:
        payload = json.dumps({"routes": {"clone_upload": clone_upload}}).encode("utf-8")
        requested_urls: list[str] = []

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                return None

            def read(self) -> bytes:
                return payload

        def fake_urlopen(url: str, timeout: int = 2):
            requested_urls.append(url)
            if fail_new_route and url.endswith("/api/gpt-sovits/capabilities"):
                raise launcher.urllib.error.URLError("new capabilities route unavailable")
            return FakeResponse()

        previous_urlopen = launcher.urllib.request.urlopen
        try:
            launcher.urllib.request.urlopen = fake_urlopen
            manager = launcher.ManagedApiProcess(
                api_host="127.0.0.1",
                api_port=19880,
                repo_dir=ROOT / "GPT-SoVITS",
                config_path=ROOT / "runtime" / "cache" / "tts_infer.yaml",
                profiles_path=ROOT / "profiles" / "voices.json",
                device="config",
                is_half=None,
                default_voice_id="genshin-keqing",
                log_level="info",
                terminal_rtf_log=False,
                debug_runtime_output=False,
            )
            result = manager.external_admin_ready()
        finally:
            launcher.urllib.request.urlopen = previous_urlopen

        expected_urls = ["http://127.0.0.1:19880/api/gpt-sovits/capabilities"]
        if fail_new_route:
            expected_urls.append("http://127.0.0.1:19880/gpt-sovits/capabilities")
        self.assertEqual(requested_urls, expected_urls)
        return result


if __name__ == "__main__":
    unittest.main()
