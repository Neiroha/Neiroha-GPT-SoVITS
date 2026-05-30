from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AdminContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = (ROOT / "scripts" / "launch_gpt_sovits.py").read_text(encoding="utf-8")

    def test_admin_contract_tabs_are_present(self) -> None:
        for label in (
            "Status",
            "Model Presets",
            "Voice Sets",
            "Synthesis Test",
            "Downloads",
            "Runtime Logs",
            "Settings",
        ):
            with self.subTest(label=label):
                self.assertIn(label, self.launcher)

    def test_app_admin_entrypoint_exists(self) -> None:
        self.assertTrue((ROOT / "app" / "admin" / "gradio_app.py").is_file())


if __name__ == "__main__":
    unittest.main()

