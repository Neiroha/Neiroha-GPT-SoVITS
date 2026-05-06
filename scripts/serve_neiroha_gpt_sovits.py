from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = WORKSPACE_ROOT / "scripts" / "launch_gpt_sovits.py"
DEFAULT_REPO_DIR = WORKSPACE_ROOT / "GPT-SoVITS"
DEFAULT_CONFIG_PATH = DEFAULT_REPO_DIR / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
DEFAULT_PROFILE_PATH = WORKSPACE_ROOT / "profiles" / "voices.json"
DEFAULT_API_PORT = int(os.environ.get("NEIROHA_GPT_SOVITS_API_PORT", "9880"))
DEFAULT_ADMIN_PORT = int(os.environ.get("NEIROHA_GPT_SOVITS_ADMIN_PORT", "7860"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Neiroha GPT-SoVITS API and Gradio admin UI together.",
    )
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--admin-host", default="0.0.0.0")
    parser.add_argument("--admin-port", type=int, default=DEFAULT_ADMIN_PORT)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--device", default="config", help="config, auto, cpu, cuda, or cuda:N")
    parser.add_argument("--half", action="store_true", help="Force half precision.")
    parser.add_argument("--no-half", action="store_true", help="Force full precision.")
    parser.add_argument("--preload-model", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    return parser.parse_args()


def local_port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def local_port_can_bind(host: str, port: int) -> tuple[bool, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            return False, str(exc)
    return True, ""


def require_usable_port(host: str, port: int, name: str) -> None:
    if local_port_is_open(port):
        raise SystemExit(
            f"{name} port {port} is already in use. Stop that process or choose another port."
        )
    can_bind, reason = local_port_can_bind(host, port)
    if not can_bind:
        raise SystemExit(
            f"{name} port {port} is not bindable on {host}: {reason}. "
            "On Windows this often means the port is in an excluded range; "
            "run `netsh interface ipv4 show excludedportrange protocol=tcp` or choose another port."
        )


def wait_for_health(api_base: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{api_base.rstrip('/')}/health"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise SystemExit(f"API process exited during startup with code {return_code}.")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise SystemExit(f"API did not become healthy before timeout: {health_url}")


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def launcher_base_args(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(LAUNCHER),
        "--repo-dir",
        str(args.repo_dir),
        "--config",
        str(args.config),
        "--profiles",
        str(args.profiles),
        "--device",
        args.device,
        "--log-level",
        args.log_level,
    ]
    if args.half:
        command.append("--half")
    if args.no_half:
        command.append("--no-half")
    return command


def main() -> None:
    args = parse_args()
    api_base = args.api_base or f"http://127.0.0.1:{args.api_port}"

    require_usable_port(args.api_host, args.api_port, "API")
    require_usable_port(args.admin_host, args.admin_port, "Admin")

    api_command = launcher_base_args(args) + [
        "--mode",
        "api",
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
    ]
    if args.preload_model:
        api_command.append("--preload-model")

    admin_command = launcher_base_args(args) + [
        "--mode",
        "admin",
        "--host",
        args.admin_host,
        "--port",
        str(args.admin_port),
        "--api-base",
        api_base,
    ]

    processes: list[subprocess.Popen] = []
    print(f"Starting FastAPI on {api_base}")
    api_process = subprocess.Popen(api_command, cwd=WORKSPACE_ROOT)
    processes.append(api_process)
    try:
        wait_for_health(api_base, api_process, args.startup_timeout)
        print(f"Starting Gradio admin on http://127.0.0.1:{args.admin_port}")
        admin_process = subprocess.Popen(admin_command, cwd=WORKSPACE_ROOT)
        processes.append(admin_process)

        print("Neiroha GPT-SoVITS stack is running. Press Ctrl+C to stop both processes.")
        while True:
            for name, process in (("API", api_process), ("Admin", admin_process)):
                return_code = process.poll()
                if return_code is not None:
                    raise SystemExit(f"{name} process exited with code {return_code}.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping Neiroha GPT-SoVITS stack...")
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    main()
