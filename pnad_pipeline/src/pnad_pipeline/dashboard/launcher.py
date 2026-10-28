"""Inicia o servidor local em segundo plano e reutiliza a instância existente."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from .runtime import ROOT, WORKSPACE_ID

RUNTIME_DIR = ROOT / "pnad_pipeline/data/processed/dashboard_runtime"


def health(port):
    url = f"http://127.0.0.1:{port}/api/saude"
    # A consulta ao loopback não deve passar por proxies configurados no sistema.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=1) as response:
            return json.loads(response.read(4096))
    except (OSError, urllib.error.URLError, ValueError):
        return None


def belongs_here(status):
    return isinstance(status, dict) and status.get("status") == "ok" and status.get("workspace") == WORKSPACE_ID


def start(port=8000):
    status = health(port)
    if belongs_here(status):
        return status
    with socket.socket() as connection:
        connection.settimeout(.5)
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"A porta {port} está ocupada por outro serviço. Use --port 8001 ou encerre o outro serviço.")
    missing = [name for name in ("fastapi", "uvicorn", "pandas", "numpy", "pyarrow", "sklearn", "joblib", "scipy", "statsmodels") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Dependências ausentes: " + ", ".join(missing) + ". Execute: python -m pip install -r pnad_pipeline/requirements.txt")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNTIME_DIR / f"servidor_{port}.log"
    options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "pnad_pipeline/scripts/13_servir_dashboard.py"), "--port", str(port)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options,
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = health(port)
        if belongs_here(status):
            (RUNTIME_DIR / f"estado_{port}.json").write_text(json.dumps({**status, "porta":port, "log":str(log_path)}, indent=2), encoding="utf-8")
            return status
        if process.poll() is not None:
            raise RuntimeError(f"O servidor não iniciou. Consulte {log_path} e verifique as dependências.")
        time.sleep(.2)
    # Apenas o processo que este comando acabou de criar é encerrado no timeout.
    process.terminate()
    raise RuntimeError(f"O servidor não respondeu em 30 segundos. Consulte {log_path}.")


def stop(port=8000):
    status = health(port)
    if status is None:
        return False
    if not belongs_here(status) or not isinstance(status.get("pid"), int) or status["pid"] <= 1:
        raise RuntimeError("O serviço nesta porta não foi identificado como o dashboard deste projeto. Nenhum processo foi encerrado.")
    os.kill(status["pid"], signal.SIGTERM)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Inicia sem abrir o navegador.")
    parser.add_argument("--stop", action="store_true", help="Encerra somente o dashboard deste projeto.")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Escolha uma porta entre 1024 e 65535.")
    try:
        if args.stop:
            print("Dashboard encerrado." if stop(args.port) else "O dashboard já está desligado.")
        else:
            status = start(args.port)
            url = f"http://127.0.0.1:{args.port}"
            print(f"Dashboard disponível: {url} (PID {status['pid']})", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"Não foi possível abrir o dashboard: {exc}", file=sys.stderr)
        return 1
