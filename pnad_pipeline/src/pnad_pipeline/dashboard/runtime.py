"""Identidade local compartilhada pelo servidor e pelo inicializador."""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ID = hashlib.sha256(os.path.normcase(str(ROOT)).encode("utf-8")).hexdigest()[:16]
