#!/usr/bin/env python3
"""Serve o dashboard e a API no mesmo endereço, sem expor os microdados."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8000)
    a = p.parse_args()
    uvicorn.run("pnad_pipeline.dashboard.api:app",host=a.host,port=a.port)
