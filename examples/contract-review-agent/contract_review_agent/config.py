"""Load `config.yaml` and resolve the SIE endpoint from env / `.env`.

Precedence for the cluster URL and key: real env var > `.env` file > `config.yaml`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict[str, Any]:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    env = dotenv_values(PROJECT_ROOT / ".env")

    cluster = cfg.setdefault("cluster", {})
    cluster["url"] = (
        os.environ.get("SIE_CLUSTER_URL") or env.get("SIE_CLUSTER_URL") or cluster.get("url") or "http://localhost:8080"
    )
    cluster["api_key"] = os.environ.get("SIE_API_KEY") or env.get("SIE_API_KEY") or cluster.get("api_key") or ""
    return cfg
