"""Self-contained sample data for the contract-review demo.

Run ``uv run make-sample`` (or ``python -m contract_review_agent.data.make_sample``)
to generate everything under ``data/generated/``: the contract markdown, the
"scanned" signature-page image, and the SQLite obligations database. Nothing is
downloaded — it is all synthetic and safe to ship.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
GENERATED_DIR = DATA_DIR / "generated"
CUAD_DIR = GENERATED_DIR / "cuad"  # real contracts fetched from CUAD
MANIFEST_PATH = GENERATED_DIR / "manifest.json"
DB_PATH = GENERATED_DIR / "obligations.db"
