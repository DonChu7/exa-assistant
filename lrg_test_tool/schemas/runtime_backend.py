from pathlib import Path
import sqlite3
from datetime import date, timedelta
from typing import Optional, List
from .runtime_schemas import LrgRuntimeSample, LrgRuntimeQueryResult

METRICS_DB = Path(__file__).resolve().parents[1] / "lrg_metrics.db"

def get_lrg_runtimes(
    lrg_id: Optional[str],
    days: int = 30,
    as_of: Optional[date] = None,
) -> List[LrgRuntimeQueryResult]:
    ...