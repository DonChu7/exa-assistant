# lrg_test_tool/metrics/schemas_runtime.py

from pydantic import BaseModel
from typing import Optional, List

class LrgRuntimeSample(BaseModel):
    lrg_id: str
    ts: str                # 'YYYY-MM-DD'
    runtime_sec: float
    suite: Optional[str] = None

class LrgRuntimeQueryResult(BaseModel):
    lrg_id: str
    samples: List[LrgRuntimeSample]