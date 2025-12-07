from datetime import date
from typing import List
from pydantic import BaseModel

class LrgRuntimeSample(BaseModel):
    label: str          # OSS_MAIN_LINUX.X64_250610.1
    date: date          # parsed date from label
    runtime_hours: float

class LrgRuntimeQueryResult(BaseModel):
    lrg_id: str
    samples: List[LrgRuntimeSample]