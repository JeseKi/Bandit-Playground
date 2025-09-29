from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from bandit_lib.agents.schemas import Metrics

class ProcessDataDump(BaseModel):
    run_id: str
    create_at: datetime = Field(default_factory=datetime.now)
    total_steps: int = Field(..., description="Total steps of the agent.")
    metrics: List[Metrics]