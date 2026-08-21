from __future__ import annotations

from pydantic import BaseModel, Field


class CalibrationPointsRequest(BaseModel):
    video: str
    points: list[list[float]] = Field(min_length=4, max_length=4)
    world_width_m: float | None = None
    world_height_m: float | None = None
    frame_index: int = 0
    # None -> server defaults to f"{video_stem}.json" (the CLI tool's own
    # convention). The FAB modal never relies on that default -- it always
    # sends an explicit f"{video_stem}_{timestamp}.json" so a fresh
    # calibration never collides with an existing file. See
    # server/routes/calibration.py's overwrite check.
    output_filename: str | None = None
    overwrite: bool = False
