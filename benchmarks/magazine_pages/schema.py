"""Pydantic schema for Magazine Pages advertisement detection."""

from typing import List

from pydantic import BaseModel, Field


class Advertisement(BaseModel):
    box: List[float] = Field(
        description=(
            "Bounding box [x0, y0, x1, y1] in original-image pixel "
            "coordinates; (x0, y0) is top-left, (x1, y1) is bottom-right."
        )
    )


class MagazinePage(BaseModel):
    advertisements: List[Advertisement] = Field(
        default_factory=list,
        description="All advertisements on the page. Empty list if none.",
    )
