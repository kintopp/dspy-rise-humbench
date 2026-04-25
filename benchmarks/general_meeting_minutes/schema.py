"""Pydantic schema for General Meeting Minutes (mirrors upstream dataclass.py)."""

from typing import List

from pydantic import BaseModel, Field


class Entry(BaseModel):
    number: str
    name: str
    address: str
    actions_o: str
    actions_p: str
    no_de_voix: str
    signature_present: bool = False
    signature: str


class TotalActions(BaseModel):
    total_o: str
    total_p: str
    total_voix: str


class MinutesPage(BaseModel):
    """Main output — one page of the General Meeting Minutes of Mines de Costano S.A."""

    document: str = Field(description="Document name (passed via prompt)")
    page_number: int = Field(description="Page number (passed via prompt)")
    entries: List[Entry]
    total_actions: TotalActions
