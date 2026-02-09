"""Pydantic schema for company list pages — matches benchmark's dataclass.py."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Entry(BaseModel):
    entry_id: str
    company_name: str
    location: str


class ListPage(BaseModel):
    page_id: str
    entries: List[Entry]
