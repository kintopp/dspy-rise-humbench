"""Pydantic schema for Fraktur Adverts (mirrors upstream dataclass.py)."""

from typing import Optional, List

from pydantic import BaseModel


class Advertisement(BaseModel):
    """A single Fraktur classified advertisement."""

    date: Optional[str] = None
    tags_section: Optional[str] = None
    text: Optional[str] = None


class Document(BaseModel):
    """Root output — a list of advertisements extracted from one page image."""

    advertisements: List[Advertisement] = []
