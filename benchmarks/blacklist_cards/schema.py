"""Pydantic schema for blacklist card documents — matches benchmark's dataclass.py."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Entry(BaseModel):
    transcription: str


class Company(BaseModel):
    transcription: str


class BID(BaseModel):
    transcription: str


class Location(BaseModel):
    transcription: str


class Card(BaseModel):
    company: Company
    location: Location
    b_id: BID
    date: Optional[str] = None
    information: Optional[list[Entry]] = None
