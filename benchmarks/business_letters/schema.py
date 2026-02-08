"""Pydantic schema for business letter documents — matches benchmark's dataclass.py."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Document(BaseModel):
    letter_title: Optional[list[str]] = None
    send_date: Optional[list[str]] = None
    sender_persons: Optional[list[str]] = None
    receiver_persons: Optional[list[str]] = None
    has_signatures: Optional[str] = None
