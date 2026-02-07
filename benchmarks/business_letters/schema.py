"""Pydantic schema for business letter documents — matches benchmark's dataclass.py."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Document(BaseModel):
    letter_title: Optional[List[str]] = None
    send_date: Optional[List[str]] = None
    sender_persons: Optional[List[str]] = None
    receiver_persons: Optional[List[str]] = None
    has_signatures: Optional[str] = None
