"""Pydantic schema for personnel card documents — matches benchmark's dataclass.py."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class FieldValue(BaseModel):
    diplomatic_transcript: str
    interpretation: Optional[str] = None
    is_crossed_out: bool = False


class TableRow(BaseModel):
    row_number: int
    dienstliche_stellung: FieldValue
    dienstort: FieldValue
    gehaltsklasse: FieldValue
    jahresgehalt_monatsgehalt_taglohn: FieldValue
    datum_gehaltsänderung: FieldValue
    bemerkungen: FieldValue


class Table(BaseModel):
    rows: list[TableRow]
