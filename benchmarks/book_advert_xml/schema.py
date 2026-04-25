"""Pydantic models for Book Advert XML benchmark output structure.

Reference-only — not used at runtime. Mirrors upstream's
humanities_data_benchmark/benchmarks/book_advert_xml/dataclass.py for
documentation parity.

Note: upstream's GT JSON files use ``number_of_fixes``, but upstream's
``dataclass.py`` (and this mirror) use ``number_of_corrections``. This is a
known upstream inconsistency. We sidestep it by scoring only ``fixed_xml``
and not surfacing the count or explanation fields to the model.
"""

from pydantic import BaseModel, Field


class CorrectedAdvert(BaseModel):
    fixed_xml: str = Field(..., description="The corrected XML content as a string.")
    number_of_corrections: int = Field(..., description="The number of corrections made to the original XML.")
    explanation: str | None = Field(None, description="Optional explanation of the corrections made.")
