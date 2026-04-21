"""DSPy Module wrapping the bibliographic data extraction signature."""

from benchmarks.bibliographic_data.signature import BibliographicDataExtraction
from benchmarks.shared.modules import build_extractor_class

Extractor = build_extractor_class(BibliographicDataExtraction, class_name="BibDataExtractor")
