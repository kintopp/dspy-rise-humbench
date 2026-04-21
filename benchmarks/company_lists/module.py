"""DSPy Module wrapping the company list extraction signature."""

from benchmarks.company_lists.signature import CompanyListExtraction
from benchmarks.shared.modules import build_extractor_class

Extractor = build_extractor_class(CompanyListExtraction, class_name="CompanyListExtractor")
