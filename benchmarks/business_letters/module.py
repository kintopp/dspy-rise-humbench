"""DSPy Module wrapping the business letter extraction signature."""

from benchmarks.business_letters.signature import BusinessLetterExtraction
from benchmarks.shared.modules import build_extractor_class

Extractor = build_extractor_class(BusinessLetterExtraction, class_name="BusinessLetterExtractor")
