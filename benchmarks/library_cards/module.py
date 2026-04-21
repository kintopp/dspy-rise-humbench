"""DSPy Module wrapping the library card extraction signature."""

from benchmarks.library_cards.signature import LibraryCardExtraction
from benchmarks.shared.modules import build_extractor_class

Extractor = build_extractor_class(LibraryCardExtraction, class_name="LibraryCardExtractor")
