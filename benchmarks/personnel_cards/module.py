"""DSPy Module wrapping the personnel card extraction signature."""

from benchmarks.personnel_cards.signature import PersonnelCardExtraction
from benchmarks.shared.modules import build_extractor_class

Extractor = build_extractor_class(PersonnelCardExtraction, class_name="PersonnelCardExtractor")
