"""DSPy Module wrapping the blacklist card extraction signature."""

from benchmarks.blacklist_cards.signature import BlacklistCardExtraction
from benchmarks.shared.modules import build_extractor_class

Extractor = build_extractor_class(BlacklistCardExtraction, class_name="BlacklistCardExtractor")
