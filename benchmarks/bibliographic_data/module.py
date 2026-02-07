"""DSPy Module wrapping the bibliographic data extraction signature."""

import dspy

from benchmarks.bibliographic_data.signature import BibliographicDataExtraction


class BibDataExtractor(dspy.Module):
    VALID_MODULE_TYPES = ("predict", "cot")

    def __init__(self, module_type: str = "predict"):
        super().__init__()
        if module_type not in self.VALID_MODULE_TYPES:
            raise ValueError(f"module_type must be one of {self.VALID_MODULE_TYPES}, got {module_type!r}")
        if module_type == "cot":
            self.predict = dspy.ChainOfThought(BibliographicDataExtraction)
        else:
            self.predict = dspy.Predict(BibliographicDataExtraction)

    def forward(self, page_image):
        return self.predict(page_image=page_image)


# Canonical alias for script compatibility
Extractor = BibDataExtractor
