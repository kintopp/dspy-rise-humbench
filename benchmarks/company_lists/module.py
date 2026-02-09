"""DSPy Module wrapping the company list extraction signature."""

import dspy

from benchmarks.company_lists.signature import CompanyListExtraction


class CompanyListExtractor(dspy.Module):
    VALID_MODULE_TYPES = ("predict", "cot")

    def __init__(self, module_type: str = "predict"):
        super().__init__()
        if module_type not in self.VALID_MODULE_TYPES:
            raise ValueError(f"module_type must be one of {self.VALID_MODULE_TYPES}, got {module_type!r}")
        if module_type == "cot":
            self.predict = dspy.ChainOfThought(CompanyListExtraction)
        else:
            self.predict = dspy.Predict(CompanyListExtraction)

    def forward(self, page_image, page_id):
        return self.predict(page_image=page_image, page_id=page_id)


# Canonical alias for script compatibility
Extractor = CompanyListExtractor
