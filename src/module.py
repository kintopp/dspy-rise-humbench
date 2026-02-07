"""DSPy Module wrapping the library card extraction signature."""

import dspy

from src.signature import LibraryCardExtraction


class LibraryCardExtractor(dspy.Module):
    def __init__(self, module_type: str = "predict"):
        super().__init__()
        if module_type == "cot":
            self.predict = dspy.ChainOfThought(LibraryCardExtraction)
        else:
            self.predict = dspy.Predict(LibraryCardExtraction)

    def forward(self, card_image):
        return self.predict(card_image=card_image)
