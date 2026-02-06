"""DSPy Module wrapping the library card extraction signature."""

import dspy

from src.signature import LibraryCardExtraction


class LibraryCardExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(LibraryCardExtraction)

    def forward(self, card_image):
        return self.predict(card_image=card_image)
