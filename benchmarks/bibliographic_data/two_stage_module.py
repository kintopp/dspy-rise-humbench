"""Two-stage extraction module for Bibliographic Data.

Stage 1 (transcribe): Image → structured text listing of entries.
Stage 2 (structure): Text listing → JSON matching the benchmark schema.

Both stages are discoverable Predict/CoT sub-modules, so MIPROv2 can
optimize their instructions and demos jointly via the end-to-end metric.
"""

import dspy

from benchmarks.bibliographic_data.two_stage_signature import (
    TranscribeEntries,
    StructureEntries,
)


class TwoStageExtractor(dspy.Module):
    """Two-stage bibliographic data extractor.

    Decouples OCR/transcription from schema structuring to reduce
    alignment cascade errors. Each stage can be independently
    inspected and debugged.
    """

    VALID_MODULE_TYPES = ("predict", "cot")

    def __init__(self, module_type: str = "cot"):
        super().__init__()
        if module_type not in self.VALID_MODULE_TYPES:
            raise ValueError(f"module_type must be one of {self.VALID_MODULE_TYPES}, got {module_type!r}")

        if module_type == "cot":
            self.transcribe = dspy.ChainOfThought(TranscribeEntries)
            self.structure = dspy.ChainOfThought(StructureEntries)
        else:
            self.transcribe = dspy.Predict(TranscribeEntries)
            self.structure = dspy.Predict(StructureEntries)

    def forward(self, page_image):
        # Stage 1: Image → structured text
        stage1 = self.transcribe(page_image=page_image)
        entries_text = stage1.entries_text

        # Stage 2: Text → JSON
        stage2 = self.structure(entries_text=entries_text)
        return stage2
