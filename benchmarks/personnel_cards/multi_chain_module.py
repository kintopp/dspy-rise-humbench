"""MultiChainComparison module for Personnel Cards.

Generates M diverse extraction attempts with the base MIPROv2-optimized
extractor at temperature=1.0, then passes all attempts to a comparator
that synthesizes the best answer.

Includes FullMultiChainComparison — a subclass that fixes DSPy's
truncation of multi-line outputs in the comparison step.
"""

import dspy
from dspy.predict.multi_chain_comparison import MultiChainComparison

from benchmarks.personnel_cards.signature import PersonnelCardExtraction


class FullMultiChainComparison(MultiChainComparison):
    """MultiChainComparison with multi-line output support.

    The base class truncates each completion's rationale and answer to the
    first line via ``.split("\\n")[0]`` (lines 39-40 of multi_chain_comparison.py).
    For JSON outputs this means the comparator only sees ``{`` — making
    comparison meaningless. This subclass removes the truncation.
    """

    def forward(self, completions, **kwargs):
        attempts = []
        for c in completions:
            rationale = c.get("rationale", c.get("reasoning", "")).strip()
            answer = str(c[self.last_key]).strip()
            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
            )

        assert len(attempts) == self.M, (
            f"The number of attempts ({len(attempts)}) doesn't match "
            f"the expected number M ({self.M})."
        )

        kwargs = {
            **{f"reasoning_attempt_{idx+1}": attempt for idx, attempt in enumerate(attempts)},
            **kwargs,
        }
        return self.predict(**kwargs)


class MultiChainExtractor(dspy.Module):
    """Wraps a base CoT extractor with MultiChainComparison synthesis.

    forward() runs the base module M times at temperature=1.0 for diversity,
    collects the completions, and passes them to FullMultiChainComparison
    which synthesizes a single refined answer.
    """

    def __init__(self, base_module=None, M=3):
        super().__init__()
        self.M = M
        self.base = base_module or dspy.ChainOfThought(PersonnelCardExtraction)
        self.comparator = FullMultiChainComparison(
            PersonnelCardExtraction, M=M,
        )

    def forward(self, card_image):
        # Generate M diverse attempts at high temperature
        completions = []
        with dspy.context(temperature=1.0):
            for _ in range(self.M):
                pred = self.base(card_image=card_image)
                completions.append(pred)

        # Synthesize via comparison (runs at default temperature)
        return self.comparator(completions, card_image=card_image)
