"""KNN dynamic demo selection module for Library Cards.

Two-pass inference: Pass 1 runs the MIPROv2-optimized program to get an
initial prediction. The prediction JSON is embedded and used to retrieve
k nearest training examples by cosine similarity. Pass 2 re-runs the
program with the retrieved examples as demos.

This preserves MIPROv2's optimized instruction while dynamically
selecting demos tailored to each card type.
"""

import json
import logging

import dspy
import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


class KNNDemoExtractor(dspy.Module):
    """Wraps a base extractor with KNN-based dynamic demo selection.

    At init, embeds all training examples' GT JSON. At inference:
    1. Run base module (pass 1) with original demos
    2. Embed the prediction JSON
    3. Find k nearest training examples
    4. Swap demos and re-run (pass 2)
    """

    def __init__(self, base_module, trainset, k=3, embedder=None):
        """
        Args:
            base_module: MIPROv2-optimized extractor (will be modified in-place).
            trainset: List of dspy.Example with 'document' field containing GT JSON.
            k: Number of nearest neighbors to retrieve.
            embedder: dspy.Embedder instance. Defaults to gemini-embedding-001.
        """
        super().__init__()
        self.base = base_module
        self.k = k
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = dspy.Embedder("gemini/gemini-embedding-001")

        # Pre-compute training embeddings from GT JSON
        logger.info(f"Embedding {len(trainset)} training examples for KNN index...")
        gt_texts = [ex.document if isinstance(ex.document, str) else json.dumps(ex.document)
                     for ex in trainset]
        self._train_embeddings = np.array(self.embedder(gt_texts))
        self._trainset = list(trainset)
        logger.info(f"KNN index ready: {len(trainset)} examples, k={k}")

    def _find_nearest(self, query_text: str, k: int) -> list[dspy.Example]:
        """Find k nearest training examples by cosine similarity."""
        query_emb = np.array(self.embedder([query_text])[0])
        similarities = [
            _cosine_similarity(query_emb, train_emb)
            for train_emb in self._train_embeddings
        ]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self._trainset[i] for i in top_k_indices]

    def forward(self, card_image):
        # Pass 1: Run with original demos
        original_demos = list(self.base.predict.demos) if hasattr(self.base.predict, 'demos') else []

        try:
            pass1_pred = self.base(card_image=card_image)
        except Exception:
            return self.base(card_image=card_image)

        # Embed the pass-1 prediction and find nearest training examples
        pred_text = pass1_pred.document if isinstance(pass1_pred.document, str) else json.dumps(pass1_pred.document)

        try:
            nearest = self._find_nearest(pred_text, self.k)
        except Exception as e:
            logger.warning(f"KNN retrieval failed: {e}. Returning pass-1 result.")
            return pass1_pred

        # Pass 2: Swap demos and re-run
        try:
            self.base.predict.demos = nearest
            pass2_pred = self.base(card_image=card_image)
        finally:
            # Always restore original demos
            self.base.predict.demos = original_demos

        return pass2_pred
