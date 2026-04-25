"""DSPy Signature for medieval manuscript transcription."""

import dspy

DOCUMENT_DESC = """\
A JSON object transcribing every folio visible on the manuscript page image.
Schema:
{
  "folios": [
    {
      "folio": "string — the folio number as written on the page (empty string if not visible)",
      "text": "string — full transcription of the main text body, \\n between visual lines",
      "addition1": "string or empty — first marginal annotation",
      "addition2": "string or empty — second marginal annotation (if any)",
      "addition3": "string or empty — third marginal annotation (if any)"
    }
  ]
}

About the source:
- 15th-century Basel manuscript, late medieval German.
- Each page typically contains one folio (or occasionally two if a recto/verso
  pair is visible).

Transcription rules:
- Transcribe text EXACTLY as it appears — preserve historical spellings,
  punctuation, and line breaks (use "\\n" to mark visual line breaks).
- DO NOT resolve abbreviations or normalize to modern orthography.
- Superscribed letters: write the superscribed letter after the base letter
  (e.g. "u with superscribed o" → "uo").
- Preserve Medieval Unicode Font Initiative glyphs verbatim where they appear
  (ꝛ = "er" abbreviation, ꝰ = "us" / "em" abbreviation, etc. — do NOT resolve).
- Empty fields: use the empty string "" (not null).
- Folio order in the output must match left-to-right / top-to-bottom order on
  the page image — scoring matches folio entries positionally.
- Return ONLY the JSON object, no commentary or code fences.
"""


class ManuscriptTranscription(dspy.Signature):
    """Transcribe every folio visible on a 15th-century Basel manuscript page image."""

    page_image: dspy.Image = dspy.InputField(
        desc="Scanned image of a medieval manuscript page (15th century, late medieval German)"
    )
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
