"""DSPy Signature for Book Advert XML correction."""

import dspy

DOCUMENT_DESC = """\
A JSON object with a single field ``fixed_xml`` containing the corrected XML
string. The input was produced by an OCR-based LLM extraction process and may
contain structural malformations:
- missing closing tags (e.g. an unclosed </BIBL>),
- incorrect element nesting,
- mismatched element names.

Schema:
{"fixed_xml": "<corrected XML as a single string>"}

Rules for the correction:
- Correct only structural malformations.
- Preserve every character of the input verbatim, including punctuation.
- Preserve element attribute values and whitespace patterns inside element bodies.
- Preserve unicode artefacts such as `√º`, `√†`, `√ü`, `√§`, `√∞`, `√ä`. These
  are intentional OCR-survival markers retained by the ground-truth annotators
  to preserve the original scan's representation; they must NOT be cleaned up,
  normalised, or replaced with their unicode equivalents.

Output the JSON object only — no explanation or commentary outside it.
"""


class CorrectMalformedXml(dspy.Signature):
    """Correct structural malformations in an XML string from a historical book advertisement."""

    xml_string: str = dspy.InputField(
        desc="Malformed XML string from a historical book advertisement.",
    )
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
