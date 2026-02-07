"""DSPy Signature for business letter extraction (multi-image input)."""

import dspy

DOCUMENT_DESC = """\
A JSON object with the extracted metadata from the letter images.
The JSON must follow this exact schema:
{
  "letter_title": ["string or null"],
  "send_date": ["YYYY-MM-DD"],
  "sender_persons": ["First Last"],
  "receiver_persons": ["First Last"],
  "has_signatures": "TRUE" or "FALSE"
}

Rules:
- All values for letter_title, send_date, sender_persons, and receiver_persons
  must be lists, even if there is only a single value.
- Use ISO format YYYY-MM-DD for dates.
- Write person names as they appear in the letter, in "First Last" order
  (e.g. "Werner Stauffacher", "Dr. Max Vischer", "Fritz Ritter").
  Include titles and honorifics only when they appear as part of the name.
  Do NOT use "Last, First" format.
- If a piece of information is not found in the letter, set the value to ["null"].
- has_signatures: "TRUE" if the letter contains handwritten signatures, "FALSE" otherwise.
- Do not return anything except the JSON object.
"""


class BusinessLetterExtraction(dspy.Signature):
    """Extract structured metadata from a series of scanned historical letter page images."""

    page_images: list[dspy.Image] = dspy.InputField(
        desc="Scanned page images of a historical letter (one or more pages)"
    )
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
