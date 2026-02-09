"""DSPy Signature for blacklist card extraction."""

import dspy

DOCUMENT_DESC = """\
A JSON object with the extracted data from a 1940s British blacklist index card.
The JSON must follow this exact schema:
{
  "company": {"transcription": "string — primary company or person name, top-left of card"},
  "location": {"transcription": "string — city or town, often following the company name"},
  "b_id": {"transcription": "string — identifier code, top-right, starting with 'B.'"},
  "date": "string — stamped date in YYYY-MM-DD format, or empty string if absent",
  "information": [
    {"transcription": "string — text block from the card body"}
  ]
}

Rules:
- company, location, and b_id each have a nested "transcription" field.
- date is a flat string (no transcription wrapper). Use "" if no date is stamped.
- information is a list of objects, each with a "transcription" field. Use "" (empty string) if the card body has no text blocks.
- Preserve line breaks in information blocks with \\n.
- Preserve original language, diacritics, and punctuation exactly as written.
- If you cannot read a value, use an empty string for its transcription.
"""


class BlacklistCardExtraction(dspy.Signature):
    """Extract structured data from a scanned 1940s British blacklist index card."""

    card_image: dspy.Image = dspy.InputField(desc="Scanned blacklist index card")
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
