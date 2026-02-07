"""DSPy Signature for personnel card extraction."""

import dspy

DOCUMENT_DESC = """\
A JSON object with the extracted table data from the personnel card image.
The JSON must follow this exact schema:
{
  "rows": [
    {
      "row_number": 1,
      "dienstliche_stellung": {
        "diplomatic_transcript": "string",
        "interpretation": "string or null",
        "is_crossed_out": false
      },
      "dienstort": { ... same sub-fields ... },
      "gehaltsklasse": { ... same sub-fields ... },
      "jahresgehalt_monatsgehalt_taglohn": { ... same sub-fields ... },
      "datum_gehaltsänderung": { ... same sub-fields ... },
      "bemerkungen": { ... same sub-fields ... }
    }
  ]
}

Column definitions:
1. dienstliche_stellung — Official position / job title
2. dienstort — Place of service / work location
3. gehaltsklasse — Salary class / grade
4. jahresgehalt_monatsgehalt_taglohn — Annual/monthly salary or daily wage
5. datum_gehaltsänderung — Date of salary change
6. bemerkungen — Remarks / notes

Rules for each field:
- diplomatic_transcript: Transcribe EXACTLY as written, including abbreviations,
  punctuation, spacing, currency symbols (e.g. "Fr. 2'400.-"), and original date
  formats. Use empty string "" for empty cells. Reproduce ditto marks (") exactly.
- interpretation: Expand abbreviations (e.g. "Assist." → "Assistent", "Prof." →
  "Professor"), replace ditto marks with actual values from the previous row,
  convert dates to ISO format YYYY-MM-DD, extract numeric salary values (remove
  currency), convert roman numerals to arabic. Use null if no interpretation needed.
- is_crossed_out: true if the text is struck through or deleted, otherwise false.

Row handling:
- Number rows sequentially starting from 1.
- Include ALL rows with ANY content in ANY column.
- Omit completely empty rows.

Return ONLY the JSON object, no additional text.
"""


class PersonnelCardExtraction(dspy.Signature):
    """Extract structured table data from a scanned Swiss personnel card image."""

    card_image: dspy.Image = dspy.InputField(desc="Scanned image of a personnel card")
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
