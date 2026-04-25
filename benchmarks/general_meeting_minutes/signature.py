"""DSPy Signature for General Meeting Minutes extraction."""

import dspy

DOCUMENT_DESC = """\
A JSON object representing one page of meeting minutes of Mines de Costano S.A. (1930s-1960s).
Schema:
{
  "document": "string — document filename (passed as input)",
  "page_number": integer — page number (passed as input),
  "entries": [
    {
      "number": "string — row number in the attendee table",
      "name": "string — attendee's name",
      "address": "string — attendee's address (separate from name)",
      "actions_o": "string — ordinary share count",
      "actions_p": "string — preferred share count",
      "no_de_voix": "string — number of votes",
      "signature_present": true or false,
      "signature": "string — transcribed signature text, or empty if absent"
    }
  ],
  "total_actions": {
    "total_o": "string — total ordinary shares",
    "total_p": "string — total preferred shares",
    "total_voix": "string — total votes"
  }
}

About the source:
- Table-like shareholder meeting minutes, typed and handwritten, 1930s–1960s.
- Languages mix French, German, and Italian within a single document.
- Name and Address share a single cell in the source table — split them into
  the separate "name" and "address" fields, preserving line breaks between
  multi-line addresses. Drop visual splitter characters (dashes) between
  name and address, but retain dashes that are part of the address itself.
- Optional fields (actions_p, signature) are often empty — use empty strings,
  not null.
- Numeric fields (actions_o, actions_p, no_de_voix, totals) are stored as
  strings for tolerant fuzzy comparison.
- Return ONLY the JSON object, no commentary or code fences.
"""


class MinutesExtraction(dspy.Signature):
    """Extract shareholder-meeting attendance + vote table from a scanned minutes page."""

    page_image: dspy.Image = dspy.InputField(
        desc="Scanned page of meeting minutes for Mines de Costano S.A."
    )
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
