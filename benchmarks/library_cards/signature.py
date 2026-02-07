"""DSPy Signature for library card extraction."""

import dspy

DOCUMENT_DESC = """\
A JSON object with the extracted bibliographic data from the library card image.
The JSON must follow this exact schema:
{
  "type": {"type": "Dissertation or thesis" OR "Reference"},
  "author": {"last_name": "string", "first_name": "string"},
  "publication": {
    "title": "string",
    "year": "string",
    "place": "string or null",
    "pages": "string or null",
    "publisher": "string or null",
    "format": "string or null",
    "editor": "string or null"
  },
  "library_reference": {
    "shelfmark": "string or null",
    "subjects": "string or null"
  }
}

Rules:
- If the card contains "s." on a separate line, type is "Reference"; otherwise "Dissertation or thesis".
- Use null for missing optional fields.
- Remove " S." suffix from pages values.
- Format is usually "8°", "8'" or "4°".
- Shelfmark often begins with "Diss." or "AT".
"""


class LibraryCardExtraction(dspy.Signature):
    """Extract structured bibliographic data from a scanned Swiss library catalog card image."""

    card_image: dspy.Image = dspy.InputField(desc="Scanned image of a library catalog card")
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
