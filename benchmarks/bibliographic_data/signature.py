"""DSPy Signature for bibliographic data extraction."""

import dspy

DOCUMENT_DESC = """\
A JSON object with the extracted bibliographic data from the scanned bibliography page.
The JSON must follow this exact schema:
{
  "metadata": {
    "title": "string — title of the bibliography",
    "year": "string — year of the bibliography section",
    "page_number": integer or null
  },
  "entries": [
    {
      "id": "string — sequential entry number from the bibliography",
      "type": "book" | "journal-article" | "other",
      "title": "string — title of the work",
      "container_title": "string or null — journal/collection name for articles",
      "author": [{"family": "string", "given": "string"}] or null,
      "note": "string or null — additional notes (reprints, volume info, etc.)",
      "publisher": "string or null",
      "editor": ["string"] or null,
      "publisher_place": "string or null — city of publication",
      "issued": "string or null — year of publication",
      "event_date": "string or null",
      "related": ["string — IDs of related entries"] or null,
      "relation": "string or null — e.g. 'reviewed' for reviews",
      "volume": "string or null — volume number (use Roman numerals as written)",
      "page": "string or null — page range",
      "fascicle": "string or null",
      "reprint": "string or null",
      "edition": "string or null",
      "incomplete": true or null — set to true if the entry is cut off at page edge
    }
  ]
}

Rules:
- Assign sequential IDs matching the numbering in the bibliography.
- Use "journal-article" for journal articles (has container_title/volume/page), "book" for books.
- If an entry is a review of another, set relation to describe it and related to the reviewed entry's ID.
- Set incomplete to true only if the entry is cut off at the page boundary.
- Use null for missing optional fields, not empty strings.
- Preserve original language and diacritics in titles.
"""


class BibliographicDataExtraction(dspy.Signature):
    """Extract structured bibliographic data from a scanned page of a historical bibliography."""

    page_image: dspy.Image = dspy.InputField(desc="Scanned page of a historical bibliography")
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
