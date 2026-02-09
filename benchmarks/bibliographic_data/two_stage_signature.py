"""DSPy Signatures for the two-stage bibliographic data extraction pipeline.

Stage 1 (TranscribeEntries): Image → structured text listing of all entries.
Stage 2 (StructureEntries): Text listing → JSON matching the benchmark schema.
"""

import dspy

ENTRIES_TEXT_DESC = """\
A structured text listing of ALL bibliographic entries visible on the page.
For each entry, include:
- Entry number (matching the numbering printed on the page)
- Type (book or journal-article)
- All bibliographic fields: author(s), title, publisher, place, year, journal name, volume, pages, etc.
- Notes about reprints, relations to other entries, or incomplete entries

Format each entry as a numbered block, e.g.:
  Entry 42:
    Type: book
    Author: Müller, Hans
    Title: Die Geschichte...
    Publisher: Verlag X
    Place: Basel
    Year: 1950

Preserve original language, diacritics, and punctuation exactly as written.
Include ALL entries, even incomplete ones cut off at the page edge.
"""

STRUCTURED_JSON_DESC = """\
A JSON object with the extracted bibliographic data.
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


class TranscribeEntries(dspy.Signature):
    """Transcribe all bibliographic entries from a scanned bibliography page into structured text."""

    page_image: dspy.Image = dspy.InputField(desc="Scanned page of a historical bibliography")
    entries_text: str = dspy.OutputField(desc=ENTRIES_TEXT_DESC)


class StructureEntries(dspy.Signature):
    """Convert a structured text listing of bibliographic entries into JSON format."""

    entries_text: str = dspy.InputField(desc="Structured text listing of bibliographic entries")
    document: str = dspy.OutputField(desc=STRUCTURED_JSON_DESC)
