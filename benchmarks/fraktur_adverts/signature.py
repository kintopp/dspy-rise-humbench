"""DSPy Signature for Fraktur advert extraction."""

import dspy

DOCUMENT_DESC = """\
A JSON object listing every advertisement on the page image.
Schema:
{
  "advertisements": [
    {
      "date": "YYYY-MM-DD",
      "tags_section": "string — the section heading under which the ad appears",
      "text": "string — the ad text as printed, including the leading ordinal number (e.g. '16. ...')"
    }
  ]
}

Source: early modern (18th-century) German newspapers typeset in Fraktur.
Pages are usually in two columns; ignore the masthead.

Extraction rules:
- Transcribe text EXACTLY as printed — preserve non-modern spellings, typos,
  historical characters, and the leading ordinal (e.g. "1.", "16.").
- Each ad's ``text`` must begin with its printed ordinal number (the scoring
  routine keys matches on this prefix).
- ``tags_section`` is the heading that introduces a block of ads (e.g.
  "Es werden zum Verkauff offerirt"). All ads under one heading share it.
- ``date`` is the newspaper issue date in ISO format; it is the same for
  every advertisement on a given page.
- Return ONLY the JSON object, no commentary or code fences.
"""


class FrakturAdvertExtraction(dspy.Signature):
    """Extract every advertisement from a scanned 18th-century Fraktur newspaper page."""

    page_image: dspy.Image = dspy.InputField(
        desc="Scanned newspaper page in 18th-century German Fraktur typeset"
    )
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
