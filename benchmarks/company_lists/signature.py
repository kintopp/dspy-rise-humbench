"""DSPy Signature for company list extraction."""

import dspy

DOCUMENT_DESC = """\
A JSON object with structured data extracted from a scanned page of a company trade index.
The JSON must follow this exact schema:
{
  "page_id": "string — the page identifier, provided as input",
  "entries": [
    {
      "entry_id": "string — unique identifier in the format '{page_id}-N' where N is the sequential entry number starting at 1",
      "company_name": "string — the name of the company or person",
      "location": "string — the location of the company, e.g. 'Zurich' or 'London, UK'. Use the string 'null' (not JSON null) if no location is given"
    }
  ]
}

About the source:
- The image stems from a trade index of the British Swiss Chamber of Commerce (1925-1958).
- Pages can show an alphabetical or a thematic list of companies.
- Companies are mostly located in Switzerland and the UK.
- Most pages have one column but some years have two columns.
- The source is in English and German but company names can be in English, German, French or Italian.

About the entries:
- Each entry describes a single company or person.
- Alphabetical entries have filling dots between the company name and a page number. Dots and page numbers are NOT part of the data — ignore them.
- Alphabetical entries seldom to never have locations.
- Thematic entries often have locations.
- Thematic entries are listed under headings that describe the type of business. Headings are NOT entries — do not extract them.
- Some thematic headings are only references to other headings, e.g. "X, s. Y". These are NOT entries.
- Do not add country information unless it is directly written with the location.
- Preserve original language, diacritics, and punctuation exactly as written.
"""


class CompanyListExtraction(dspy.Signature):
    """Extract structured company data from a scanned page of a historical trade index."""

    page_image: dspy.Image = dspy.InputField(desc="Scanned page from a company trade index")
    page_id: str = dspy.InputField(desc="Page identifier used for generating entry IDs")
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
