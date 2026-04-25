"""DSPy Signature for magazine-page advertisement detection."""

import dspy

DOCUMENT_DESC = """\
A JSON object listing every advertisement on the magazine page, with each
advertisement expressed as a pixel-coordinate bounding box on the ORIGINAL
image (not a resized / cropped view).

Schema:
{
  "advertisements": [
    {"box": [x0, y0, x1, y1]}
  ]
}

CRITICAL — coordinate space:
- ``x0, y0`` is the top-left corner of the advertisement bounding box;
  ``x1, y1`` is the bottom-right.
- ALL four values are integer pixel coordinates on the ORIGINAL image,
  scaled to the dimensions given in the ``page_size`` input.
- DO NOT use normalized [0, 1] coordinates.
- DO NOT use the model's default 0-1000 grounding grid.
- Coordinates MUST satisfy: 0 ≤ x0 < x1 ≤ width, 0 ≤ y0 < y1 ≤ height,
  where ``width`` and ``height`` come from ``page_size``. For a typical
  magazine page (~2480x3500), valid x values run from 0 to ~2480 and
  y values from 0 to ~3500.

Example for a page_size of "2479x3508" with one advertisement
occupying the lower-left quadrant:
  {"advertisements": [{"box": [175, 1723, 1169, 3192]}]}

Other rules:
- If the page contains no advertisements, return ``{"advertisements": []}``.
- Do NOT include editorial articles, page numbers, mastheads, or decorative
  elements — only genuine advertisements.
- Return ONLY the JSON object, no commentary or code fences.
"""


class MagazinePageAdDetection(dspy.Signature):
    """Locate every advertisement on a magazine page and emit its pixel bounding box."""

    page_image: dspy.Image = dspy.InputField(desc="Magazine page scan")
    page_size: str = dspy.InputField(
        desc=(
            "Original page dimensions as 'WIDTHxHEIGHT' in pixels (e.g. "
            "'2480x3508'). Use these dimensions as the coordinate space "
            "for every bounding box."
        )
    )
    document: str = dspy.OutputField(desc=DOCUMENT_DESC)
