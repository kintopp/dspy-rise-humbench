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

Rules:
- ``x0, y0`` is the top-left corner of the advertisement bounding box;
  ``x1, y1`` is the bottom-right. All four values are floats in original-
  image pixel coordinates.
- The page-image width and height (in pixels) are provided via the
  ``page_size`` input — box coordinates must lie within ``[0, width]`` and
  ``[0, height]`` respectively.
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
