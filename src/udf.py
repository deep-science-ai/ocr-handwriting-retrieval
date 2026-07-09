"""UDF for handwriting OCR: the Enterprise feature-engineering path.

The OCR model logic here is identical to the OSS stage in ``src/02_ocr.py`` (the
same DSPy program from ``dspy_programs.make_handwriting_reader``). The difference
is packaging: here it's a Geneva UDF, so it can be registered as a column and
backfilled by Geneva instead of driven by a hand-written read/merge loop.

Defining the UDF in its own importable module (rather than inline) is what lets
Geneva distribute it to its workers.
"""

from __future__ import annotations

import io

import pyarrow as pa
from geneva import udf
from PIL import Image

from dspy_programs import make_handwriting_reader


@udf(data_type=pa.string(), input_columns=["image"], num_cpus=0.25)
class HandwritingOCR:
    """Transcribe a handwritten medicine name from raw PNG bytes.

    Geneva instantiates the UDF once per worker and calls ``setup()`` before the
    first row, so the DSPy program (and its model configuration) loads once per
    worker rather than once per image. The ``image`` parameter maps to the
    ``image`` column Geneva feeds in.
    """

    def __init__(self) -> None:
        self.reader = None

    def setup(self) -> None:
        # Same program the OSS pipeline uses; configures DSPy on this worker.
        _, self.reader = make_handwriting_reader()

    def __call__(self, image: bytes) -> str:
        if self.reader is None:
            self.setup()
        import dspy

        pil_image = Image.open(io.BytesIO(image))
        pil_image.load()
        pred = self.reader(image=dspy.Image(pil_image))
        return pred.ocr_text.strip()
