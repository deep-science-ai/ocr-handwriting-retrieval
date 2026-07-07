from __future__ import annotations

import os
from dataclasses import dataclass

from config import OPENAI_EXTRACTION_MODEL, OPENAI_VISION_MODEL, load_local_env


def configure_dspy(model_name: str):
    load_local_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for live DSPy/OpenAI model calls.")

    import dspy

    lm = dspy.LM(f"openai/{model_name}")
    dspy.configure(lm=lm)
    return dspy


def make_handwriting_reader():
    dspy = configure_dspy(OPENAI_VISION_MODEL)

    class HandwritingOCR(dspy.Signature):
        """Transcribe the handwritten medicine name shown in the image, exactly as written."""

        image: dspy.Image = dspy.InputField(desc="A handwritten medicine name")
        ocr_text: str = dspy.OutputField(desc="The transcribed text, exactly as written")

    class HandwritingReader(dspy.Module):
        def __init__(self):
            super().__init__()
            self.read = dspy.Predict(HandwritingOCR)

        def forward(self, image: dspy.Image):
            return self.read(image=image)

        async def aforward(self, image: dspy.Image):
            return await self.read.acall(image=image)

    return dspy, HandwritingReader()


def make_medical_term_extractor():
    dspy = configure_dspy(OPENAI_EXTRACTION_MODEL)

    class ExtractMedicalTerm(dspy.Signature):
        """Extract structured metadata from OCR text produced from handwritten medical text."""

        ocr_text: str = dspy.InputField()
        normalized_text: str = dspy.OutputField(desc="Clean normalized version of the OCR text")
        category: str = dspy.OutputField(
            desc="One of: medication, symptom, instruction, measurement, body_part, unknown"
        )
        is_medical: bool = dspy.OutputField(desc="Whether the text appears medically relevant")
        needs_human_review: bool = dspy.OutputField(
            desc="True when the OCR text is ambiguous, low confidence, or medically important"
        )
        searchable_summary: str = dspy.OutputField(desc="Short phrase suitable for embedding and retrieval")

    class MedicalTermExtractor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract = dspy.Predict(ExtractMedicalTerm)

        def forward(self, ocr_text: str):
            return self.extract(ocr_text=ocr_text)

    return MedicalTermExtractor()


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class HeuristicExtraction:
    normalized_text: str
    category: str
    is_medical: bool
    needs_human_review: bool
    searchable_summary: str


def heuristic_extract(ocr_text: str, generic_name: str | None, existing_review_flag: bool | None) -> HeuristicExtraction:
    normalized = " ".join((ocr_text or "").strip().split())
    generic = (generic_name or "").strip()
    summary = f"Medication mention: {normalized}"
    if generic:
        summary += f" ({generic})"
    return HeuristicExtraction(
        normalized_text=normalized,
        category="medication" if normalized else "unknown",
        is_medical=bool(normalized),
        needs_human_review=bool(existing_review_flag) or not bool(normalized),
        searchable_summary=summary,
    )
