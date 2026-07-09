import os
from dataclasses import dataclass
from pathlib import Path

from config import EXTRACTION_MODEL, VISION_MODEL, load_local_env


def litellm_model_id(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    if model_name.startswith("gemini-"):
        return f"gemini/{model_name}"
    if model_name.startswith("gpt-"):
        return f"openai/{model_name}"
    return model_name


def api_key_env_for_model(model_name: str) -> str | None:
    model_id = litellm_model_id(model_name)
    if model_id.startswith("gemini/"):
        return "GEMINI_API_KEY"
    if model_id.startswith("openai/"):
        return "OPENAI_API_KEY"
    return None


def configure_dspy(model_name: str):
    load_local_env()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if not any(
        os.environ.get(key)
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    ):
        os.environ.setdefault("NO_PROXY", "*")
        os.environ.setdefault("no_proxy", "*")

    model_id = litellm_model_id(model_name)
    api_key_env = api_key_env_for_model(model_id)
    if api_key_env and not os.environ.get(api_key_env):
        raise RuntimeError(f"{api_key_env} is required for live DSPy model calls with {model_id}.")

    import dspy

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    lm = dspy.LM(model_id, cache=False, temperature=0.0)
    dspy.configure(lm=lm)
    return dspy


def make_handwriting_reader(model_name: str = VISION_MODEL, program_path: str | Path | None = None):
    dspy = configure_dspy(model_name)

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

    reader = HandwritingReader()
    if program_path:
        reader.load(str(program_path))
    return dspy, reader


def make_medical_term_extractor():
    dspy = configure_dspy(EXTRACTION_MODEL)

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
