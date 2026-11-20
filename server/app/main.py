import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .cache import DiskCache, digest
from .dataset import load_aci
from .facts import extract_facts
from .judge import judge_all, judge_precision
from .llm import EXTRACT_MODEL, MODEL, NOTE_MODELS, LLMError, chat
from .notes import generate_note

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_ds = None

# Disk-backed caches (survive restarts): first request is slow, repeats are instant.
notes_cache = DiskCache("notes")  # (split, encounter, model) -> generated note
facts_cache = DiskCache("facts")  # hash(note text) -> extracted fact sections
eval_cache = DiskCache("evaluations")  # hash(notes + dialogue) -> full /evaluate result


def _cache_key(split: str, encounter_id: str, model: str) -> str:
    return f"{split}|{encounter_id}|{model}"


async def cached_extract_facts(note: str) -> dict[str, list[str]]:
    key = digest(EXTRACT_MODEL, note)
    sections = facts_cache.get(key)
    if sections is None:
        _, sections = await extract_facts(note)
        if any(sections.values()):  # never cache a failed/empty extraction
            facts_cache.set(key, sections)
    return sections


def get_ds():
    global _ds
    if _ds is None:
        _ds = load_aci()
    return _ds


class SubmitRequest(BaseModel):
    message: str


class EvaluateRequest(BaseModel):
    generated_note: str
    reference_note: str
    dialogue: str


class NoteRequest(BaseModel):
    split: str = "train"
    encounter_id: str
    model: str = MODEL


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/submit")
async def submit(body: SubmitRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "message is required")
    try:
        return {"reply": await chat([{"role": "user", "content": message}])}
    except LLMError as e:
        raise HTTPException(502, str(e))


@app.get("/api/encounters")
def encounters(split: str = "train"):
    ds = get_ds()
    if split not in ds:
        raise HTTPException(404, f"Unknown split. Choose from {list(ds)}")
    return {"split": split, "encounter_ids": list(ds[split]["encounter_id"])}


@app.get("/api/note-models")
def note_models():
    return {
        "default": MODEL,
        "models": [{"id": k, "label": v} for k, v in NOTE_MODELS.items()],
    }


@app.post("/generate-note")
async def generate(body: NoteRequest):
    """Generate a note from a dataset dialogue; also returns the reference note for comparison."""
    ds = get_ds()
    if body.model not in NOTE_MODELS:
        raise HTTPException(400, f"Unknown model. Choose from {list(NOTE_MODELS)}")
    if body.split not in ds:
        raise HTTPException(404, f"Unknown split. Choose from {list(ds)}")
    rows = ds[body.split].filter(lambda r: r["encounter_id"] == body.encounter_id)
    if not len(rows):
        raise HTTPException(404, "encounter_id not found in that split")
    row = rows[0]
    key = _cache_key(body.split, body.encounter_id, body.model)
    generated = notes_cache.get(key)
    cached = generated is not None
    if not cached:
        try:
            generated = await generate_note(row["dialogue"], model=body.model)
        except LLMError as e:
            raise HTTPException(502, str(e))
        if generated:  # never cache an empty reply
            notes_cache.set(key, generated)
    return {
        "encounter_id": row["encounter_id"],
        "dialogue": row["dialogue"],
        "model": body.model,
        "generated_note": generated,
        "reference_note": row["note"],
        "cached": cached,
    }


@app.post("/evaluate")
async def evaluate(body: EvaluateRequest):
    """Recall: reference facts vs generated note. Precision: generated facts vs reference, then dialogue."""
    key = digest(body.generated_note, body.reference_note, body.dialogue)
    hit = eval_cache.get(key)
    if hit is not None:
        return {**hit, "cached": True}
    try:
        ref_sections, gen_sections = await asyncio.gather(
            cached_extract_facts(body.reference_note), cached_extract_facts(body.generated_note)
        )
        if not any(ref_sections.values()):
            raise HTTPException(502, "No facts could be extracted from the reference note")
        recall, precision = await asyncio.gather(
            judge_all(body.generated_note, ref_sections),
            judge_precision(gen_sections, body.reference_note, body.dialogue),
        )
    except LLMError as e:
        raise HTTPException(502, str(e))
    result = {**recall, "precision": precision}
    # Only keep fully judged results: a run hit by 429s shouldn't be frozen with Unjudged rows.
    if not recall["counts"]["unjudged"] and not precision["counts"]["unjudged"]:
        eval_cache.set(key, result)
    return {**result, "cached": False}
