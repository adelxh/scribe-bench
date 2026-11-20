from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .dataset import load_aci
from .facts import extract_facts
from .judge import judge_all
from .llm import MODEL, NOTE_MODELS, LLMError, chat
from .notes import generate_note

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_ds = None


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
    try:
        generated = await generate_note(row["dialogue"], model=body.model)
    except LLMError as e:
        raise HTTPException(502, str(e))
    return {
        "encounter_id": row["encounter_id"],
        "dialogue": row["dialogue"],
        "model": body.model,
        "generated_note": generated,
        "reference_note": row["note"],
    }


@app.post("/evaluate")
async def evaluate(body: EvaluateRequest):
    """Extract key facts from the reference note, then judge each against the generated note."""
    try:
        _, sections = await extract_facts(body.reference_note)
        if not any(sections.values()):
            raise HTTPException(502, "No facts could be extracted from the reference note")
        return await judge_all(body.generated_note, sections)
    except LLMError as e:
        raise HTTPException(502, str(e))
