from .llm import MODEL, chat

SYSTEM_PROMPT = """You are a medical scribe. You will be given a transcript of a conversation between a doctor and a patient. Write the clinical note for the encounter.

Rules:
- Use only information stated in the conversation. Never invent findings, values, medications or history.
- Write in third person, clinical style, concise.
- Omit a section entirely if the conversation has no information for it.
- Use these section headings in capital letters, in this order, each on its own line followed by a blank line:
  CHIEF COMPLAINT
  HISTORY OF PRESENT ILLNESS
  MEDICAL HISTORY
  SOCIAL HISTORY
  FAMILY HISTORY
  MEDICATIONS
  REVIEW OF SYSTEMS
  VITALS
  PHYSICAL EXAM
  RESULTS
  ASSESSMENT AND PLAN
  INSTRUCTIONS
- In ASSESSMENT AND PLAN, number each problem and give its assessment and the plan (treatment, tests, counseling).
- Output only the note, with no preamble, no markdown symbols and no commentary."""


async def generate_note(dialogue: str, client=None, model: str = MODEL) -> str:
    return await chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation:\n\n{dialogue}\n\nWrite the clinical note."},
        ],
        client=client,
        model=model,
    )
