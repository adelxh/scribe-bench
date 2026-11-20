"""Generate notes for a dataset split and save them next to the reference notes.

Usage (from server/):  .venv/bin/python -m scripts.generate_notes [split] [limit]
Example:               .venv/bin/python -m scripts.generate_notes valid 3
"""
import asyncio
import json
import sys

import httpx

from app.dataset import load_aci
from app.llm import LLMError
from app.notes import generate_note


async def main(split: str, limit: int | None):
    ds = load_aci()[split]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    results = []
    async with httpx.AsyncClient(timeout=180) as client:
        for i, row in enumerate(ds, 1):
            print(f"[{i}/{len(ds)}] {row['encounter_id']} ...", flush=True)
            try:
                generated = await generate_note(row["dialogue"], client=client)
            except LLMError as e:
                print(f"   failed: {e}")
                generated = None
            results.append(
                {
                    "encounter_id": row["encounter_id"],
                    "generated_note": generated,
                    "reference_note": row["note"],
                }
            )
    path = f"outputs/{split}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} notes to {path}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "valid"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(main(split, limit))
