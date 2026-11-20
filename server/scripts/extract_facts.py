"""Extract the key facts from a reference (dataset) note into facts.txt.

Usage (from server/):  .venv/bin/python -m scripts.extract_facts [split] [encounter_id]
Example:               .venv/bin/python -m scripts.extract_facts valid D2N077
"""
import asyncio
import sys

from app.dataset import load_aci
from app.facts import extract_facts



async def main(split: str, encounter_id: str | None):
    ds = load_aci()[split]
    if encounter_id:
        rows = ds.filter(lambda r: r["encounter_id"] == encounter_id)
        if not len(rows):
            sys.exit(f"{encounter_id} not found in split {split}")
        row = rows[0]
    else:
        row = ds[0]

    facts, _ = await extract_facts(row["note"])
    with open("facts.txt", "w") as f:
        f.write(f"Encounter: {row['encounter_id']} ({split})\n\n{facts}\n")
    print(f"Saved facts for {row['encounter_id']} to facts.txt\n")
    print(facts)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "valid", sys.argv[2] if len(sys.argv) > 2 else None))
