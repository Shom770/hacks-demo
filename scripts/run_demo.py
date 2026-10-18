"""Process every sample ticket through the full pipeline and print results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_text  # noqa: E402
from app.pipeline import process_ticket  # noqa: E402


def main():
    tickets = json.loads(load_text("sample_tickets.json"))
    results = []
    for ticket in tickets:
        print(f"Processing {ticket['id']}...")
        results.append(process_ticket(ticket))

    out = Path("output")
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Done. {len(results)} tickets processed.")


if __name__ == "__main__":
    main()
