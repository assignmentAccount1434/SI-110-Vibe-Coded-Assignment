import json
from pathlib import Path

import fitz
from datasets import load_dataset


benchmark_root = Path("benchmarks/scifact")
pdf_root = Path("imports/scifact")

benchmark_root.mkdir(parents=True, exist_ok=True)
pdf_root.mkdir(parents=True, exist_ok=True)

corpus = load_dataset(
    "BeIR/scifact",
    "corpus",
    split="corpus",
)

queries = load_dataset(
    "BeIR/scifact",
    "queries",
    split="queries",
)

qrels = load_dataset(
    "BeIR/scifact-qrels",
    split="test",
)

for index, row in enumerate(corpus, start=1):
    identifier = str(row["_id"])

    if not identifier.isdigit():
        raise ValueError(
            f"Unexpected corpus identifier: {identifier}"
        )

    target = pdf_root / f"{identifier}.pdf"

    if target.exists():
        continue

    title = row.get("title", "") or ""
    text = (title + "\n\n" + row["text"]).strip()

    if not text:
        raise ValueError(f"Empty corpus entry: {identifier}")

    pdf = fitz.open()

    try:
        # Fixed slices are intentionally short to bound page overflow.
        for start in range(0, len(text), 2000):
            page = pdf.new_page()

            remaining = page.insert_textbox(
                fitz.Rect(40, 40, 555, 800),
                text[start:start + 2000],
                fontsize=9,
            )

            if remaining < 0:
                raise RuntimeError(
                    f"PDF text overflow for {identifier}. "
                    "Reduce the page slice size."
                )

        pdf.set_metadata({"title": title})
        pdf.save(target)

    finally:
        pdf.close()

    if index % 250 == 0:
        print(
            f"Processed {index}/{len(corpus)} abstracts.",
            flush=True,
        )

query_map = {
    str(row["_id"]): row["text"]
    for row in queries
}

judgments = {}

for row in qrels:
    if int(row["score"]) > 0:
        query_id = str(row["query-id"])

        judgments.setdefault(query_id, []).append(
            str(row["corpus-id"]) + ".pdf"
        )

cases = [
    {
        "query": query_map[query_id],
        "expected_filenames": sorted(set(filenames)),
    }
    for query_id, filenames in judgments.items()
    if query_id in query_map
]

if not cases:
    raise RuntimeError("No evaluation cases were generated.")

evaluation_file = benchmark_root / "evaluation.json"

evaluation_file.write_text(
    json.dumps(cases, indent=2),
    encoding="utf-8",
)

print(
    f"Available corpus: {len(corpus)} abstracts. "
    f"Evaluation: {len(cases)} queries."
)
print("Evaluation file:", evaluation_file)
