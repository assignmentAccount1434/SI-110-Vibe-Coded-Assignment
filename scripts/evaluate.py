import json
import os
import time

from library.engine import prefs, retrieve
from library.models import Document


evaluation_file = os.environ.get(
    "EVAL_FILE",
    "/imports/evaluation.json",
)

with open(evaluation_file, encoding="utf-8") as source:
    cases = json.load(source)

if not isinstance(cases, list) or not cases:
    raise RuntimeError("Evaluation file must contain a nonempty list.")

documents = Document.objects.filter(
    source_key__startswith="import:scifact/"
)

scope = list(documents.values_list("pk", flat=True))

if not scope:
    raise RuntimeError("Import SciFact before running evaluation.")

if documents.exclude(state="ready").exists():
    raise RuntimeError(
        "All SciFact documents must finish ingestion successfully."
    )

available = set(documents.values_list("filename", flat=True))

for case in cases:
    expected = set(case["expected_filenames"])

    if not expected or not expected.issubset(available):
        raise RuntimeError(
            "Evaluation references missing corpus documents."
        )

report = {
    "configured_chunk_k": prefs().top_k,
    "lexical_weight": prefs().lexical_weight,
    "methods": {},
}

for mode in ["bm25", "hybrid"]:
    recalls = []
    reciprocal_ranks = []
    durations = []

    for index, case in enumerate(cases, start=1):
        start = time.monotonic()

        results = retrieve(
            case["query"],
            scope,
            mode,
        )

        durations.append(time.monotonic() - start)

        found = []

        for result in results:
            filename = result["chunk"].document.filename

            if filename not in found:
                found.append(filename)

        expected = set(case["expected_filenames"])

        recalls.append(
            len(expected.intersection(found)) / len(expected)
        )

        first_relevant_rank = next(
            (
                rank
                for rank, filename in enumerate(found, start=1)
                if filename in expected
            ),
            None,
        )

        reciprocal_ranks.append(
            1 / first_relevant_rank
            if first_relevant_rank
            else 0
        )

        if index % 25 == 0:
            print(
                f"{mode}: {index}/{len(cases)} queries.",
                flush=True,
            )

    report["methods"][mode] = {
        "queries": len(cases),
        "mean_document_recall_at_configured_chunk_k": (
            sum(recalls) / len(recalls)
        ),
        "mean_reciprocal_document_rank": (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
        ),
        "mean_seconds": sum(durations) / len(durations),
    }

print(json.dumps(report, indent=2))

print(
    "Retrieval evaluation only. "
    "These judgments do not evaluate generated answers."
)

print(
    "Several chunks can belong to the same document, so configured "
    "chunk-k is not the same as document-k."
)
