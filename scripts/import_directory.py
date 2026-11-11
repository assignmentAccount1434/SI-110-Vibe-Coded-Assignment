from pathlib import Path

from library.engine import store


root = Path("/imports").resolve()

if not root.is_dir():
    raise SystemExit("/imports is not an available directory.")

stored = 0
failed = 0
skipped = 0

for source in sorted(root.rglob("*")):
    if (
        not source.is_file()
        or source.suffix.lower() not in {".pdf", ".tex"}
    ):
        continue

    if root not in source.resolve().parents:
        print("SKIPPED external symlink:", source, flush=True)
        skipped += 1
        continue

    relative = source.relative_to(root)

    try:
        with source.open("rb") as stream:
            doc, notice = store(
                stream,
                source.name,
                "import:" + str(relative),
                (
                    "benchmark"
                    if relative.parts[0] == "scifact"
                    else ""
                ),
            )

        stored += 1
        print(
            relative,
            "document ID:",
            doc.pk,
            notice,
            flush=True,
        )

    except Exception as exc:
        failed += 1
        print("FAILED:", relative, str(exc), flush=True)

print(
    f"Import finished: {stored} stored/unchanged/duplicate, "
    f"{failed} failed, {skipped} skipped.",
    flush=True,
)

if failed:
    raise SystemExit(
        "Some files failed to store. Review the messages above."
    )
