import os
import time

from library.models import Document


timeout = int(os.environ.get("INGESTION_WAIT_SECONDS", "1800"))
deadline = time.monotonic() + timeout

while time.monotonic() < deadline:
    pending = Document.objects.filter(
        state__in=["queued", "processing"]
    ).count()

    failed = list(
        Document.objects.filter(state="failed").values(
            "pk",
            "filename",
            "error",
        )
    )

    print(
        "Pending:",
        pending,
        "Failed:",
        len(failed),
        flush=True,
    )

    if pending == 0:
        if failed:
            for item in failed:
                print(item, flush=True)

            raise SystemExit(
                "Ingestion finished with failures. "
                "Inspect document details and retry."
            )

        print("All queued ingestion finished successfully.", flush=True)
        break

    time.sleep(5)

else:
    raise SystemExit(
        "Wait timed out. This does not cancel ingestion. "
        "Inspect Index status and worker logs."
    )
