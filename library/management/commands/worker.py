import logging
import time
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db import (
    OperationalError,
    ProgrammingError,
    close_old_connections,
    transaction,
)
from django.utils import timezone

from library.engine import ingest
from library.models import Document


log = logging.getLogger(__name__)

LEASE_MINUTES = 15
POLL_SECONDS = 2


class Command(BaseCommand):
    help = (
        "Run restartable ingestion with database-backed job claiming."
    )

    def handle(self, *args, **options):
        self.stdout.write("LocalRAG ingestion worker started.")

        while True:
            close_old_connections()

            try:
                cutoff = timezone.now() - timedelta(
                    minutes=LEASE_MINUTES,
                )

                # An embedding request has a bounded timeout shorter
                # than this lease. Very slow extraction or database
                # commits may still require a longer lease.
                recovered = Document.objects.filter(
                    state="processing",
                    heartbeat__lt=cutoff,
                ).update(
                    state="queued",
                    error=(
                        "Interrupted or expired job automatically requeued."
                    ),
                    heartbeat=None,
                )

                if recovered:
                    log.warning(
                        "Requeued %s expired ingestion jobs.",
                        recovered,
                    )

                with transaction.atomic():
                    doc = (
                        Document.objects.select_for_update(
                            skip_locked=True,
                        )
                        .filter(state="queued")
                        .order_by("created", "pk")
                        .first()
                    )

                    if doc is not None:
                        doc.state = "processing"
                        doc.heartbeat = timezone.now()

                        doc.save(
                            update_fields=[
                                "state",
                                "heartbeat",
                            ],
                        )

                if doc is None:
                    time.sleep(POLL_SECONDS)
                    continue

                log.info(
                    "Processing document %s: %s",
                    doc.pk,
                    doc.filename,
                )

                ingest(doc)

            except (OperationalError, ProgrammingError):
                log.warning(
                    "Waiting for database availability or initialization.",
                    exc_info=True,
                )
                time.sleep(5)

            except Exception:
                log.exception("Unexpected worker-loop error.")
                time.sleep(5)
