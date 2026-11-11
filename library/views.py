import json
import math
import statistics
import time
from pathlib import Path

from django.db import transaction
from django.db.models import Avg, Count, Q, Sum
from django.http import FileResponse, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .engine import answer, embed, prefs, retrieve, safe_storage, store
from .models import Chunk, Document, Timing


def page(request, title, template, **context):
    return render(
        request,
        template,
        {"title": title, **context},
    )


def home(request):
    notices = []

    if request.method == "POST":
        action = request.POST.get("action")
        category = request.POST.get("category", "")[:100]

        if action == "upload":
            files = request.FILES.getlist("files")

            if not files:
                notices.append(
                    "No files received; ingestion was not started."
                )

            for uploaded in files:
                try:
                    doc, notice = store(
                        uploaded,
                        uploaded.name,
                        "upload:" + Path(uploaded.name).name,
                        category,
                    )
                    notices.append(
                        f"{uploaded.name}: {notice} Document ID {doc.pk}."
                    )
                except Exception as exc:
                    notices.append(
                        f"{uploaded.name}: upload/storage failed; "
                        f"ingestion not started: {exc}"
                    )

        elif action == "import":
            root = Path("/imports").resolve()
            subdir = request.POST.get("directory", "").strip()
            selected = (root / subdir).resolve()

            if selected != root and root not in selected.parents:
                notices.append("Import path must remain under /imports.")
            elif not selected.is_dir():
                notices.append("Import directory does not exist.")
            else:
                matched = 0

                # Copying is synchronous; extraction and indexing are
                # handled by the background worker.
                for source in selected.rglob("*"):
                    if (
                        not source.is_file()
                        or source.suffix.lower() not in {".pdf", ".tex"}
                    ):
                        continue

                    matched += 1

                    if root not in source.resolve().parents:
                        notices.append(
                            f"Skipped external symlink: {source.name}"
                        )
                        continue

                    try:
                        with source.open("rb") as stream:
                            doc, notice = store(
                                stream,
                                source.name,
                                "import:" + str(source.relative_to(root)),
                                category,
                            )
                        notices.append(
                            f"{source.name}: {notice} "
                            f"Document ID {doc.pk}."
                        )
                    except Exception as exc:
                        notices.append(
                            f"{source.name}: import/storage failed; "
                            f"ingestion not started: {exc}"
                        )

                if matched == 0:
                    notices.append(
                        "No supported PDF or LaTeX files found."
                    )

    documents = Document.objects.order_by("-created", "-pk")
    query = request.GET.get("q", "").strip()
    category = request.GET.get("category", "").strip()
    file_format = request.GET.get("format", "").strip()

    if query:
        documents = documents.filter(
            Q(filename__icontains=query)
            | Q(title__icontains=query)
        )

    if category:
        documents = documents.filter(category__iexact=category)

    if file_format in {"pdf", "tex"}:
        documents = documents.filter(format=file_format)

    return page(
        request,
        "Document library",
        "library.html",
        documents=documents[:200],
        matching_count=documents.count(),
        notices=notices,
        query=query,
        category=category,
        file_format=file_format,
    )


def detail(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    notice = ""

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "retry":
            changed = (
                Document.objects.filter(pk=pk)
                .exclude(state="processing")
                .update(
                    state="queued",
                    progress=0,
                    error="",
                    heartbeat=None,
                )
            )
            notice = (
                "Ingestion queued."
                if changed
                else "Document is already processing."
            )

        elif action == "metadata":
            doc.title = (
                request.POST.get("title", "").strip()[:500]
                or doc.filename
            )
            doc.category = request.POST.get("category", "").strip()[:100]
            doc.save(update_fields=["title", "category", "updated"])
            notice = "Metadata saved."

        elif action == "delete":
            if request.POST.get("confirm_remove") != "yes":
                notice = "Select the removal confirmation checkbox."
            else:
                with transaction.atomic():
                    current = (
                        Document.objects.select_for_update()
                        .get(pk=pk)
                    )

                    if current.state == "processing":
                        notice = (
                            "Wait for processing to finish before removing."
                        )
                    else:
                        # Indexed passages are deleted by CASCADE.
                        # Original files remain on disk for recovery.
                        current.delete()
                        return redirect("/")

        doc.refresh_from_db()

    source_text = ""
    source_error = ""

    if doc.format == "tex":
        try:
            with Path(doc.path).open(encoding="utf-8") as source_file:
                source_text = source_file.read(200000)
        except (OSError, UnicodeError) as exc:
            source_error = str(exc)

    return page(
        request,
        doc.title,
        "detail.html",
        doc=doc,
        notice=notice,
        source_text=source_text,
        source_error=source_error,
        chunks=Chunk.objects.filter(document=doc).count(),
    )


def source(request, pk):
    doc = get_object_or_404(Document, pk=pk)

    try:
        source_file = open(doc.path, "rb")
    except OSError:
        return HttpResponse(
            "Original file is unavailable. Check the storage mount.",
            status=404,
            content_type="text/plain",
        )

    response = FileResponse(
        source_file,
        content_type=(
            "application/pdf"
            if doc.format == "pdf"
            else "text/plain; charset=utf-8"
        ),
        as_attachment=False,
        filename=doc.filename,
    )
    response["X-Content-Type-Options"] = "nosniff"
    return response


def selected_ids(request):
    values = (
        request.POST.getlist("documents")
        if request.method == "POST"
        else request.GET.getlist("documents")
    )

    result = []

    for value in values:
        # A signed bigint fits in at most 19 decimal digits.
        if value.isdigit() and len(value) <= 19:
            identifier = int(value)
            if (
                0 < identifier <= 9223372036854775807
                and identifier not in result
            ):
                result.append(identifier)

    return result


def search_view(request):
    query = request.GET.get("q", "").strip()[:2000]
    mode = request.GET.get("mode", "hybrid")
    ids = selected_ids(request)

    if mode not in {"hybrid", "bm25", "semantic"}:
        mode = "hybrid"

    results = []
    error = ""

    if query:
        try:
            results = retrieve(query, ids, mode)
        except Exception as exc:
            error = str(exc)

    return page(
        request,
        "Content search",
        "search.html",
        query=query,
        mode=mode,
        selected=ids,
        results=results,
        error=error,
        documents=Document.objects.order_by("filename"),
    )


def ask_view(request):
    question = request.POST.get("question", "").strip()[:2000]
    ids = selected_ids(request)
    results = []
    data = None
    error = ""

    if request.method == "POST" and question:
        try:
            results = retrieve(question, ids)
            data = answer(question, results)

            for index, result in enumerate(results, start=1):
                result["citation"] = f"S{index}"
                result["used"] = (
                    result["citation"] in data["citations"]
                )
        except Exception as exc:
            error = str(exc)

    return page(
        request,
        "Ask the collection",
        "ask.html",
        question=question,
        selected=ids,
        results=results,
        data=data,
        error=error,
        documents=Document.objects.order_by("filename"),
    )


def settings_view(request):
    configuration = prefs()
    notice = ""

    if request.method == "POST":
        try:
            subdir = request.POST["upload_subdir"].strip()
            weight = float(request.POST["lexical_weight"])
            top_k = int(request.POST["top_k"])

            if not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("BM25 weight must be between 0 and 1.")

            if not 1 <= top_k <= 8:
                raise ValueError("Retrieved contexts must be between 1 and 8.")

            targets = {
                key: float(request.POST[key])
                for key in [
                    "search_sla",
                    "embedding_sla",
                    "ingestion_sla",
                ]
            }

            if any(
                not math.isfinite(value)
                or not 0 < value <= 86400
                for value in targets.values()
            ):
                raise ValueError(
                    "Timing targets must be greater than 0 "
                    "and no more than 86400 seconds."
                )

            if len(subdir) > 200:
                raise ValueError("Upload subdirectory is too long.")

            safe_storage(subdir)

            configuration.upload_subdir = subdir
            configuration.lexical_weight = weight
            configuration.top_k = top_k

            for key, value in targets.items():
                setattr(configuration, key, value)

            configuration.save()
            notice = (
                "Settings saved. Existing files have not been moved."
            )

        except (KeyError, ValueError, OSError) as exc:
            notice = str(exc)

    return page(
        request,
        "Settings",
        "settings.html",
        configuration=configuration,
        notice=notice,
    )


def status_view(request):
    rows = (
        Timing.objects.values("kind")
        .annotate(count=Count("id"), average=Avg("seconds"))
        .order_by("kind")
    )

    return page(
        request,
        "Index and timing status",
        "status.html",
        documents=Document.objects.order_by("-updated")[:100],
        rows=rows,
        chunk_count=Chunk.objects.count(),
        document_count=Document.objects.count(),
        timings=Timing.objects.order_by("-created")[:50],
    )


@require_POST
def benchmark(request):
    try:
        if request.content_type == "application/json":
            body = json.loads(request.body or "{}")
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object.")
        else:
            body = request.POST

        query = str(
            body.get("query", "definitions and assumptions")
        ).strip()[:2000]

        if not query:
            raise ValueError("Benchmark query must not be empty.")

        repeats = min(10, max(1, int(body.get("repeats", 1))))
        samples = []

        for _ in range(repeats):
            start = time.monotonic()
            embed(query)
            embedding_seconds = time.monotonic() - start

            start = time.monotonic()
            results = retrieve(query)

            samples.append({
                "embedding_seconds": embedding_seconds,
                "hybrid_search_seconds": time.monotonic() - start,
                "result_count": len(results),
            })

        return JsonResponse({
            "samples": samples,
            "median_embedding_seconds": statistics.median(
                sample["embedding_seconds"] for sample in samples
            ),
            "median_hybrid_search_seconds": statistics.median(
                sample["hybrid_search_seconds"] for sample in samples
            ),
            "note": (
                "Hybrid search includes another query embedding. "
                "Warm and cold model states are not controlled."
            ),
        })

    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=400)


def metrics(request):
    configuration = prefs()

    lines = [
        "# TYPE localrag_documents gauge",
        f"localrag_documents {Document.objects.count()}",
        "# TYPE localrag_chunks gauge",
        f"localrag_chunks {Chunk.objects.count()}",
        "# TYPE localrag_document_states gauge",
    ]

    for state in ["queued", "processing", "ready", "failed"]:
        count = Document.objects.filter(state=state).count()
        lines.append(
            f'localrag_document_states{{state="{state}"}} {count}'
        )

    targets = {
        "embedding": configuration.embedding_sla,
        "search": configuration.search_sla,
        "ingestion": configuration.ingestion_sla,
    }

    # Gauges allow these persisted values to decrease after a restore.
    lines.extend([
        "# TYPE localrag_operation_count gauge",
        "# TYPE localrag_operation_seconds_sum gauge",
        "# TYPE localrag_operation_failures gauge",
        "# TYPE localrag_sla_exceedances gauge",
    ])

    for kind in ["embedding", "search", "ingestion", "generation"]:
        rows = Timing.objects.filter(kind=kind)
        totals = rows.aggregate(count=Count("id"), seconds=Sum("seconds"))

        lines.extend([
            f'localrag_operation_count{{kind="{kind}"}} '
            f'{totals["count"]}',
            f'localrag_operation_seconds_sum{{kind="{kind}"}} '
            f'{totals["seconds"] or 0}',
            f'localrag_operation_failures{{kind="{kind}"}} '
            f'{rows.filter(success=False).count()}',
        ])

        if kind in targets:
            lines.append(
                f'localrag_sla_exceedances{{kind="{kind}"}} '
                f'{rows.filter(seconds__gt=targets[kind]).count()}'
            )

    return HttpResponse(
        "\n".join(lines) + "\n",
        content_type="text/plain; version=0.0.4",
    )
