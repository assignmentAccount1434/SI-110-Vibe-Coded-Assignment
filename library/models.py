from django.contrib.postgres.indexes import GinIndex
from django.db import models
from pgvector.django import HnswIndex, VectorField


class Preferences(models.Model):
    upload_subdir = models.CharField(
        max_length=200,
        default="uploads",
    )

    lexical_weight = models.FloatField(default=0.5)
    top_k = models.PositiveIntegerField(default=6)

    # Monitoring targets, in seconds.
    # These are not enforced cancellation deadlines.
    search_sla = models.FloatField(default=5)
    embedding_sla = models.FloatField(default=10)
    ingestion_sla = models.FloatField(default=120)


class Document(models.Model):
    # Stable logical identity used to recognize updated imports.
    source_key = models.CharField(
        max_length=1000,
        unique=True,
    )

    filename = models.CharField(max_length=255)
    title = models.CharField(max_length=500)

    category = models.CharField(
        max_length=100,
        blank=True,
    )

    format = models.CharField(max_length=10)
    path = models.TextField()

    # Hash of the latest stored file.
    sha256 = models.CharField(max_length=64)

    # Hash of the version represented by the current index.
    indexed_sha256 = models.CharField(
        max_length=64,
        blank=True,
    )

    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    pages = models.PositiveIntegerField(default=0)

    upload_state = models.CharField(
        max_length=20,
        default="stored",
    )

    # queued, processing, ready, or failed.
    state = models.CharField(
        max_length=20,
        default="queued",
    )

    progress = models.PositiveIntegerField(default=0)

    error = models.TextField(blank=True)
    warning = models.TextField(blank=True)

    heartbeat = models.DateTimeField(null=True)
    seconds = models.FloatField(default=0)


class Chunk(models.Model):
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
    )

    ordinal = models.PositiveIntegerField()
    page = models.PositiveIntegerField(default=1)

    heading = models.CharField(
        max_length=500,
        blank=True,
    )

    text = models.TextField()

    # Token-frequency mapping used for BM25 scoring.
    tokens = models.JSONField()

    # Total token count, including additional identifier tokens.
    length = models.PositiveIntegerField()

    # Fixed to the chosen embedding model's output dimension.
    embedding = VectorField(dimensions=768)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["document", "ordinal"],
                name="document_chunk",
            ),
        ]

        indexes = [
            GinIndex(fields=["tokens"]),
            HnswIndex(
                name="chunk_vector_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]


class Timing(models.Model):
    kind = models.CharField(max_length=40)
    seconds = models.FloatField()

    success = models.BooleanField(default=True)

    created = models.DateTimeField(auto_now_add=True)
    detail = models.TextField(blank=True)
