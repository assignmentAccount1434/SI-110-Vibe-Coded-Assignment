import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# The fallback permits collectstatic during the image build.
# Runtime containers receive the generated value from .env.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "build-only-not-for-runtime",
)

DEBUG = False

ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "web",
    "testserver",
]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "tailwind",
    "theme",
    "library",
]

TAILWIND_APP_NAME = "theme"

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "localrag",
        "USER": "localrag",
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": "db",
        "PORT": "5432",
        "CONN_MAX_AGE": 60,
    },
}

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

USE_TZ = True
TIME_ZONE = "UTC"

# Limits non-file request data. Uploaded files are handled separately.
DATA_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024

# Files larger than this are spooled to temporary disk during upload.
FILE_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024

OLLAMA_URL = "http://ollama:11434"

GENERATION_MODEL = os.environ.get(
    "GENERATION_MODEL",
    "llama3.1:8b",
)

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}
