from django.conf import settings
from django.urls import path, re_path
from django.views.static import serve

from library import views


urlpatterns = [
    path("", views.home, name="library"),
    path(
        "document/<int:pk>/",
        views.detail,
        name="document-detail",
    ),
    path(
        "source/<int:pk>/",
        views.source,
        name="document-source",
    ),
    path("search/", views.search_view, name="content-search"),
    path("ask/", views.ask_view, name="question-answering"),
    path("settings/", views.settings_view, name="settings"),
    path("status/", views.status_view, name="index-status"),
    path("api/benchmark/", views.benchmark, name="benchmark"),
    path("metrics/", views.metrics, name="metrics"),
    re_path(
        r"^static/(?P<path>.*)$",
        serve,
        {"document_root": settings.STATIC_ROOT},
        name="static-files",
    ),
]
