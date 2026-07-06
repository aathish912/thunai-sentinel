from __future__ import annotations

import certifi
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

try:
    from config import Settings
except ImportError:  # pragma: no cover - supports package execution
    from .config import Settings


DOCUMENTS_COLLECTION = "documents"
EVIDENCE_UNITS_COLLECTION = "evidence_units"
DOCUMENT_SUMMARIES_COLLECTION = "document_summaries"
QUERIES_COLLECTION = "queries"
ANSWERS_COLLECTION = "answers"


def get_client(settings: Settings) -> MongoClient:

    return MongoClient(

        settings.mongodb_uri,

        tls=True,

        tlsCAFile=certifi.where(),

    )


def get_database(settings: Settings) -> Database:
    client = get_client(settings)
    return client[settings.mongodb_db_name]


def get_collection(database: Database, name: str) -> Collection:
    return database[name]


def ensure_indexes(database: Database) -> None:
    database[DOCUMENTS_COLLECTION].create_index("doc_id", unique=True)
    database[DOCUMENTS_COLLECTION].create_index([("workspace_id", 1), ("filename", 1)])

    database[EVIDENCE_UNITS_COLLECTION].create_index("evidence_id", unique=True)
    database[EVIDENCE_UNITS_COLLECTION].create_index([("workspace_id", 1), ("doc_id", 1)])
    database[EVIDENCE_UNITS_COLLECTION].create_index("metadata.candidate_name")

    database[QUERIES_COLLECTION].create_index("query_id", unique=True)
    database[ANSWERS_COLLECTION].create_index("answer_id", unique=True)
    database[DOCUMENT_SUMMARIES_COLLECTION].create_index([("workspace_id", 1), ("doc_id", 1)])
