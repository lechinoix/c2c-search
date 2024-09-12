from pinecone import Pinecone
from typing import List
from sentence_transformers import SentenceTransformer

from c2c_search.types import CamptocampDocument, IndexEntry

pc = Pinecone(api_key="fake")
model = SentenceTransformer("all-mpnet-base-v2")
INDEX_NAME = "camptocamp"


def to_index(camptocamp_document: CamptocampDocument) -> IndexEntry:
    doc_id = camptocamp_document.id

    text_to_embed = f"{camptocamp_document.title} / difficulté: {camptocamp_document.global_rating} / {camptocamp_document.summary}".strip()

    # Generate embedding
    embedding = model.encode([text_to_embed])[0].tolist()
    print(f"Document {doc_id} encoded")

    # Create metadata
    metadata = {
        "title": camptocamp_document.title,
        "summary": camptocamp_document.summary,
        "elevation_max": camptocamp_document.elevation_max,
        "global_rating": camptocamp_document.global_rating,
        "rock_free_rating": camptocamp_document.rock_free_rating,
        "activities": camptocamp_document.activities,
    }

    return IndexEntry(id=str(doc_id), values=embedding, metadata=metadata)


def upload_to_pinecone(index_entries: List[IndexEntry], index_name=INDEX_NAME):
    index = pc.Index(index_name)
    to_upsert = [(entry.id, entry.values, entry.metadata) for entry in index_entries]
    index.upsert(vectors=to_upsert)


def search_courses(query, index_name=INDEX_NAME, top_k=5):
    # Generate embedding for the query
    query_embedding = model.encode([query])[0]

    index = pc.Index(index_name)

    # Search in Pinecone
    results = index.query(
        vector=query_embedding.tolist(), top_k=top_k, include_metadata=True
    )

    return results
