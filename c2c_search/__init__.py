import dataclasses
import math
import json
from tqdm import tqdm
from dacite import from_dict

from c2c_search.camptocamp import fetch_camptocamp_data
from c2c_search.pinecone import search_courses, to_index, upload_to_pinecone
from c2c_search.types import CamptocampDocument

BATCH_SIZE = 20
DATA_FILE_PATH = "camptocamp_data.json"


def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_raw_data():
    print("Fetching data from Camptocamp API...")
    camptocamp_documents = fetch_camptocamp_data()
    print(f"Fetched {len(camptocamp_documents)} items.")

    print("Saving camptocamp data to file...")
    with open(DATA_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [dataclasses.asdict(doc) for doc in camptocamp_documents],
            f,
            ensure_ascii=False,
            indent=2,
        )


def update_index(batch_size=BATCH_SIZE):
    print("Loading data from file...")
    camptocamp_documents = load_json_data(DATA_FILE_PATH)
    total_batches = math.ceil(len(camptocamp_documents) / batch_size)

    print("Starting index update...")

    # Upload data in batches
    for i in tqdm(range(0, len(camptocamp_documents), batch_size)):
        print(f"Loading batch {i}/{total_batches}")
        batch = camptocamp_documents[i : i + batch_size]
        index_entries = [
            to_index(from_dict(data_class=CamptocampDocument, data=document))
            for document in batch
        ]
        upload_to_pinecone(index_entries)


def search():
    search_query = "Easy climb in Chartreuse"
    results = search_courses(search_query)

    print(f"Search query: '{search_query}'\n")
    print("Natural language response:")
    print(results)


def debug():
    print("Use this for debug")
