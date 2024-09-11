import requests
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Initialize the BERT model for generating embeddings
model = SentenceTransformer("bert-base-nli-mean-tokens")


def fetch_camptocamp_data(base_url: str, limit: int = 30) -> List[Dict]:
    all_data = []
    offset = 0

    while True:
        url = f"{base_url}&offset={offset}&limit={limit}"
        response = requests.get(url)
        data = response.json()

        documents = data.get("documents", [])
        all_data.extend(documents)

        if len(documents) < limit:
            break

        offset += limit

    return all_data


def format_for_pinecone(data: List[Dict]) -> List[Dict]:
    formatted_data = []

    for item in data:
        doc_id = item["document_id"]

        # Extract the French title and summary if available
        fr_locale = next((loc for loc in item["locales"] if loc["lang"] == "fr"), None)
        title = fr_locale["title"] if fr_locale != None else ""
        summary = fr_locale.get("summary", "") if fr_locale != None else ""

        # Combine title and summary for embedding
        text_to_embed = f"{title} {summary}".strip()

        # Generate embedding
        embedding = model.encode([text_to_embed])[0].tolist()

        # Create metadata
        metadata = {
            "title": title,
            "summary": summary,
            "elevation_max": item.get("elevation_max"),
            "global_rating": item.get("global_rating"),
            "rock_free_rating": item.get("rock_free_rating"),
            "activities": item.get("activities", []),
        }

        formatted_item = {"id": str(doc_id), "values": embedding, "metadata": metadata}

        formatted_data.append(formatted_item)

    return formatted_data


def main():
    base_url = "https://api.camptocamp.org/routes?bbox=575187,5525117,746290,5734057&qa=medium,great&act=mountain_climbing,rock_climbing"

    print("Fetching data from Camptocamp API...")
    raw_data = fetch_camptocamp_data(base_url)
    print(f"Fetched {len(raw_data)} items.")

    print("Formatting data for Pinecone...")
    pinecone_data = format_for_pinecone(raw_data)

    print("Saving formatted data to file...")
    with open("camptocamp_pinecone_data.json", "w", encoding="utf-8") as f:
        json.dump(pinecone_data, f, ensure_ascii=False, indent=2)

    print("Data processing complete. Results saved to 'camptocamp_pinecone_data.json'")


if __name__ == "__main__":
    main()
