import requests
from typing import List, Dict

from c2c_search.types import CamptocampDocument

base_url = "https://api.camptocamp.org/routes?bbox=575187,5525117,746290,5734057&qa=medium,great&act=mountain_climbing,rock_climbing"


def fetch_camptocamp_data(limit: int = 30) -> List[CamptocampDocument]:
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

    return [format_document(item) for item in all_data]


def format_document(document: Dict) -> CamptocampDocument:
    fr_locale = next((loc for loc in document["locales"] if loc["lang"] == "fr"), None)

    return CamptocampDocument(
        id=str(document["document_id"]),
        title=fr_locale["title"] if fr_locale != None else "",
        summary=fr_locale.get("summary", "") if fr_locale != None else "",
        elevation_max=str(document.get("elevation_max", "")) or "",
        global_rating=document.get("global_rating", "") or "",
        rock_free_rating=document.get("rock_free_rating", "") or "",
        activities=document.get("activities", []) or [],
    )
