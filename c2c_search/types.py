from dataclasses import dataclass
from typing import Dict, List


@dataclass
class IndexEntry:
    id: str
    values: List[float]
    metadata: Dict


@dataclass
class CamptocampDocument:
    id: str
    title: str
    summary: str
    global_rating: str
    elevation_max: str
    rock_free_rating: str
    activities: List[str]
