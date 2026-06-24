"""Quick Qdrant collection count check."""
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(Path(__file__).parent.parent / ".env.development")

url = os.environ["QDRANT_URL"]
if not re.search(r":\d+$", url.split("://", 1)[-1]):
    url = f"{url}:6333"

client = QdrantClient(url=url, api_key=os.environ["QDRANT_API_KEY"])
for col in client.get_collections().collections:
    info = client.get_collection(col.name)
    print(f"{col.name}: {info.points_count} points")
