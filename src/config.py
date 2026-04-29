# Imports:
from pathlib import Path
import os

# Dirs
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'

CHROMA_DIR = DATA_DIR / 'chroma'
METRICS_FILE = LOGS_DIR / 'rag_runs.jsonl'

TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '15'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '300'))

# Ensure Directories
def ensure_directories() -> None:
    DATA_DIR.mkdir(parents = True, exist_ok = True)
    LOGS_DIR.mkdir(parents = True, exist_ok = True)
    CHROMA_DIR.mkdir(parents = True, exist_ok = True)