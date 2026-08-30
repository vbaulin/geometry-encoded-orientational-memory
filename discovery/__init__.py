# discovery package
import os
from pathlib import Path

def _load_dotenv():
    for path in [Path(".env"), Path("../.env"), Path(__file__).resolve().parent.parent / ".env"]:
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
                break
            except Exception:
                pass

_load_dotenv()

