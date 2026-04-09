from __future__ import annotations

import os
from dotenv import load_dotenv

from pathlib import Path
import datajoint as dj

load_dotenv()

def connect() -> dj.Connection:
    """Configure and open DataJoint connection from .env

    Expected in .env:
        DJ_HOST
        DJ_PORT
        DJ_USER
        DJ_PASS
    
    Returns:
        dj.Connection: Connection instance of database.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)
    
    dj.config["database.host"] = os.environ["DJ_HOST"]
    dj.config["database.port"] = int(os.environ["DJ_PORT"])
    dj.config["database.user"] = os.environ["DJ_USER"]
    dj.config["database.password"] = os.environ["DJ_PASS"]
    dj.config["database.use_tls"] = False

    return dj.conn()