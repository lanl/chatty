"""Write or reopen the upgrade fixture with the installed Chatty and FAISS."""

import sys
from pathlib import Path

import faiss
import numpy as np

from chatty.client.openai_client import Message
from chatty.config import load_config
from chatty.core.session import Session, SessionMetadata, load_session, save_session

settings = load_config().config
sessions = settings.get_session_path()
workspace = settings.get_rag_workspace()
assert workspace == Path("workspace")
index_file = workspace / "indices/chunks.faiss"
vector = np.array([[1, 0]], dtype="float32")

if sys.argv[1] == "create":
    index_file.parent.mkdir(parents=True)
    index = faiss.IndexFlatIP(2)
    index.add(vector)
    faiss.write_index(index, str(index_file))
    session = Session(
        SessionMetadata.create("Before upgrade", "model", 1),
        "",
        [Message(role="user", content="Hello")],
    )
    save_session(session, sessions)
else:
    index = faiss.read_index(str(index_file))
    scores, ids = index.search(vector, 1)
    assert (scores.item(), ids.item()) == (1.0, 0)
    (session_file,) = sessions.glob("*.json")
    assert load_session(session_file).messages[0].content == "Hello"
