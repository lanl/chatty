"""Exercise the installed Chatty provider and session format with the upgrade fixture."""

import shutil
import sys
from pathlib import Path

from chatty.client.openai_client import Message
from chatty.config import load_config
from chatty.core.session import Session, SessionMetadata, load_session, save_session
from chatty.rag.litkit_provider import LitkitProvider

QUERY = "How much rain did the copper gauge collect?"
TITLE = "Rainfall at the cedar grove"
PASSAGE = (
    "The copper rain gauge collected seventeen millimeters of rain during the overnight storm."
)


def check_ivfpq(workspace: Path) -> None:
    # The two-chunk example is too small to train LitKit's production-size codebook.
    # Use its real vectors/IDs and a one-bit codebook in a separate workspace to
    # exercise the nonempty (ranked IDs, search settings) contract from issue #5.
    import faiss
    from litkit.retrieval.search import faiss_search

    copy = Path("ivfpq")
    shutil.copytree(workspace / "sqlite", copy / "sqlite")
    shutil.copytree(workspace / "indices", copy / "indices")
    path = copy / "indices/chunks.faiss"
    flat = faiss.read_index(str(path))
    ids = faiss.vector_to_array(flat.id_map)
    vectors = flat.index.reconstruct_n(0, flat.ntotal)
    quantizer = faiss.IndexFlatIP(flat.d)
    index = faiss.IndexIVFPQ(quantizer, flat.d, 1, 1, 1, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add_with_ids(vectors, ids)
    faiss.write_index(index, str(path))
    hits, _, settings = faiss_search(path, vectors[:1], 2)
    assert hits and settings["nlist"] == 1 and settings["nprobe"] > 0, settings
    chunks = LitkitProvider(copy).retrieve(QUERY)
    assert any(c.paper_title == TITLE and PASSAGE in c.text for c in chunks), chunks
    print("IVF-PQ retrieval returned the source and passage with search settings", settings)


def probe(mode: str) -> None:
    config = load_config().config
    workspace = config.get_rag_workspace()
    assert workspace == Path("workspace") and config.rag_provider == "litkit"
    sessions = config.get_session_path()
    assert sessions == Path("saved sessions")
    if mode == "create":
        session = Session(
            SessionMetadata.create("Before upgrade", "model", 1),
            "",
            [Message(role="user", content=QUERY)],
        )
        save_session(session, sessions)
        print(TITLE, PASSAGE, sep="\n")
    elif mode == "read":
        (file,) = sessions.glob("*.json")
        session = load_session(file)
        assert session.metadata.name == "Before upgrade" and session.messages[0].content == QUERY
    elif mode == "ivfpq":
        check_ivfpq(workspace)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    probe(sys.argv[1])
