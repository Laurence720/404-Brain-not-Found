import os, glob, json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_docs(root="data/teaching_corpus"):
    docs = []
    for p in glob.glob(f"{root}/**/*", recursive=True):
        if os.path.isdir(p): 
            continue
        text = None
        if p.lower().endswith((".md",".txt",".rst")):
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
        elif p.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(p)
                text = "\n".join([pg.extract_text() or "" for pg in reader.pages])
            except Exception:
                pass
        elif p.lower().endswith(".html"):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(Path(p).read_text(encoding="utf-8", errors="ignore"), "lxml")
            text = soup.get_text(" ", strip=True)
        if text and text.strip():
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(p)}))
    return docs

def get_embedder():
    # Default to local sentence-transformers; set the WATSONX_EMBED=1 env var to switch to watsonx embeddings
    if os.getenv("WATSONX_EMBED", "0") == "1":
        from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
        project_id = os.getenv("WATSONX_PROJECT_ID") or os.getenv("PROJECT_ID")
        model_id = os.getenv("WATSONX_EMBED_MODEL", "ibm/slate-125m-english-rtrvr")
        emb = Embeddings(model_id=model_id, project_id=project_id)
        class _Embed:
            def embed(self, texts): return emb.embed_documents(texts)
        return _Embed()
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(os.getenv("SENTENCE_MODEL","all-MiniLM-L6-v2"))
        class _Embed:
            def embed(self, texts): 
                return m.encode(texts, normalize_embeddings=True).tolist()
        return _Embed()

def main():
    import faiss, numpy as np
    os.makedirs("data/vectordb", exist_ok=True)

    docs = load_docs()
    if not docs:
        print("No docs in data/teaching_corpus/. Add md/txt/pdf then rerun.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embedder = get_embedder()

    texts = [c.page_content for c in chunks]
    metas = [{"source": c.metadata.get("source",""), "text": c.page_content} for c in chunks]
    vecs = embedder.embed(texts)

    import numpy as np, faiss
    mat = np.array(vecs, dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    faiss.write_index(index, "data/vectordb/teaching.index")
    Path("data/vectordb/teaching.meta.json").write_text(
        json.dumps(metas, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done. {len(chunks)} chunks -> data/vectordb")

if __name__ == "__main__":
    main()
