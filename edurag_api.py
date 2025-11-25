import os
import io
import math
import uuid
import time
import tempfile
import logging
import traceback
from typing import List, Dict, Any, Tuple

import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Audio/video processing
import whisper
from moviepy.editor import VideoFileClip
from PIL import Image

# Embedding and models
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel

# FAISS
import faiss

# Configuration 
# Model names 
WHISPER_MODEL = "small"  # whisper model size (tiny, base, small, medium, large)
CLIP_MODEL = "openai/clip-vit-base-patch32"  # HF model id for CLIP
# For the Vision-Language LLM you can plug a model supporting image+text inputs, e.g. BLIP2 variants.
# For prototyping we'll default to a text-only LLM (flan-t5-small) that receives textual "context"
LLM_MODEL = "google/flan-t5-small"  # simple text LLM, we can replace with a vision-LLM if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Segment settings
SEGMENT_DURATION_SECONDS = 10  # length of each transcript/frame segment
SEGMENT_OVERLAP_SECONDS = 1    # overlap for continuity

# Index path
INDEX_DIR = "edurag_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Utilities 
def make_clickable_time(base_seconds: float, video_id: str = "audio_recording.mp3") -> str:
    """Return a dummy clickable link for the lecture. Replace with your LMS/Canvas deep link format."""
    return f"{video_id}#t={int(base_seconds)}s"

# Whisper transcription 
print("Loading Whisper ASR model:", WHISPER_MODEL)
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

def transcribe_audio(audio_path: str) -> List[Dict[str, Any]]:
    """
    Run Whisper on audio file and produce segments with start, end, and text.
    Returns list of segments: [{'start': float, 'end': float, 'text': str}, ...]
    """
    print("Transcribing:", audio_path)
    result = whisper_model.transcribe(audio_path, verbose=False)
    # Whisper returns 'segments' with 'start','end','text'
    segments = []
    for s in result.get("segments", []):
        segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()})
    return segments

# Frame extraction 
def extract_frames_for_segments(video_path: str, segments: List[Dict[str, Any]], max_frames_per_segment=1) -> List[List[Image.Image]]:
    """
    For each segment, extract `max_frames_per_segment` representative frame(s) (PIL Images).
    Chooses the middle timestamp of the segment.
    Returns list-of-lists: images_per_segment[i] -> list of PIL images.
    """
    print("Extracting frames from:", video_path)
    clip = VideoFileClip(video_path)
    images_per_segment = []
    for seg in segments:
        mid_t = max(seg["start"], 0.0) + (seg["end"] - seg["start"]) / 2.0
        mid_t = min(mid_t, clip.duration)
        frames = []
        for k in range(max_frames_per_segment):
            # small jitter to get slightly different frames if more than 1
            t = min(max(mid_t + (k - max_frames_per_segment//2) * 0.5, 0), clip.duration)
            frame = clip.get_frame(t)  # numpy HWC RGB
            pil = Image.fromarray(frame.astype("uint8"), "RGB")
            frames.append(pil)
        images_per_segment.append(frames)
    clip.reader.close()
    if clip.audio is not None:
        clip.audio.reader.close_proc()

    return images_per_segment

# CLIP embedder 
print("Loading CLIP model:", CLIP_MODEL)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Use CLIP text tower to produce float32 L2-normalized embeddings.
    Returns numpy array shape (N, D)
    """
    batch = clip_processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**batch)
    text_emb = text_emb.cpu().numpy()
    # normalize
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    return text_emb.astype(np.float32)

def embed_images(images: List[Image.Image]) -> np.ndarray:
    """
    Use CLIP image tower to produce float32 L2-normalized embeddings.
    images: list of PIL Images
    """
    batch = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**batch)
    img_emb = img_emb.cpu().numpy()
    img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    return img_emb.astype(np.float32)

def combine_embeddings(text_emb: np.ndarray, img_emb: np.ndarray, alpha=0.5) -> np.ndarray:
    """
    Combine text and image embedding vectors (same dim). Weighted average then renormalize.
    alpha: weight for text. 0.5 = equal.
    """
    assert text_emb.shape == img_emb.shape
    comb = alpha * text_emb + (1 - alpha) * img_emb
    comb = comb / np.linalg.norm(comb, axis=1, keepdims=True)
    return comb.astype(np.float32)

# FAISS index helpers 
def create_faiss_index(d: int) -> faiss.IndexFlatIP:
    # We'll use inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(d)
    return index

def save_faiss_index(index: faiss.IndexFlatIP, path: str):
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.IndexFlatIP:
    return faiss.read_index(path)

# MMR retrieval 
def mmr(query_emb: np.ndarray, candidate_embs: np.ndarray, top_k: int = 5, lambda_param: float = 0.7) -> List[int]:
    """
    Maximal Marginal Relevance (simple implementation).
    query_emb: (D,)
    candidate_embs: (N, D)
    returns indices of selected candidates
    """
    selected = []
    N = candidate_embs.shape[0]
    similarity_to_query = (candidate_embs @ query_emb).flatten()  # shape (N,)
    # normalize candidate-candidate similarity matrix
    candidate_sim = candidate_embs @ candidate_embs.T  # (N,N)
    # greedy selection
    remaining = set(range(N))
    if N == 0:
        return []
    # pick top by similarity first
    first = int(np.argmax(similarity_to_query))
    selected.append(first)
    remaining.remove(first)
    while len(selected) < min(top_k, N):
        mmr_score = {}
        for i in remaining:
            relevance = similarity_to_query[i]
            redundancy = max(candidate_sim[i, j] for j in selected)
            mmr_score[i] = lambda_param * relevance - (1 - lambda_param) * redundancy
        next_sel = max(mmr_score.items(), key=lambda x: x[1])[0]
        selected.append(next_sel)
        remaining.remove(next_sel)
    return selected

# LLM responder (text-only prototyping) 
print("Loading text LLM (for prototyping):", LLM_MODEL)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(DEVICE)

def llm_answer(query: str, contexts: List[str], citations: List[Dict[str, Any]], max_length: int = 256) -> str:
    """
    A simple template-based prompt to LLM (text only). For real vision-LLM,
    you'd pass images + text to the model. This function returns an answer string including citations.
    """
    # Build context string (short)
    ctx_text = "\n\n".join([f"[{i}] ({c['start']:.1f}s-{c['end']:.1f}s): {contexts[i]}" for i, c in enumerate(citations)])
    prompt = f"""You are an assistant that answers student questions by using short lecture snippets.
Question: {query}

Relevant snippets:
{ctx_text}

Give a short, correct answer (1-4 sentences), and list the snippets by index that support your answer.
Also include a clickable timestamp link for each cited snippet.
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    out = llm.generate(**inputs, max_length=max_length, do_sample=False)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    # append clickable links for each cited snippet found in answer (simple heuristic: use indices)
    # find numeric references like [0], [1] if present; otherwise include top citations
    cited_indices = []
    for i in range(len(citations)):
        token = f"[{i}]"
        if token in ans:
            cited_indices.append(i)
    if not cited_indices:
        cited_indices = list(range(min(3, len(citations))))
    # add links info
    link_lines = []
    for i in cited_indices:
        start = citations[i]["start"]
        link = make_clickable_time(start, video_id=citations[i].get("source", "audio_recording.mp3"))
        link_lines.append(f"Snippet [{i}]: {link} ({start:.0f}s)")
    ans += "\n\n" + "\n".join(link_lines)
    return ans

# Data store (in-memory demo) 
# For production persist these structures (index files + metadata DB)
class SegmentMeta(BaseModel):
    id: str
    start: float
    end: float
    text: str
    source: str  # filename
    image_paths: List[str] = []

# Global state (simple)
STATE = {
    "segments": [],  # list[SegmentMeta]
    "embeddings": None,  # numpy array (N, D)
    "index": None,  # faiss index
    "dim": None
}

# Pipeline to ingest a file (audio or video) 
def ingest_media(file_path: str, is_video: bool = False, source_name: str = "audio_recording.mp3"):
    """
    Full pipeline:
      - run whisper -> segments
      - if video: extract frame(s) per segment
      - embed text and images with CLIP
      - build FAISS index
    """
    print("Ingesting media:", file_path, "video:", is_video)
    segments = transcribe_audio(file_path)  # whisper segments
    # If segments are empty, create fixed-length segments fallback:
    if not segments:
        # fallback slice audio into fixed windows
        duration = 0.0
        try:
            import soundfile as sf
            info = sf.info(file_path)
            duration = info.duration
        except Exception:
            duration = 0.0
        segs = []
        t = 0.0
        while t < duration:
            segs.append({"start": t, "end": min(t + SEGMENT_DURATION_SECONDS, duration), "text": ""})
            t += SEGMENT_DURATION_SECONDS - SEGMENT_OVERLAP_SECONDS
        segments = segs

    images_per_segment = []
    if is_video:
        images_per_segment = extract_frames_for_segments(file_path, segments, max_frames_per_segment=1)
    else:
        # create blank images or attempt to generate waveform preview — for now use placeholder single-image per segment
        images_per_segment = [[Image.new("RGB", (224, 224), color=(255,255,255))] for _ in segments]

    # Prepare combined embeddings per segment
    texts = [seg["text"] for seg in segments]
    print("Embedding texts (N=%d) ..." % len(texts))
    text_emb = embed_texts(texts)
    # For images we need to flatten images into one list then map back
    flat_images = [img for imgs in images_per_segment for img in imgs]
    print("Embedding images (N=%d) ..." % len(flat_images))
    img_emb_flat = embed_images(flat_images) if len(flat_images) > 0 else np.zeros_like(text_emb)
    # map image embeddings back to per-segment by averaging images per segment
    img_embs_per_segment = []
    idx = 0
    for imgs in images_per_segment:
        n = len(imgs)
        if n > 0:
            emb_slice = img_emb_flat[idx:idx+n]
            avg = np.mean(emb_slice, axis=0)
            avg = avg / np.linalg.norm(avg)
            img_embs_per_segment.append(avg)
        else:
            img_embs_per_segment.append(np.zeros(text_emb.shape[1], dtype=np.float32))
        idx += n
    img_embs_per_segment = np.stack(img_embs_per_segment, axis=0)
    # combine
    combined = combine_embeddings(text_emb, img_embs_per_segment, alpha=0.6)

    # Build FAISS index
    d = combined.shape[1]
    index = create_faiss_index(d)
    index.add(combined)
    # store metadata
    metas = []
    for i, seg in enumerate(segments):
        m = SegmentMeta(
            id=str(uuid.uuid4()),
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            source=source_name,
            image_paths=[],
        )
        metas.append(m)
    STATE["segments"] = metas
    STATE["embeddings"] = combined
    STATE["index"] = index
    STATE["dim"] = d
    # Optionally save index to disk:
    save_faiss_index(index, os.path.join(INDEX_DIR, f"faiss_{source_name}.index"))
    print("Ingest complete: segments:", len(metas), "dim:", d)
    return {"segments": len(metas), "dim": d}

# Querying 
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

def query_index(query: str, top_k: int = 5) -> Dict[str, Any]:
    if STATE["index"] is None:
        raise RuntimeError("Index not ready. Ingest media first.")
    # embed query with CLIP text tower
    q_emb = embed_texts([query])[0]  # (D,)
    # FAISS similarity search to get candidates
    D, I = STATE["index"].search(np.expand_dims(q_emb, axis=0), k=min(50, STATE["embeddings"].shape[0]))
    cand_ids = I[0]  # indices
    cand_embs = STATE["embeddings"][cand_ids]
    # apply MMR to choose top_k diverse
    selected_local = mmr(q_emb, cand_embs, top_k=top_k, lambda_param=0.7)
    selected_indices = [int(cand_ids[i]) for i in selected_local]
    # Build contexts and citations
    contexts = [STATE["segments"][i].text for i in selected_indices]
    citations = [{"start": STATE["segments"][i].start, "end": STATE["segments"][i].end, "source": STATE["segments"][i].source} for i in selected_indices]
    # Get LLM answer
    answer = llm_answer(query, contexts, citations, max_length=256)
    # Provide results with timestamps and snippet text
    results = []
    for idx, sidx in enumerate(selected_indices):
        seg = STATE["segments"][sidx]
        results.append({
            "index": idx,
            "segment_id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "link": make_clickable_time(seg.start, video_id=seg.source)
        })
    return {"answer": answer, "results": results}

# FastAPI app 
app = FastAPI(title="EduRAG Prototype API")

# Configure logging once (near the other global config)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edurag")

# Replace the existing upload_file endpoint with the following:
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts an uploaded file named audio_recording.mp3 or video.mp4
    Saves to temp, determines type, runs ingestion pipeline synchronously.
    This wrapper adds robust error handling and better logging for debugging.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext == "":
        # attempt to infer from content-type header
        content_type = file.content_type or ""
        if "audio" in content_type:
            ext = ".mp3"
        elif "video" in content_type:
            ext = ".mp4"

    if ext not in [".mp3", ".wav", ".m4a", ".flac", ".mp4", ".mov", ".avi", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: '{ext}'")

    tmp = None
    try:
        # Write uploaded bytes to a temporary file
        suffix = ext
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        logger.info(f"Saved upload to temp file: {tmp.name} (original filename: {filename})")

        is_video = ext in [".mp4", ".mov", ".avi", ".mkv"]
        # Run the ingestion pipeline (this may be slow)
        res = ingest_media(tmp.name, is_video=is_video, source_name=filename or "audio_recording.mp3")

        # successful ingestion
        return JSONResponse({"status": "ingested", "meta": res})

    except Exception as e:
        # log full traceback to server console
        tb = traceback.format_exc()
        logger.error("Exception during /upload/: %s\n%s", str(e), tb)

        # Attempt to remove the temporary file if it exists
        try:
            if tmp is not None and os.path.exists(tmp.name):
                os.unlink(tmp.name)
                logger.info("Removed temp file after error: %s", tmp.name)
        except Exception as rm_e:
            logger.warning("Could not remove temp file %s: %s", getattr(tmp, "name", "<unknown>"), str(rm_e))

        # Return a JSON error with a helpful message (but not the full traceback)
        # If you are debugging locally you can include the exception text; don't do that in production.
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}. Check server logs for traceback.")

@app.post("/query/")
def query_endpoint(req: QueryRequest):
    try:
        out = query_index(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return out

@app.get("/status/")
def status():
    ready = STATE["index"] is not None
    return {"ready": ready, "n_segments": len(STATE["segments"]) if STATE["segments"] else 0}


