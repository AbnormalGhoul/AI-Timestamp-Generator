# AI-Timestamp-Generator

AI-Timestamp-Generator is a multimodal Retrieval-Augmented Generation
(RAG) system that automatically:

-   Transcribes audio or video files\
-   Extracts key visual frames\
-   Generates CLIP embeddings\
-   Stores them in a FAISS vector database\
-   Answers questions using audio, visual, and transcript context\
-   Returns timestamps, relevant excerpts, and AI-generated explanations

This project enables question-answering over lectures, videos, podcasts,
and other long-form media.

## Features

### Automatic Transcription

Uses OpenAI Whisper to convert audio and video files into time-aligned
transcript segments.

### Visual Frame Extraction

Extracts frames from uploaded video files using MoviePy and converts
them into CLIP embeddings.

### Vector Search

All transcript and frame embeddings are indexed in a FAISS vector store
for fast semantic retrieval.

### AI Question Answering

Uses a language model (FLAN-T5 by default) to produce answers using
retrieved text and images.

### FastAPI Backend

Provides REST endpoints for uploading media and running semantic search
queries.

## Installation

### 1. Clone the Repository

``` bash
git clone https://github.com/your-username/AI-Timestamp-Generator
cd AI-Timestamp-Generator
```

### 2. Create a Virtual Environment

``` bash
python -m venv venv
```

Activate it (PowerShell):

``` powershell
.env\Scriptsctivate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

## Running the Server

Start the FastAPI application using Uvicorn:

``` bash
uvicorn edurag_api:app --reload
```

Open the interactive API documentation at:

    http://127.0.0.1:8000/docs

## Uploading Audio or Video

Use the `/upload/` endpoint in the API documentation or send a request
manually:

``` bash
curl -X POST "http://127.0.0.1:8000/upload/"   -F "file=@lecture.mp4"
```

The server will:

-   Transcribe the content\
-   Extract frames\
-   Generate CLIP embeddings\
-   Build a FAISS index

## Querying the Content

Send a query to the `/query/` endpoint:

Example payload:

``` json
{
  "query": "When does the professor discuss backpropagation?",
  "top_k": 5
}
```

Example response:

``` json
{
  "answer": "The professor explains backpropagation at around 2:12 in the lecture.",
  "segments": [
    {
      "start_time": 120.0,
      "end_time": 135.0,
      "text": "So backpropagation is...",
      "score": 0.82,
      "link": "lecture.mp4#t=120s"
    }
  ]
}
```

## Technology Stack

  Component          Library / Tool
  ------------------ ------------------------------
  Transcription      OpenAI Whisper
  Frame Extraction   MoviePy
  Embeddings         OpenAI CLIP (image and text)
  Vector Index       FAISS
  Backend API        FastAPI
  Server             Uvicorn
