import yt_dlp
import requests
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer,CrossEncoder
from groq import Groq

GROQ_API_KEY = ""

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=GROQ_API_KEY)

def get_transcript(video_url: str):
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'skip_download': True,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        subtitles = info.get('subtitles', {})
        auto_captions = info.get('automatic_captions', {})
        en_subs = subtitles.get('en') or auto_captions.get('en', [])
        if not en_subs:
            raise ValueError("No English transcript found for this video.")
        vtt_url = next((s['url'] for s in en_subs if s.get('ext') == 'vtt'), None)
        if not vtt_url:
            vtt_url = en_subs[0]['url']
        response = requests.get(vtt_url)
        text = clean_vtt(response.text)
        return text

def clean_vtt(vtt_text: str):
    lines = vtt_text.split('\n')
    seen = set()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('WEBVTT') or '-->' in line or line.isdigit():
            continue
        # remove vtt tags like <00:00:00.000><c>
        line = re.sub(r'<[^>]+>', '', line)
        line = re.sub(r'&amp;', '&', line)
        line = re.sub(r'&nbsp;', ' ', line)
        if line and line not in seen:
            seen.add(line)
            cleaned.append(line)
    return " ".join(cleaned)

def chunk_text(text: str, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def create_db_from_youtube_video_url(video_url: str):
    transcript = get_transcript(video_url)
    chunks = chunk_text(transcript)
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return {"index": index, "chunks": chunks}

def get_response_from_query(db, query, k=4):

    # Step 1: Embed query
    query_embedding = model.encode([query]).astype("float32")

    # Step 2: Retrieve more chunks initially
    initial_k = 20

    _, indices = db["index"].search(query_embedding, initial_k)

    retrieved_docs = [db["chunks"][i] for i in indices[0]]

    # Step 3: Prepare query-document pairs for reranker
    pairs = [[query, doc] for doc in retrieved_docs]

    # Step 4: Get reranker scores
    scores = reranker.predict(pairs)

    # Step 5: Sort documents by reranker score
    scored_docs = list(zip(retrieved_docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Step 6: Keep best top-k documents
    top_docs = [doc for doc, score in scored_docs[:k]]

    docs_page_content = " ".join(top_docs)

    # Step 7: Send to LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions about youtube videos based on the transcript."
            },
            {
                "role": "user",
                "content": f"""
                Answer the following question: {query}

                By searching the following video transcript:
                {docs_page_content}

                Only use the factual information from the transcript.
                If you don't have enough information, say "I don't know".

                Your answers should be verbose and detailed.
                """
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.replace("\n", ""), top_docs