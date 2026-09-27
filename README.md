# 🎥 YouTube Transcript Q&A Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about YouTube videos and receive answers based on the video's transcript.

The system extracts the YouTube transcript, processes it into meaningful chunks, generates semantic embeddings, retrieves relevant information using **FAISS**, reranks the retrieved chunks using a **CrossEncoder**, and generates a contextual response using an **LLM**.

## 🚀 Features

* 📺 Extract transcripts from YouTube videos
* 🧹 Clean and preprocess transcript data
* ✂️ Split transcripts into meaningful chunks
* 🔢 Generate semantic embeddings using Sentence Transformers
* 🔎 Retrieve relevant chunks using FAISS
* 🎯 Rerank retrieved chunks using CrossEncoder
* 🤖 Generate contextual answers using an LLM
* 💬 Interactive Streamlit interface

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **YouTube Transcript API / yt-dlp**
* **Pandas & NumPy**
* **Sentence Transformers**
* **FAISS**
* **CrossEncoder**
* **LLM API**

## 🧠 How It Works

```text
YouTube Video
      ↓
Extract Transcript
      ↓
Clean & Preprocess
      ↓
Text Chunking
      ↓
Sentence Transformer
      ↓
Generate Embeddings
      ↓
FAISS Vector Search
      ↓
Retrieve Top-K Chunks
      ↓
CrossEncoder Reranking
      ↓
Relevant Context
      ↓
LLM
      ↓
Final Answer
```

## 🔍 RAG Pipeline

### 1. Transcript Extraction

The application extracts the transcript from the provided YouTube video and converts it into clean text.

### 2. Text Chunking

The transcript is divided into smaller chunks so that relevant sections can be retrieved efficiently.

### 3. Embedding Generation

Each text chunk is converted into a vector representation using a **Sentence Transformer**.

### 4. FAISS Retrieval

The embeddings are stored in a FAISS index. When the user asks a question, the question is converted into an embedding and compared against the stored vectors to retrieve the most relevant chunks.

### 5. CrossEncoder Reranking

The retrieved chunks are passed through a **CrossEncoder**, which evaluates the relevance between the question and each chunk and reranks them.

### 6. LLM Response Generation

The highest-ranked chunks are provided as context to the LLM along with the user's question. The LLM then generates an answer based on the retrieved transcript content.

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Lestonrego/youtube-rag-assistant.git
cd youtube-rag-assistant
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file:

```env
LLM_API_KEY=your_api_key_here
```

Do not commit your API key to GitHub.

## ▶️ Running the Application

```bash
streamlit run app.py
```

Enter a YouTube URL and ask questions about the video's content.

## 💡 Example

**YouTube URL:**

```text
https://www.youtube.com/watch?v=XXXXXXXXXXX
```

**Question:**

```text
What are the main concepts discussed in this video?
```

**Pipeline:**

```text
Question
   ↓
Query Embedding
   ↓
FAISS Retrieval
   ↓
Top-K Chunks
   ↓
CrossEncoder Reranking
   ↓
Relevant Context
   ↓
LLM
   ↓
Answer
```
