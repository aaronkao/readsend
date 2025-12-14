# ReadSend

ReadSend is a personal tool that breathes life into your Kindle highlights. It ingests your local `My Clippings.txt` file, stores highlights in a vector database (Pinecone) for semantic search, and delivers a daily email digest of 5 random highlights to your inbox.

## Features

- **Kindle Ingestion**: Parses the standard Kindle `My Clippings.txt` file.
- **Vector Search**: Uses Pinecone's inference API (`multilingual-e5-large`) to generate embeddings and store highlights.
- **Daily Digest**: Fetches 5 random highlights (ensuring variety) and emails them via Resend.
- **CLI Reader**: Simple command-line tool to view random highlights on demand.

## Prerequisites

- **Python 3.13+**
- **[uv](https://github.com/astral-sh/uv)** package manager
- **Pinecone Account**: You need an API key and a serverless index.
- **Resend Account**: You need an API key and a verified sending domain (or use the testing domain).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd readsend
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_HOST=your_index_host_url

# Resend Configuration
RESEND_API_KEY=your_resend_api_key
EMAIL_FROM=onboarding@resend.dev  # Or your verified sender
EMAIL_TO=your_email@example.com
```

## Usage

### 1. Ingest Highlights

Place your `My Clippings.txt` file in the project root directory. Then run the ingestion script:

```bash
uv run src/ingest.py
```

This will:
- Parse the clippings file.
- Generate embeddings for each highlight.
- Upsert them into your Pinecone index.

### 2. Send Daily Email

To send the daily email manually (or via a cron job):

```bash
uv run src/email_daily.py
```

### 3. Read Highlights in CLI

To view 5 random highlights directly in your terminal:

```bash
uv run src/daily_read.py
```

## Project Structure

- `src/ingest.py`: Handles parsing and uploading highlights to Pinecone.
- `src/daily_read.py`: Logic for fetching random highlights from Pinecone.
- `src/email_daily.py`: Formats and sends the email digest using Resend.
- `src/parser.py`: Utility for parsing the raw Kindle clippings format.
