# localrag

A RAG stack that runs entirely on a laptop: Ollama for embeddings and
generation, LanceDB for vectors, FastAPI in front, all in docker compose.
Same shape as the production pieces in this repo, zero API cost.

```bash
brew install ollama && ollama serve &
ollama pull llama3.1:8b && ollama pull nomic-embed-text
docker compose up --build

curl -F "file=@./docs/sample.pdf" http://localhost:8000/ingest
curl -X POST http://localhost:8000/query -H 'content-type: application/json' -d '{"q":"What is the main topic?","k":3}'
```

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | /health | | `{status, ollama, embed, llm}` |
| GET | /stats | | `{chunks, table}` |
| POST | /ingest | multipart PDF | `{file, chunks}` |
| POST | /query | `{q, k?}` | `{answer, sources: [{page, text}]}` |

Open WebUI is included on port 3000 for manual testing. Unit tests cover
chunking; the API test needs the compose stack (`RUN_INTEGRATION=1`).

```
localrag/
  main.py     FastAPI routes
  config.py   pydantic-settings (OLLAMA_HOST, LANCEDB_URI, EMBED_MODEL, LLM_MODEL)
  ingest.py   PDF -> chunks -> embeddings -> LanceDB
  rag.py      retrieval + generation
Dockerfile, docker-compose.yml
```
