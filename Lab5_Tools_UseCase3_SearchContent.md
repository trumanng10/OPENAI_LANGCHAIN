# Lab 3: File Search with OpenAI Vector Stores (RAG-ready)

## Learning Objectives

By the end, learners can:

1. Create and list **Vector Stores** for RAG.
2. Upload files to the **Files API** and attach them to a vector store.
3. Block until ingestion completes using `create_and_poll` / batch APIs.
4. Query with `tools=[{"type":"file_search", "vector_store_ids":[...]}]` and read plain-text answers.
5. (Optional) Inspect/verify store contents and handle common edge cases.

> References
> File Search guide: [https://platform.openai.com/docs/guides/tools-file-search](https://platform.openai.com/docs/guides/tools-file-search)
> LangChain + OpenAI (optional): [https://python.langchain.com/docs/integrations/chat/openai/](https://python.langchain.com/docs/integrations/chat/openai/)

---

## Step-by-Step Lab Guide

### Step 1 — Create a Vector Store (get its ID)

```python
# Step 1: Grab the Vector Store ID
from openai import OpenAI
client = OpenAI()

vs = client.vector_stores.create(name="rag-knowledge")
print("VECTOR_STORE_ID =", vs.id)   # e.g., "vs_abc123"
```

### Step 2 — List Existing Vector Stores (sanity check)

```python
# Step 2: list existing stores
stores = client.vector_stores.list(limit=10)
print([ (s.id, s.name) for s in stores.data ])
```

### Step 3 — Upload a File and Attach to the Vector Store

```python
# Step 3: Upload file and attach them to the vector store
# a) Upload a file to Files API
f = client.files.create(file=open("stevejob.pdf", "rb"), purpose="assistants")

# b) Attach the file to the vector store and wait until processed
client.vector_stores.files.create_and_poll(
    vector_store_id=vs.id,
    file_id=f.id,
)
```

#### (Optional) Batch Upload

```python
# Optional, for batch upload:
with open("a.pdf","rb") as a, open("b.md","rb") as b:
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vs.id,
        files=[a, b],
    )
```

### Step 4 — (Optional) Inspect Vector Store Files

```python
# Optional: list files inside this vector store
files = client.vector_stores.files.list(vector_store_id=vs.id, limit=20)
print([(x.id, x.status, x.last_error) for x in files.data])
```

### Step 5 — Ask Questions Scoped to Your Vector Store

```python
# Step 5: Search
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input="what are the story in Steve job speech",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vs.id]
    }]
)
print(response)

# just the assistant’s plain text
print(response.output_text)
```

---

## Quick Notes (what I think matters)

* Keep `purpose="assistants"` for files you’ll use with tools.
* Use the **exact** `vs.id` you created; persist it if you’ll query later in the notebook/session.
* `create_and_poll` / `upload_and_poll` are your friends—don’t query before ingestion finishes.
* File types: PDFs and text work well for first runs; noisy scans may require OCR first.
* If answers feel generic, tighten the prompt (e.g., “From **the uploaded speech**, summarize the three stories and quote the lines that introduce each.”).

Want me to add a short **evaluation cell** that asks two specific questions (summary + quotes) to prove retrieval is actually happening?
