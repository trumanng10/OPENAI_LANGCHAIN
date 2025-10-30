# Lab: Two-Stage LangChain (Updated: 9 Sep 2025)

## Learning Objectives

By the end of this lab, learners can:

1. Construct a **prompt → model → parser** pipeline with `LangChain`’s runnable syntax.
2. Generate Markdown that embeds runnable **Python** code for a given networking concept.
3. Extract fenced code from Markdown using a **custom extractor** (`RunnableLambda`).
4. Chain multiple models: first to **generate code**, then to **explain it line-by-line**.
5. Compose runnables into a single **sequential chain** that accepts a plain string input.

---

## Step-by-Step Lab Guide

> Use the exact code cells below in order. Where shown, run and inspect outputs.

### Step 1 — Imports and Core Runnable Building Blocks

```python
# Chain 1
# pip install -U langchain langchain-openai

import os, re
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
```

### Step 2 — Build Chain 1 Prompt (Role + Task)

````python
# 1) Prompt (chat-style)
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are an Experienced Network Engineer and Python Programming Expert."),
    ("human",  "Write a Python function to implement the concept of {concept}. "
               "Return Markdown: brief explanation + a fenced ```python code block.")
])
````

**Why:** Establishes domain authority (networking) and enforces deliverable format (Markdown + fenced code).

### Step 3 — Initialize Model for Chain 1 (Code Generation)

```python
# 2) Model (modern OpenAI chat model)
llm1 = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=1024)
```

**Tip:** Lower temperature (0.3–0.6) yields more deterministic code.

### Step 4 — Compose Chain 1 (Prompt → Model → Text)

```python
# 3) Chain = prompt → model → plain text (replaces LLMChain)
chain1 = prompt1 | llm1 | StrOutputParser()
```

**Outcome:** `chain1.invoke({"concept": "..."})` returns Markdown string containing a Python code block.

### Step 5 — Smoke Test Chain 1

```python
# 4) (Test) invoke with a concept
output1 = chain1.invoke({"concept": "Subnetting"})
print(output1)
```

**Expect:** A short explanation plus a fenced block like:

````
```python
def calculate_subnets(...):
    ...
````

````

### Step 6 — Create a Code Extractor (Markdown → Python Source)
```python
# 5) Extractor — pull code from Chain 1’s Markdown
def extract_code(md: str):
    m = re.search(r"```python\s*(.*?)```", md, flags=re.I | re.S)
    code = m.group(1).strip() if m else md  # fallback if no fenced block
    return {"function": code}

extractor = RunnableLambda(extract_code)
````

**Why:** Normalizes Chain 1 output into a dict `{"function": "...python source..."}` for the next chain.

### Step 7 — Build Chain 2 Prompt (Line-by-Line Explainer)

````python
#Chain 2 — explain the code line-by-line
# 6)
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a Technical Expert. Explain Python code line by line, precisely and clearly."),
    ("human",  "Given the function below, explain it LINE by LINE in Markdown.\n\n```python\n{function}\n```")
])
````

**Why:** Forces a granular walkthrough—useful for code reviews and pedagogy.

### Step 8 — Initialize Model for Chain 2 (Explanation)

```python
llm2 = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, max_tokens=1024)

chain2 = prompt2 | llm2 | StrOutputParser()
```

**Tip:** Slightly higher temperature can yield more fluent explanations; adjust if it drifts.

### Step 9 — Compose the Overall Sequential Chain

```python
#7)Compose a simple sequential chain (string → concept → chain1 → extractor → chain2)
# Accept a plain string like "Subnetting", map it to {"concept": "..."}
as_concept = RunnableLambda(lambda s: {"concept": s})

overall_chain = as_concept | chain1 | extractor | chain2
```

**Flow:**
`"Subnetting"` → `{"concept": "Subnetting"}` → Markdown w/ code → `{"function": "<code>"}` → line-by-line explanation.

### Step 10 — Run End-to-End

```python
# 8) Run it (like SimpleSequentialChain.run("Subnetting"))
output = overall_chain.invoke("Subnetting")
print(output)
```

**Expect:** Clear, line-by-line Markdown explanation of the generated function.

---

## Extension Exercises (fast)

* Swap concept: `"CIDR"`, `"VLAN Trunking"`, `"Spanning Tree"`, `"NAT"`.
* Harden extractor: capture non-python fences too (` ```\w+ `) as fallback.
* Add a third chain to **unit-test** the generated function on sample inputs.
* Replace `StrOutputParser()` in Chain 2 with a **structured parser** (e.g., bullets per line).

## Troubleshooting Notes

* **No fenced block found:** extractor falls back to full Markdown; add stricter prompt constraints if needed.
* **Long outputs clipped:** raise `max_tokens` on `llm1`/`llm2`.
* **Inconsistent code quality:** lower `temperature` on `llm1`.
* **Verbose explanations:** lower `temperature` on `llm2` or add “limit to 10 bullets” in the prompt.

