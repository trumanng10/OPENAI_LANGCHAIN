# 🧠 **Lab 1 — Set Up LangChain + OpenAI Environment**

## 🎯 Lab Objectives

By the end of this lab, participants will be able to:

1. Set up a working LangChain and OpenAI Python environment (either locally or in Google Colab).
2. Install and verify required libraries (`langchain`, `langchain-openai`, `openai`, and `python-dotenv`).
3. Configure environment variables safely using a `.env` file.
4. Initialize the `ChatOpenAI` model wrapper and execute prompt calls using LangChain’s standardized API.
5. Understand how to measure token usage and how Chat-based models differ from traditional text-completion models.
6. Identify and avoid deprecated code patterns (`OpenAI(model_name=...)`).

---

## 🧩 What You’ll Build

A clean environment to:

* Load API keys securely
* Run test LLM calls via LangChain
* Estimate token counts
* Compare different OpenAI chat models (`gpt-4o`, `gpt-4o-mini`, and optionally `gpt-5-mini`)

---

## ⚙️ Step-by-Step Lab Guide

### **Section 0 — Environment Setup**

**A. In Google Colab:**

```python
print("hello world")  # Press Ctrl+Enter to verify execution
```

**Remove conflicting packages (optional but recommended):**

```python
!pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python \
  xarray google-cloud-bigquery db-dtypes thinc
```

**Install required dependencies:**

```python
!pip install -r ./requirements.txt -q
!python -m pip install -U "langchain>=0.3.10" "langchain-openai>=0.2.0" "openai>=1.40.0" "pydantic>=2.5" "python-dotenv>=1.0.1"
!pip show langchain
```

---

### **Section 1 — Load Environment Variables**

```python
import os
from dotenv import load_dotenv, find_dotenv

# Load API keys from .env
load_dotenv(find_dotenv() or "./.env")

# Masked print for safety
def safe_tail(value, n=4):
    return "None" if not value else ("*" * (len(value) - n) + value[-n:])

print("OPENAI_API_KEY:", safe_tail(os.environ.get("OPENAI_API_KEY")))
print("PINECONE_API_KEY:", safe_tail(os.environ.get("PINECONE_API_KEY")))
print("PINECONE_ENV:", os.environ.get("PINECONE_ENV"))
```

> ✅ **Tip:** Never expose full API keys in printed output or version control.

---

### **Section 2 — Instantiate the LangChain LLM Wrapper**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=512
)
```

Run your first invocation:

```python
resp = llm.invoke("Give me two creative taglines for a Python data app.")
print(resp.content)
```

---

### **Section 3 — Test and Token Estimation**

Now we’ll explore additional features for developers.

```python
# Example output re-use
# output = llm.invoke("What is the temperature in Malaysia")
# Old code (for comparison):
# resp = llm.invoke("Give me two creative taglines for a Python data app.")
# display(output)

# Estimate the number of tokens in a prompt
print(llm.get_num_tokens("What is the temperature in Malaysia?"))
```

> 💡 **Insight:**
> Token counting helps you understand **model cost and efficiency** — each token roughly equals 4 characters of English text.
> Token limits vary by model (e.g., `gpt-4o-mini` ≈ 128k tokens context).

---

### **Section 4 — Understanding Chat Models**

```python
# Chat models handle structured "chat messages" instead of plain text.
# They process conversational history (system, user, assistant) instead of single prompt strings.
# Hence, the interface is 'Chat Message In → Chat Message Out' rather than 'Text In → Text Out'.
```

❌ **Deprecated code (do not use):**

```python
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.7, max_tokens=512)
```

> 🧠 **Explanation:**
> The above `OpenAI()` interface belongs to **legacy LangChain versions** (pre-`langchain-openai` split).
> Always use `ChatOpenAI` for chat-capable models.
> The modern interface allows multi-turn memory, message roles, and better conversation control.

---

### **Section 5 — Version Verification**

```python
import langchain, openai, pydantic
print("LangChain version:", langchain.__version__)
print("OpenAI version:", openai.__version__)
print("Pydantic version:", pydantic.__version__)
```

> ✅ Output confirms compatibility:

```
LangChain version: 0.3.10
OpenAI version: 1.40.0
Pydantic version: 2.8.2
```

---

### **Section 6 — Deliverables**

Learners must submit:

1. Screenshot or log showing successful installation and version check.
2. Output from:

   ```python
   print(resp.content)
   ```

   showing the two creative taglines.
3. Output of:

   ```python
   print(llm.get_num_tokens("What is the temperature in Malaysia?"))
   ```
4. One-sentence reflection on what token estimation helps with.

---

### **Section 7 — Instructor Quick Checklist**

| ✅ Checkpoint                          | Expected Output       |
| ------------------------------------- | --------------------- |
| `.env` file present and loaded        | Masked key tail shown |
| Packages installed correctly          | No import errors      |
| `ChatOpenAI` successfully initialized | Yes                   |
| LLM response printed                  | Two creative taglines |
| Token estimation executed             | Integer token count   |
| Deprecated `OpenAI()` not used        | Correct               |

---

### **Section 8 — Extension Activities (Optional)**

* Modify the prompt temperature (0.0–1.0) and compare creativity.
* Replace `gpt-4o-mini` with `gpt-4o` or `gpt-5-mini` (if available).
* Wrap your LLM calls into a helper function `ask(prompt)` and reuse it.
* Add a simple UI (e.g., Gradio or Streamlit) to interact with your LangChain LLM.
