
# **Lab 4: Web Search Tool Integration using LangChain OpenAI**

## **Objective**

This lab introduces how to **bind and invoke tools** directly within the **LangChain OpenAI interface**.
Participants will learn how to:

1. Initialize a lightweight LLM (`gpt-4.1-mini`) via `langchain_openai.ChatOpenAI`.
2. Bind an external tool (in this case, a web-search preview tool) to the model.
3. Invoke the model to automatically use the tool to retrieve up-to-date information.
4. Understand how LangChain handles **tool-binding abstraction** for OpenAI function-calling models.

---

## **Step-by-Step Lab Guide**

### **Step 1: Import the Required Class**

Start your Google Colab cell with the import from **LangChain OpenAI**:

```python
from langchain_openai import ChatOpenAI
```

This class wraps OpenAI chat models with additional LangChain functionality such as structured responses, tool binding, and streaming.

---

### **Step 2: Initialize the Model**

Create an instance of the OpenAI model you want to use.
Here we use a fast and cost-efficient model for testing.

```python
llm = ChatOpenAI(model="gpt-4.1-mini", output_version="responses/v1")
```

**Explanation:**

* `model="gpt-4.1-mini"` selects the small, responsive GPT-4 variant.
* `output_version="responses/v1"` ensures the SDK returns standardized structured responses compatible with tool-calling.

---

### **Step 3: Define a Tool**

LangChain supports “binding” tools in dictionary form.
Here we attach a **web search preview** tool that allows the model to retrieve the latest online content.

```python
tool = {"type": "web_search_preview"}
```

This tells the model it may use a lightweight, OpenAI-hosted web-search function whenever it deems relevant to the query.

---

### **Step 4: Bind the Tool to the Model**

Bind your defined tool to the model before invoking prompts.

```python
llm_with_tools = llm.bind_tools([tool])
```

**Key Idea:**
`bind_tools()` attaches one or more tools to the model so that when invoked, the model can automatically call them in response to a user question.

---

### **Step 5: Invoke the Model**

Ask a real-world question that benefits from web data.

```python
response = llm_with_tools.invoke("What is Malaysia news today?")
```

**What Happens Behind the Scenes:**

1. The model recognizes that answering “Malaysia news today” needs current information.
2. It calls the `web_search_preview` tool automatically.
3. The result is returned as structured JSON and human-readable text.

---

### **Step 6: Display the Response**

You can print or inspect the returned content.

```python
print(response)
```

Typical output (simplified example):

```
{'content': 'Today in Malaysia, government ministers discussed new EV subsidies and increased foreign investment in Penang’s tech sector...'}
```

---

### **Step 7: Experiment**

Try replacing the query with other examples:

* `"Latest fintech regulations in Malaysia"`
* `"Current weather in Penang"`
* `"Top headlines in ASEAN business news"`

Observe how the model dynamically decides to use the web-search tool each time.

---

## **Discussion**

* **Tool binding** in LangChain abstracts the function-calling interface so you don’t manually define JSON schemas as in pure OpenAI SDK calls.
* The **model autonomously decides** whether to invoke the bound tool based on context.
* This approach simplifies rapid prototyping for **agentic AI** workflows involving search, retrieval, or live APIs.

---

## **End of Lab 4**

You have successfully:

* Created a LangChain-wrapped OpenAI chat model
* Bound a web-search tool
* Invoked the model to retrieve real-time information

This lab bridges **OpenAI Function Calling** and **LangChain Tool Abstraction**, forming the basis for multi-tool agents that combine reasoning and real-time data access.
