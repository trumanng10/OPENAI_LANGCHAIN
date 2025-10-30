Here’s the complete **Lab 2: Tools for OpenAI (Function Calling)** guide formatted for your Google Colab course module.
No environment setup or prerequisites — it starts directly from the tool-calling implementation example.

---

# **Lab 2: Tools for OpenAI (Function Calling)**

## **Objective**

This lab introduces the concept of **Tool Calling (Function Calling)** in OpenAI models.
By the end of this lab, participants will be able to:

1. Understand how function calling enables models to interact with external systems.
2. Define callable tools in Python for dynamic data retrieval or automation.
3. Execute and integrate function call outputs into model responses using the OpenAI SDK.
4. Extend the technique to build custom tools (e.g., for weather, finance, or HR data).

---

## **Lab Steps**

### **Step 1: Understanding Tool Calling**

Tool (Function) calling gives OpenAI models controlled access to external functionalities or APIs.
The model can decide when and how to invoke these tools to retrieve data or perform actions.

**Use Cases:**

* Getting today’s weather for a specific location
* Accessing account details for a user
* Issuing a refund for a customer
* Generating custom data like a horoscope or report

Reference:

* [Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
* [LangChain Integration](https://python.langchain.com/docs/integrations/chat/openai/)

---

### **Step 2: Create the Python Script**

In Google Colab, create a new code cell and paste the following:

```python
from openai import OpenAI
import json

client = OpenAI()

# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "get_horoscope",
        "description": "Get today's horoscope for an astrological sign.",
        "parameters": {
            "type": "object",
            "properties": {
                "sign": {
                    "type": "string",
                    "description": "An astrological sign like Taurus or Aquarius",
                },
            },
            "required": ["sign"],
        },
    },
]

# 2. Define the required Input to perform the task
sign="Taurus"

# 3. Create a conversation input
input_list = [
    {"role": "user", "content": "What is the horoscope for August baby?"}
]
```

---

### **Step 3: Send Request to the Model with Tools Defined**

```python
# 4. Send input with tool definition
response = client.responses.create(
    model="gpt-5",
    tools=tools,
    input=input_list,
)

# Save function call outputs for next requests
input_list += response.output
```

Run this cell — the model may produce a **function call suggestion** (e.g., `get_horoscope`).

---

### **Step 4: Execute the Tool Function**

```python
# 5. Execute the function logic for get_horoscope
print("response.output"+"\n")
print("\n" + response.output_text)

```

---

### **Lab Completion**

**End of Lab 2.**
You have successfully built and executed a function-calling example in OpenAI using Google Colab.
This lab demonstrates how **AI Agents** interact with external tools and data in a secure, controlled manner — the foundation of modern Agentic AI workflows.


