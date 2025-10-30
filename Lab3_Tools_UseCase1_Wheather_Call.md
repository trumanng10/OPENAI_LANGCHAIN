# **Lab 3: Tool Example Using Weather API Call**

## **Objective**

This lab demonstrates how OpenAI’s **function-calling (tool-calling)** capability can connect to a live API — here, **Open-Meteo** — to fetch real-time Malaysia weather data.
By the end of this lab, participants will be able to:

1. Define a callable tool that queries an external REST API.
2. Handle parameter passing between model and Python function.
3. Implement fallbacks when live requests fail.
4. Produce human-readable weather summaries through OpenAI’s response pipeline.

---

## **Lab Steps**

### **Step 1: Define the Weather Tool**

In Google Colab, paste the full code block below.
This section registers a callable function `get_weather()` that retrieves Malaysia weather data via Open-Meteo.

```python
# Malaysia Weather via OpenAI function-calling + Open-Meteo (with fallback)
from openai import OpenAI
import json, datetime, requests

client = OpenAI()

# --- 1) Define a callable tool for weather ---
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a Malaysia city using Open-Meteo; returns temperature, humidity, wind, precipitation.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Malaysia city name, e.g. 'Kuala Lumpur', 'George Town', 'Johor Bahru', 'Kota Kinabalu', 'Kuching', 'Ipoh', 'Malacca City'"
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "Units for temperature/wind. Default metric."
                }
            },
            "required": ["city"]
        },
    },
]
```

---

### **Step 2: Create the Gazetteer and Function Logic**

This block maps Malaysia city names to their coordinates and defines the core `get_weather()` function.
The function tries Open-Meteo first, and if that fails, returns a mock fallback.

```python
# --- 2) Simple gazetteer for Malaysia cities -> (lat, lon) ---
MY_COORDS = {
    "kuala lumpur": (3.1390, 101.6869),
    "george town": (5.4141, 100.3288),
    "penang": (5.4141, 100.3288),
    "johor bahru": (1.4927, 103.7414),
    "kota kinabalu": (5.9804, 116.0735),
    "kuching": (1.5533, 110.3592),
    "ipoh": (4.5975, 101.0901),
    "malacca": (2.1896, 102.2501),
    "malacca city": (2.1896, 102.2501),
    "melaka": (2.1896, 102.2501),
    "putrajaya": (2.9264, 101.6964),
    "shah alam": (3.0738, 101.5183),
    "seri kembangan": (3.0250, 101.7055),
}

def get_weather(city: str, units: str = "metric"):
    key = city.strip().lower()
    if key not in MY_COORDS:
        if "penang" in key and "george town" not in key:
            key = "george town"
        elif "melaka" in key or "malacca" in key:
            key = "malacca city"

    if key not in MY_COORDS:
        return {"error": f"Unknown Malaysia city: '{city}'. Try one of: {sorted(set(MY_COORDS.keys()))[:8]}..."}

    lat, lon = MY_COORDS[key]
    is_metric = (units != "imperial")
    wind_unit = "kmh" if is_metric else "mph"

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
        f"&temperature_unit={'celsius' if is_metric else 'fahrenheit'}"
        f"&wind_speed_unit={'kmh' if is_metric else 'mph'}"
        "&timezone=Asia%2FKuala_Lumpur"
    )

    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json().get("current", {})
        return {
            "city": city,
            "timestamp": data.get("time"),
            "temperature": {"value": data.get("temperature_2m"), "unit": "°C" if is_metric else "°F"},
            "apparent_temperature": {"value": data.get("apparent_temperature"), "unit": "°C" if is_metric else "°F"},
            "humidity": {"value": data.get("relative_humidity_2m"), "unit": "%"},
            "wind_speed": {"value": data.get("wind_speed_10m"), "unit": wind_unit},
            "precipitation": {"value": data.get("precipitation"), "unit": "mm"},
            "weather_code": data.get("weather_code"),
            "source": "open-meteo",
        }
    except Exception as e:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return {
            "city": city,
            "timestamp": now,
            "temperature": {"value": 31.0 if is_metric else 87.8, "unit": "°C" if is_metric else "°F"},
            "apparent_temperature": {"value": 36.0 if is_metric else 96.8, "unit": "°C" if is_metric else "°F"},
            "humidity": {"value": 74, "unit": "%"},
            "wind_speed": {"value": 9 if is_metric else 5.6, "unit": wind_unit},
            "precipitation": {"value": 0.0, "unit": "mm"},
            "weather_code": 1,
            "source": "mock_fallback",
            "note": f"Live request failed: {type(e).__name__}",
        }
```

---

### **Step 3: Start the Conversation**

Ask the model about Malaysia’s weather.

```python
# --- 3) Start the conversation asking for Malaysia weather ---
input_list = [
    {"role": "user", "content": "What's the weather in Kuala Lumpur right now? Use metric units."}
]
```

---

### **Step 4: Let the Model Decide When to Call the Tool**

```python
# --- 4) Ask the model; it should choose to call get_weather ---
response = client.responses.create(
    model="gpt-5",
    tools=tools,
    input=input_list,
)

input_list += response.output
```

---

### **Step 5: Execute Function Calls and Return Outputs**

```python
# --- 5) Execute any function calls & send back outputs ---
for item in response.output:
    if item.type == "function_call":
        if item.name == "get_weather":
            args = json.loads(item.arguments)
            weather = get_weather(**args)
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"weather": weather}, ensure_ascii=False)
            })
```

---

### **Step 6: Get the Final Model Response**

```python
# --- 6) Ask the model to answer with only the tool-produced weather ---
final = client.responses.create(
    model="gpt-5",
    instructions="Respond only with the weather generated by a tool in concise, readable English.",
    tools=tools,
    input=input_list,
)

print("Final output JSON:")
print(final.model_dump_json(indent=2))
print("\nPlaintext:")
print(final.output_text)
```

---

### **Step 7: Observation**

Expected output (simplified):

```
Final output JSON:
{
  "weather": {
    "city": "Kuala Lumpur",
    "temperature": {"value": 31.2, "unit": "°C"},
    "humidity": {"value": 74, "unit": "%"},
    "wind_speed": {"value": 9, "unit": "kmh"},
    "precipitation": {"value": 0.0, "unit": "mm"}
  }
}

Plaintext:
Kuala Lumpur: 31 °C, humid (74 %), light wind (9 km/h), no rain.
```

---

### **Step 8: Discussion**

* This lab demonstrates how **OpenAI models can act as orchestrators** — deciding when to call an external tool and how to integrate its data.
* The `get_weather()` function is where the actual HTTP request and fallback logic live.
* The model performs reasoning + tool usage together, a core capability of **Agentic AI** systems.

---

### **Lab Completion**

You have now completed **Lab 3: Weather API Tool Example**.
You successfully implemented a live API integration using OpenAI’s function-calling mechanism to fetch and present Malaysian weather data.

