# Malaysia Weather Function Calling Lab Guide

## Learning Objectives

By the end of this lab, you will be able to:
- **Implement** function calling with real-world API integration
- **Create** custom tools for AI models to interact with external data sources
- **Handle** API failures gracefully with fallback mechanisms
- **Process** geographic coordinates and weather data
- **Build** robust AI applications that combine LLMs with live data

## Prerequisites
- Python 3.7+
- OpenAI API key
- Required packages: `openai`, `requests`

---

## Step-by-Step Lab Guide

### Step 1: Environment Setup

```python
# Install required packages
!pip install openai requests

# Import libraries
from openai import OpenAI
import json
import datetime
import requests

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key-here")  # Replace with your actual API key
```

### Step 2: Understanding Tool Definition

```python
# Define the weather tool
tools = [
    {
        "type": "function",
        "name": "get_weather",  # Tool name that AI will call
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
            "required": ["city"]  # city is mandatory, units is optional
        },
    },
]

print("Tool definition created successfully!")
print(f"Tool name: {tools[0]['name']}")
print(f"Required parameter: {tools[0]['parameters']['required']}")
```

### Step 3: Create City Coordinate Database

```python
# Malaysia cities gazetteer - maps city names to coordinates
MY_COORDS = {
    "kuala lumpur": (3.1390, 101.6869),
    "george town": (5.4141, 100.3288),        # Penang (George Town)
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

def find_city_coordinates(city_name: str):
    """Helper function to find city coordinates with fuzzy matching"""
    key = city_name.strip().lower()
    
    # Handle common aliases
    if "penang" in key and "george town" not in key:
        key = "george town"
    elif "melaka" in key or "malacca" in key:
        key = "malacca city"
    
    if key in MY_COORDS:
        return MY_COORDS[key], key
    else:
        return None, f"Unknown Malaysia city: '{city_name}'. Available cities: {list(MY_COORDS.keys())}"

# Test the coordinate finder
test_cities = ["Kuala Lumpur", "Penang", "Unknown City"]
for city in test_cities:
    coords, result = find_city_coordinates(city)
    print(f"{city}: {result}")
```

### Step 4: Implement the Weather Function

```python
def get_weather(city: str, units: str = "metric"):
    """
    Returns current weather for a Malaysia city.
    Uses Open-Meteo API with fallback to mock data if API fails.
    """
    # Find city coordinates
    coords, result = find_city_coordinates(city)
    if coords is None:
        return {"error": result}
    
    lat, lon = coords
    is_metric = (units != "imperial")
    
    # Build API URL
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
        f"&temperature_unit={'celsius' if is_metric else 'fahrenheit'}"
        f"&wind_speed_unit={'kmh' if is_metric else 'mph'}"
        "&timezone=Asia%2FKuala_Lumpur"
    )
    
    print(f"Calling weather API for {city}...")
    
    try:
        # Make API request with timeout
        response = requests.get(url, timeout=8)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json().get("current", {})
        
        return {
            "city": city,
            "timestamp": data.get("time"),
            "temperature": {"value": data.get("temperature_2m"), "unit": "°C" if is_metric else "°F"},
            "apparent_temperature": {"value": data.get("apparent_temperature"), "unit": "°C" if is_metric else "°F"},
            "humidity": {"value": data.get("relative_humidity_2m"), "unit": "%"},
            "wind_speed": {"value": data.get("wind_speed_10m"), "unit": "km/h" if is_metric else "mph"},
            "precipitation": {"value": data.get("precipitation"), "unit": "mm"},
            "weather_code": data.get("weather_code"),
            "source": "open-meteo",
        }
        
    except Exception as e:
        # Fallback mock data - ensures the tool always returns something
        print(f"API call failed, using fallback data: {e}")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return {
            "city": city,
            "timestamp": now,
            "temperature": {"value": 31.0 if is_metric else 87.8, "unit": "°C" if is_metric else "°F"},
            "apparent_temperature": {"value": 36.0 if is_metric else 96.8, "unit": "°C" if is_metric else "°F"},
            "humidity": {"value": 74, "unit": "%"},
            "wind_speed": {"value": 9 if is_metric else 5.6, "unit": "km/h" if is_metric else "mph"},
            "precipitation": {"value": 0.0, "unit": "mm"},
            "weather_code": 1,
            "source": "mock_fallback",
            "note": f"Live request failed: {type(e).__name__}",
        }

# Test the weather function
test_weather = get_weather("Kuala Lumpur")
print("Weather function test:")
print(json.dumps(test_weather, indent=2))
```

### Step 5: Complete Function Calling Implementation

```python
def get_malaysia_weather_ai(query: str, use_gpt4: bool = True):
    """
    Complete function calling workflow for Malaysia weather queries
    """
    # Choose model (Note: GPT-5 doesn't exist, using available models)
    model = "gpt-4o" if use_gpt4 else "gpt-3.5-turbo"
    
    # Start conversation
    messages = [
        {"role": "user", "content": query}
    ]
    
    print(f"Using model: {model}")
    print(f"User query: {query}")
    
    # Step 1: Initial AI response (may call tools)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Let AI decide whether to use tools
    )
    
    response_message = response.choices[0].message
    messages.append(response_message)
    
    print(f"AI decided to use tools: {len(response_message.tool_calls) if response_message.tool_calls else 0}")
    
    # Step 2: Execute tool calls if any
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_weather":
                # Parse arguments and call our function
                function_args = json.loads(tool_call.function.arguments)
                print(f"Calling get_weather with: {function_args}")
                
                function_response = get_weather(**function_args)
                
                # Add tool response to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_response)
                })
    
    # Step 3: Get final response with tool results
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    
    return final_response.choices[0].message.content

# Test the complete workflow
test_queries = [
    "What's the weather in Kuala Lumpur right now? Use metric units.",
    "How hot is it in Penang today?",
    "Tell me the weather conditions in Johor Bahru with imperial units."
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*50}")
    print(f"TEST {i}: {query}")
    print(f"{'='*50}")
    result = get_malaysia_weather_ai(query)
    print(f"RESULT: {result}")
```

### Step 6: Advanced Features - Error Handling and Validation

```python
def enhanced_weather_tool(city: str, units: str = "metric"):
    """
    Enhanced version with better error handling and validation
    """
    # Input validation
    if not city or not isinstance(city, str):
        return {"error": "City must be a non-empty string"}
    
    if units not in ["metric", "imperial"]:
        return {"error": "Units must be 'metric' or 'imperial'"}
    
    # Get coordinates
    coords, error = find_city_coordinates(city)
    if error:
        return {"error": error}
    
    # Proceed with API call
    return get_weather(city, units)

def test_enhanced_tool():
    """Test various scenarios with the enhanced tool"""
    test_cases = [
        ("Kuala Lumpur", "metric"),      # Normal case
        ("", "metric"),                  # Empty city
        ("Unknown City", "metric"),      # Unknown city
        ("Kuala Lumpur", "invalid"),     # Invalid units
    ]
    
    for city, units in test_cases:
        print(f"\nTesting: city='{city}', units='{units}'")
        result = enhanced_weather_tool(city, units)
        print(f"Result: {json.dumps(result, indent=2)}")

# Run enhanced tests
test_enhanced_tool()
```

## Key Concepts Demonstrated

### 1. **Tool Definition**
- `type: "function"` - specifies this is a callable function
- `name` - the identifier the AI will use
- `description` - helps AI understand when to use the tool
- `parameters` - defines input structure with types and validation

### 2. **API Integration Pattern**
```python
try:
    # Attempt real API call
    response = requests.get(url, timeout=8)
    return process_success(response)
except Exception as e:
    # Fallback to ensure tool always works
    return create_fallback_data()
```

### 3. **Function Calling Workflow**
1. User sends query to AI
2. AI detects need for external data
3. AI calls appropriate tool with parameters
4. System executes the function
5. Results are sent back to AI
6. AI formulates final response

## Exercises for Practice

1. **Add More Cities**: Extend `MY_COORDS` with 5 additional Malaysia cities
2. **Create Temperature Converter**: Build a tool that converts between Celsius and Fahrenheit
3. **Error Reporting**: Modify the tool to provide more detailed error messages
4. **Weather Forecast**: Extend the tool to provide 3-day forecasts
5. **Multiple Locations**: Handle queries asking for weather in multiple cities

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| API timeout | Implement timeout and fallback |
| Unknown city | Provide helpful error with available cities |
| Invalid parameters | Add input validation |
| Network errors | Use try-catch with fallback data |

This lab provides hands-on experience with real-world AI tool implementation, combining LLMs with live data sources while maintaining robustness through error handling and fallbacks.
