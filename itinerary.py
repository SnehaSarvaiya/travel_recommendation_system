from flask import request, session
from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="openai/gpt-oss-20b",
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=1,
  max_tokens=4096,
  stream=True
)

for chunk in completion:
  if not getattr(chunk, "choices", None):
    continue
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
      print(reasoning, end="")
      if chunk.choices and chunk.choices[0].delta.content is not None:
          print(chunk.choices[0].delta.content, end="")
  
def generate_itinerary(user_message, recommended_places, days, budget, transport, food, interests):
    
    # 🔥 IMPORTANT: Force AI to use only selected places
    places_list = list(dict.fromkeys(recommended_places))  # remove duplicates
    places = ", ".join(places_list) if places_list else "No specific places selected"
    prompt = f"""
    Create a {days}-day travel itinerary.

    ONLY use these destinations:
    {places}

    DO NOT include any other city.
    Format the output in clean HTML:
- Use <h3> for Day headings
- Use <ul><li> for activities
- Use proper paragraphs
- DO NOT use tables or | symbols
- DO NOT use <br>

    Budget: {budget}
    Travel Mode: {transport}
    Food Preference: {food}
    Interests: {interests}

    Include:
    - Day-wise plan
    - Travel route between selected places
    - Cost estimation
    - Local tips
    """
    

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000
    )

    return completion.choices[0].message.content