import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="models/gemini-flash-lite-latest",
    contents="Say OK"
)

print(response.text)
