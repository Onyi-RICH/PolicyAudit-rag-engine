from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

models = list(client.models.list())

print("MODEL COUNT:", len(models))
for m in models:
    print(m.name)
