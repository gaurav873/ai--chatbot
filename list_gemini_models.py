import google.generativeai as genai
import os

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
    exit(1)

genai.configure(api_key=api_key)

try:
    models = genai.list_models()
    print("Available Gemini models:")
    for model in models:
        print(model)
except Exception as e:
    print(f"Error listing models: {e}") 