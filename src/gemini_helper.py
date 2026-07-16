import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from mixed_emotion import EMOTION_RESPONSES

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-2.5-flash")


def build_prompt(field, problem, emotion, confidence):
    return f"""You are a helpful learning assistant. A student studying {field} is feeling {emotion} (confidence: {confidence:.1%}) about this problem:

"{problem}"

Provide a clear, supportive response with:
1. Brief acknowledgment of their feeling
2. One specific tip or strategy for {field}
3. One encouraging next step

Use simple, clear language. Keep each point to 1-2 sentences. No markdown formatting."""


def try_gemini(prompt):
    if not GEMINI_API_KEY:
        raise ValueError("No Gemini key configured")
    response = model_gemini.generate_content(prompt)
    return response.text.strip()


def try_openrouter(prompt):
    if not OPENROUTER_API_KEY:
        raise ValueError("No OpenRouter key configured")
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def try_deepseek(prompt):
    if not DEEPSEEK_API_KEY:
        raise ValueError("No DeepSeek key configured")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def try_grok(prompt):
    if not GROK_API_KEY:
        raise ValueError("No Grok key configured")
    client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
    response = client.chat.completions.create(
        model="grok-4.1-fast",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def get_gemini_response(field, problem, emotion, confidence, use_ai=True):
    """
    If use_ai is True: tries Gemini -> OpenRouter -> DeepSeek -> Grok in order.
    Falls back to a predefined template response if AI is disabled OR
    every provider fails (no keys configured, quota exceeded, network error, etc.)
    """
    if not use_ai:
        return EMOTION_RESPONSES[emotion]["response"]

    prompt = build_prompt(field, problem, emotion, confidence)

    providers = [
        ("Gemini", try_gemini),
        ("OpenRouter", try_openrouter),
        ("DeepSeek", try_deepseek),
        ("Grok", try_grok),
    ]

    for name, fn in providers:
        try:
            result = fn(prompt)
            print(f"✅ Response generated via {name}")
            return result
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")
            continue

    # All 4 providers failed — fall back to the template response
    print("⚠️ All AI providers failed — using template fallback")
    return EMOTION_RESPONSES[emotion]["response"]


if __name__ == "__main__":
    result = get_gemini_response(
        field="Computer Science",
        problem="I keep getting a wrong answer no matter what I try in this recursive function",
        emotion="Frustrated",
        confidence=0.96,
    )
    print("\nFinal response:\n", result)

    print("\n--- Testing with AI disabled ---")
    result_template = get_gemini_response(
        field="Computer Science",
        problem="test",
        emotion="Frustrated",
        confidence=0.96,
        use_ai=False,
    )
    print("Template response:\n", result_template)