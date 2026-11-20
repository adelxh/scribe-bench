import asyncio  # Provides async utilities like asyncio.sleep() for non-blocking delays
import os  # Gives access to operating system environment variables

import httpx  # Async HTTP client library used to make API requests
from dotenv import load_dotenv  # Loads key-value pairs from a .env file into environment variables

load_dotenv()  # Read server/.env so OPENROUTER_API_KEY is available via os.getenv()

# Model used for fact extraction, separate from the note models to avoid sharing their rate limits.
EXTRACT_MODEL = "nvidia/nemotron-3.5-lightning:free"
# Models users can pick in the UI for note generation: {id: label}.
NOTE_MODELS = {  # Dictionary mapping OpenRouter model IDs to human-readable UI names
    "inclusionai/ling-3.2-flash:free": "Ling 3.2 Flash",  # Free Ling 3.2 Flash model option
    "nvidia/nemotron-3.5-lightning:free": "NVIDIA Nemotron 3.5 Lightning",  # Free NVIDIA Nemotron model option
} 

# Default model (also used for fact extraction and /submit). Note generation passes the
# user's choice per request instead of rewriting this global, so concurrent users can't clash.
MODEL = "inclusionai/ling-3.2-flash:free"  # Default model ID used when no model is explicitly passed
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"  # OpenRouter's chat completions API endpoint


class LLMError(Exception):  # Custom exception class for LLM-specific errors
    pass  # Inherits all standard Exception behavior without adding extra fields


async def chat(  # Async function that sends a chat conversation to OpenRouter and returns the reply text
    messages: list[dict], client: httpx.AsyncClient | None = None, model: str = MODEL  # Takes chat messages, optional reusable HTTP client, and model ID
) -> str:  # Returns the model's response as a plain string
    key = os.getenv("OPENROUTER_API_KEY")  # Fetch the OpenRouter API key from environment variables
    if not key:  # Check if the API key is missing or empty
        raise LLMError("OPENROUTER_API_KEY is not set in server/.env")  # Stop early with a clear error if no key is configured

    owns_client = client is None  # Track whether this function created its own HTTP client (so it knows whether to close it)
    client = client or httpx.AsyncClient(timeout=180)  # Reuse the provided client, or create a new one with a 180-second timeout
    try:  # Wrap the request loop in try/finally so a locally created client is always closed
        for attempt in range(3):  # Try the API request up to 3 times (attempt = 0, 1, 2)
            r = await client.post(  # Send an async HTTP POST request to OpenRouter
                OPENROUTER_URL,  # Target API endpoint URL
                headers={"Authorization": f"Bearer {key}"},  # Authenticate using the Bearer token header
                json={"model": model, "messages": messages, "temperature": 0},  # Send the model, messages, and temperature=0 for deterministic output
            )  # End of POST request
            # Free models fail transiently (rate limits / provider errors): retry.
            if r.status_code == 200 or (r.status_code != 429 and r.status_code < 500):  # Stop retrying on success (200) or non-retryable client errors (not 429 or 5xx)
                break  # Exit the retry loop
            await asyncio.sleep(2 * (attempt + 1))  # Wait before retrying: 2s after attempt 0, 4s after attempt 1, 6s after attempt 2
    finally:  # Always runs whether the try block succeeded or raised an error
        if owns_client:  # Only close the client if this function created it
            await client.aclose()  # Cleanly close the async HTTP client and release connections

    data = r.json()  # Parse the JSON response body into a Python dictionary
    if r.status_code != 200:  # If the final response was still not a success
        print(f"OpenRouter error {r.status_code} for {model}: {r.text[:500]}", flush=True)  # Show the real reason in the server log
        raise LLMError(data.get("error", {}).get("message", r.reason_phrase))  # Raise an LLMError with the API's error message (or HTTP status text as fallback)
    choices = data.get("choices") or [{}]  # Safely get the list of response choices, defaulting to a single empty dict
    return (choices[0].get("message", {}).get("content") or "").strip()  # Extract the first choice's text content, default to "", and strip leading/trailing whitespace
