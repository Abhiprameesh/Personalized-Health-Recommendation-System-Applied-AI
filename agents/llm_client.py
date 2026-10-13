"""
Shared LLM Client helper for Google Gemini.
Dynamically supports models (including gemini-3.6, gemini-1.5-flash, gemini-2.0-flash)
and handles environment variable loading and resilient fallback.
"""

import os
from typing import Tuple, Optional, Any
from dotenv import load_dotenv


def get_gemini_client() -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """
    Returns (client, model_name, api_key) if valid key exists, otherwise (None, None, None).
    Accepts both GEMINI_API_KEY and LLM_API_KEY.
    """
    # Reload .env in case user updated it live
    load_dotenv(override=True)
    
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
    if not api_key or api_key.startswith("your_") or len(api_key) < 10:
        return None, None, None

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6").strip()
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        client = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2
        )
        return client, model_name, api_key
    except Exception as e:
        print(f"Failed to create ChatGoogleGenerativeAI with model '{model_name}': {e}")
        return None, None, None


def invoke_gemini_safely(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Invokes Gemini with the configured model (defaulting to gemini-3.6).
    If the specified model fails (e.g., model name not provisioned on user's project),
    tries standard fallback endpoints (gemini-1.5-flash, gemini-2.0-flash).
    Returns (response_text, model_used_name) or (None, None).
    """
    client, model_name, api_key = get_gemini_client()
    if not client or not api_key:
        return None, None

    try:
        resp = client.invoke(prompt)
        return resp.content.strip(), model_name
    except Exception as primary_err:
        print(f"Gemini call with model '{model_name}' failed: {primary_err}")
        
        # Try alternate fallback models if primary model is unavailable
        fallback_candidates = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        for alt_model in fallback_candidates:
            if alt_model.lower() == model_name.lower():
                continue
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                alt_client = ChatGoogleGenerativeAI(
                    model=alt_model,
                    google_api_key=api_key,
                    temperature=0.2
                )
                alt_resp = alt_client.invoke(prompt)
                print(f"Fallback to '{alt_model}' succeeded!")
                return alt_resp.content.strip(), alt_model
            except Exception:
                continue

    return None, None
