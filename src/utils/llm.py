from typing import Optional, Any
import time
from anthropic import InternalServerError
import google.generativeai as genai
from openai import OpenAI
import anthropic
import os
import threading
import tiktoken

from src.utils.token_counter import token_counter

def count_tokens(text: str, model: str = "gemini-2.0-flash") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Default to cl100k_base encoding
    return len(encoding.encode(text))

def prompt_llm(
    prompt: str,
    max_retries: int = 8,
    initial_wait: int = 1,
    timeout: int = 30,
    llm_type: str = "gemini-2.0-flash",
    system_prompt: Optional[str] = None,
    temperature: float = 0.8,
    client: Any = None
) -> str:
    """
    Send a prompt to the specified LLM and get the response.
    
    Args:
        prompt: The prompt to send to the LLM
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time between retries (doubles with each retry)
        timeout: Timeout for the API call
        llm_type: Type of LLM to use
        system_prompt: System prompt to use (for models that support it)
        temperature: Temperature parameter for response generation
        client: API client instance
        
    Returns:
        The LLM's response as a string
        
    Raises:
        Exception: If unable to get a response after max_retries
    """
    thread_id = f"thread-{threading.get_ident()}"
    input_tokens = count_tokens(prompt)
    
    for attempt in range(max_retries):
        try:
            if llm_type.startswith("gpt"):
                messages = [
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {"role": "user", "content": prompt}
                ]
                messages = [m for m in messages if m is not None]
                
                response = client.chat.completions.create(
                    model=llm_type,
                    messages=messages,
                    timeout=timeout
                )
                output_text = response.choices[0].message.content
                output_tokens = count_tokens(output_text)
                token_counter.add_tokens(thread_id, input_tokens, output_tokens)
                return output_text
                
            elif llm_type.startswith("claude"):
                response = client.messages.create(
                    model=llm_type,
                    max_tokens=1000,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                output_text = response.content[0].text
                output_tokens = count_tokens(output_text)
                token_counter.add_tokens(thread_id, input_tokens, output_tokens)
                return output_text
                
            elif llm_type.startswith("gemini"):
                model = genai.GenerativeModel(llm_type)
                if llm_type == "gemini-2.0-flash":
                    response = model.generate_content(
                        prompt,
                        safety_settings=[
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
                        ]
                    )
                else:
                    response = model.generate_content(prompt)
                output_text = response.text
                output_tokens = count_tokens(output_text)  # Using same token counter as rough estimate
                token_counter.add_tokens(thread_id, input_tokens, output_tokens)
                return output_text
                
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")

        except (InternalServerError, Exception, TimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            wait_time = initial_wait * (2 ** attempt)
            print(f"Error occurred: {str(e)}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise Exception("Failed to get a response after multiple retries") 