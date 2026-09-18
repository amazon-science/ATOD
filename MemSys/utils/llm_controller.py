# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Dict, Optional
import json
import os
import re
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import sys
import time
import random
from pathlib import Path
from .rate_limiter import wait_for_rate_limit


def resolve_model_id(model_id: Optional[str] = None) -> str:
    """Resolve a model ID from an argument or the ATOD_MODEL_ID environment variable."""
    resolved = model_id or os.environ.get("ATOD_MODEL_ID")
    if not resolved:
        raise ValueError(
            "No model ID configured. Pass --model-id or set ATOD_MODEL_ID."
        )
    return resolved


def extract_json_from_llm_response(response: str) -> str:
    """
    Extracts the first valid JSON object or array from a string,
    handling markdown code blocks and other surrounding text.
    """
    if not response:
        return ""

    # First, try to find a JSON block within ```json ... ``` code blocks
    json_code_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```', response)
    if json_code_match:
        potential_json = json_code_match.group(1).strip()
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            pass

    # If not found in code blocks, try to find standalone JSON objects or arrays
    json_match = re.search(r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}|\[(?:[^\[\]]|(?:\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]))*\])', response)
    if json_match:
        potential_json = json_match.group(1).strip()
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            pass

    # For single-word or simple responses (like status values)
    if len(response.split()) <= 5:
        return response.strip()

    # Try to strip markdown code block manually if still not found
    if response.strip().startswith("```"):
        lines = response.strip().splitlines()
        # Remove first and last line if they are code block markers
        if lines[0].startswith("```") and lines[-1].startswith("```"):
            response = "\n".join(lines[1:-1]).strip()
            try:
                json.loads(response)
                return response
            except Exception:
                pass

    # Fallback: try to find anything that resembles a JSON structure
    simplified_json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
    if simplified_json_match:
        return simplified_json_match.group(1).strip()

    return response.strip()

class LLMClient:
    """Simple client for model calls through the Bedrock runtime."""

    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            model_id: Bedrock model identifier. Defaults to ATOD_MODEL_ID.
        """
        self.model_id = resolve_model_id(model_id)
        config = Config(
            retries={'max_attempts': 10, 'mode': 'adaptive'},
            read_timeout=150,
            connect_timeout=150
        )
        region = (
            os.environ.get("BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        client_options = {
            "service_name": "bedrock-runtime",
            "config": config,
        }
        if region:
            client_options["region_name"] = region
        self.bedrock = boto3.client(**client_options)

    def call(self, prompt, max_tokens=4096, temperature=0.5, max_retries=8):
        """
        Make a simple LLM API call with rate limiting and exponential backoff.

        Args:
            prompt: Text prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts for throttling

        Returns:
            Generated text response
        """
        # Global temperature override (submission setting: temperature 0). When
        # ATOD_FORCE_TEMP is set, every memory-system + judge call uses it, making the
        # run deterministic and consistent with the paper's stated setup.
        import os as _os
        _ft = _os.environ.get('ATOD_FORCE_TEMP')
        if _ft not in (None, ''):
            temperature = float(_ft)

        # Detect provider from model_id to build the correct request/response format
        mid = self.model_id.lower()
        if "anthropic" in mid or mid.startswith("anthropic") or ".anthropic." in mid:
            provider = "anthropic"
        elif (
            "qwen" in mid
            or "moonshot" in mid
            or "kimi" in mid
            or "minimax" in mid
            or "deepseek" in mid
            or "mistral" in mid
            or "zai" in mid
            or "glm" in mid
        ):
            provider = "openai_compat"  # OpenAI-compatible messages format
        else:
            provider = "anthropic"  # Default

        if provider == "anthropic":
            req = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            }
        else:  # openai_compat (Qwen, Kimi, MiniMax, DeepSeek)
            req = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting if available
                wait_for_rate_limit()

                resp = self.bedrock.invoke_model(modelId=self.model_id, body=json.dumps(req))
                out = json.loads(resp["body"].read())

                # Parse response based on provider
                if provider == "anthropic":
                    content = out.get("content", [])
                    if isinstance(content, list):
                        texts = [b.get("text", "") for b in content
                                 if isinstance(b, dict) and (b.get("type") == "text"
                                 or ("text" in b and b.get("type") != "thinking"))]
                        joined = "".join(t for t in texts if t)
                        if joined:
                            return joined
                    return out["content"][0]["text"]
                else:  # openai_compat
                    return out["choices"][0]["message"]["content"]

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')

                if error_code == 'ThrottlingException' and attempt < max_retries:
                    # Exponential backoff with jitter for throttling (capped)
                    base_delay = min(2 ** attempt, 20)  # 1,2,4,8,16,20,20,... capped at 20s
                    jitter = random.uniform(0.5, 1.5)  # Add jitter
                    delay = base_delay * jitter

                    print(f"LLM throttling detected, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    continue
                else:
                    # Re-raise if not throttling or max retries exceeded
                    raise
            except Exception as e:
                # Re-raise non-ClientError exceptions immediately
                raise

class LLMController:
    """
    LLM controller that wraps the simple LLMClient for backward compatibility.

    This maintains the existing interface while using the simpler LLM client internally.
    """

    def __init__(self,
                 backend: str = "bedrock",
                 model: Optional[str] = None
                 ):
        """
        Initialize LLM controller.

        Args:
            backend: LLM backend (only "bedrock" supported)
            model: Model identifier. Defaults to ATOD_MODEL_ID.
        """
        if backend != "bedrock":
            print(f"Warning: Backend '{backend}' not supported. Using Bedrock instead.")

        self.llm_client = LLMClient(model)

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """
        Get completion from LLM with optional response format handling.

        Args:
            prompt: Text prompt
            response_format: Optional response format specification (for JSON)
            temperature: Sampling temperature

        Returns:
            LLM response text
        """
        try:
            # Add JSON formatting instruction if response_format is specified
            if response_format:
                prompt += "\n\nPlease respond with valid JSON format."

            # Call the simple LLM client
            response = self.llm_client.call(prompt, temperature=temperature)

            # Extract JSON if response_format was specified
            if response_format:
                return extract_json_from_llm_response(response)

            return response

        except Exception as e:
            print(f"LLM Error: {e}")
            if response_format:
                return '{"error": "LLM call failed"}'
            return "Error: LLM call failed"
