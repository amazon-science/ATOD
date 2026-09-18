#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
LLM Utilities for ATODEval

Provides a unified interface for model calls across metric evaluation scripts.
"""

import json
import os
import re
from typing import Dict, Optional
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import time
import random


def resolve_model_id(model_id: Optional[str] = None) -> str:
    """Resolve a model ID from an argument or the ATOD_MODEL_ID environment variable."""
    resolved = model_id or os.environ.get("ATOD_MODEL_ID")
    if not resolved:
        raise ValueError(
            "No model ID configured. Pass --model-id or set ATOD_MODEL_ID."
        )
    return resolved


def wait_for_rate_limit():
    """Simple rate limiting placeholder."""
    time.sleep(0.1)  # Small delay to avoid overwhelming the API


def extract_json_from_llm_response(response: str) -> str:
    """Extracts the first valid JSON object or array from a string,
    handling markdown code blocks and other surrounding text."""
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
        """Initialize the LLM client.

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

    def call(self, prompt, max_tokens=4096, temperature=0.5, max_retries=3):
        """Make a simple LLM API call with rate limiting and exponential backoff.

        Args:
            prompt: Text prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts for throttling

        Returns:
            Generated text response
        """
        req = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ],
        }

        for attempt in range(max_retries + 1):
            try:
                # Apply rate limiting
                wait_for_rate_limit()

                resp = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(req)
                )
                out = json.loads(resp["body"].read())
                return out["content"][0]["text"]

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'ThrottlingException' and attempt < max_retries:
                    # Exponential backoff with jitter for throttling
                    base_delay = 2 ** attempt  # 1s, 2s, 4s, 8s...
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
    """LLM controller that wraps the simple LLMClient for backward compatibility.
    This maintains the existing interface while using the simpler LLM client internally."""

    def __init__(self, backend: str = "bedrock",
                 model_id: Optional[str] = None):
        """Initialize LLM controller.

        Args:
            backend: LLM backend (only "bedrock" supported)
            model_id: Model identifier. Defaults to ATOD_MODEL_ID.
        """
        if backend != "bedrock":
            print(f"Warning: Backend '{backend}' not supported. Using Bedrock instead.")

        self.llm_client = LLMClient(model_id)

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """Get completion from LLM with optional response format handling.

        Args:
            prompt: Text prompt
            response_format: Optional response format specification (for JSON)
            temperature: Sampling temperature

        Returns:
            LLM response text
        """
        # Add JSON formatting instruction if response_format is specified
        if response_format:
            prompt += "\n\nPlease respond with valid JSON format."

        # Call the simple LLM client
        response = self.llm_client.call(prompt, temperature=temperature)

        # Extract JSON if response_format was specified
        if response_format:
            return extract_json_from_llm_response(response)

        return response


def evaluate_with_llm_judge(
    prompt: str,
    model_id: Optional[str] = None,
    verbose: bool = False,
) -> float:
    """
    Evaluate using LLM judge. Returns a score between 0.0 and 1.0.

    Args:
        prompt: Evaluation prompt
        model_id: LLM model identifier

    Returns:
        Score between 0.0 and 1.0
    """
    if verbose:
        print(f"Creating LLMController with model_id: {model_id}")
    controller = LLMController(model_id=model_id)
    if verbose:
        print(f"Calling LLM with prompt length: {len(prompt)}")
    response = controller.get_completion(prompt, temperature=0.1)  # Low temperature for consistency

    if verbose:
        print(f"LLM Response: {response[:200]}...")  # Debug output

    # Try to extract numeric score from response
    score_match = re.search(r'(\d+(?:\.\d+)?)', response)
    if score_match:
        score = float(score_match.group(1))
        # Normalize to 0-1 range if it's on 0-10 scale
        if score > 1.0:
            score = score / 10.0
        return min(1.0, max(0.0, score))
    else:
        raise ValueError(f"Could not extract numeric score from LLM response: {response}")


def evaluate_yes_no_with_llm(
    prompt: str,
    model_id: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """
    Evaluate yes/no question using LLM judge.

    Args:
        prompt: Evaluation prompt expecting YES/NO answer
        model_id: LLM model identifier

    Returns:
        True for YES, False for NO
    """
    if verbose:
        print(f"Creating LLMController for Yes/No with model_id: {model_id}")
    controller = LLMController(model_id=model_id)
    if verbose:
        print(f"Calling LLM for Yes/No with prompt length: {len(prompt)}")
    response = controller.get_completion(prompt, temperature=0.1)  # Low temperature for consistency

    if verbose:
        print(f"LLM Yes/No Response: {response[:200]}...")  # Debug output

    response_lower = response.lower().strip()

    # Look for explicit YES/NO at the beginning or as standalone words
    if re.search(r'\byes\b', response_lower):
        return True
    elif re.search(r'\bno\b', response_lower):
        return False
    else:
        # If no clear yes/no, look for positive/negative indicators
        positive_indicators = ['appropriate', 'helpful', 'effective', 'good', 'relevant', 'timely', 'advance']
        negative_indicators = ['inappropriate', 'unhelpful', 'ineffective', 'bad', 'irrelevant', 'untimely']

        positive_count = sum(1 for word in positive_indicators if word in response_lower)
        negative_count = sum(1 for word in negative_indicators if word in response_lower)

        if positive_count > negative_count:
            return True
        elif negative_count > positive_count:
            return False
        else:
            # Default to True if we can't determine (assume positive)
            if verbose:
                print(f"Warning: Ambiguous response, defaulting to True: {response[:100]}")
            return True
