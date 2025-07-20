#!/usr/bin/env python3
"""
LLM Client for different providers
"""

import logging
from typing import Optional

# LLM imports
try:
    import openai
except ImportError:
    print("OpenAI not installed - some features may be limited")

try:
    import anthropic
except ImportError:
    print("Anthropic not installed - some features may be limited")

class LLMClient:
    """Enhanced LLM client supporting multiple providers"""
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
            self.model = model or "gpt-4o-mini"
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            self.model = model or "claude-3-sonnet-20240229"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate_response(self, prompt: str, max_tokens: int = 200, temperature: float = 0.1) -> str:
        """Generate response from LLM with error handling"""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if response and response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content is not None:
                        return content.strip()
                return "ERROR: OpenAI returned empty response"
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                if response and response.content and len(response.content) > 0:
                    content = response.content[0].text
                    if content is not None:
                        return content.strip()
                return "ERROR: Anthropic returned empty response"
                
        except Exception as e:
            logging.error(f"LLM generation failed for {self.provider}: {e}")
            return f"ERROR: {self.provider} API call failed - {str(e)}"