import json
import os
import re
from typing import Dict, Any, Optional
from pydantic import BaseModel


class AgentOutput(BaseModel):
    """Standardized base output for all agent decisions."""
    agent_name: str
    summary: str
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]


class BaseAgent:
    """Base class for all LLM-powered trading agents."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        provider: str = "gemini",
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature

    def call_llm(self, prompt: str) -> str:
        """Invokes LLM with fallback mechanism."""
        # 1. Try litellm if API keys are available
        try:
            import litellm
            litellm.suppress_debug_info = True
            
            # Format model string for litellm
            model = self.model_name
            if self.provider == "gemini" and not model.startswith("gemini/"):
                model = f"gemini/{model}"
            elif self.provider == "openai" and not model.startswith("openai/"):
                model = f"openai/{model}"
            elif self.provider == "anthropic" and not model.startswith("anthropic/"):
                model = f"anthropic/{model}"

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to local heuristic / mock reasoning if no API key is set
            return self._heuristic_fallback(prompt)

    def extract_json(self, raw_text: str) -> Dict[str, Any]:
        """Safely extracts JSON object from LLM output."""
        try:
            # First try direct parse
            return json.loads(raw_text)
        except Exception:
            pass

        # Search for markdown code block ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Search for first { to last }
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw_text[start:end+1])
            except Exception:
                pass

        return {}

    def _heuristic_fallback(self, prompt: str) -> str:
        """Baseline heuristic fallback when LLM API keys are unconfigured."""
        return json.dumps({
            "agent_name": self.name,
            "summary": "Heuristic fallback analysis applied based on quantitative signals.",
            "confidence": 0.60,
            "details": {}
        })
