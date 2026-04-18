# -*- coding: utf-8 -*-
import json
import os
from typing import Dict, List

from utils.llm_services import LLMService


class ModelAgnosticDataGenerator:
    """
    Generates synthetic Finnish Q&A pairs from Kila corpus chunks.

    Prompt design notes:
    - Ultra-strict 2-pair cap: small models (3B/4B) hallucinate repeating lists
      endlessly without a hard stop instruction.
    - Finnish-only rule enforced in both system prompt and user instruction.
    - Minimum length filters (>10 chars Q, >20 chars A) drop garbage tokens.
    """

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def generate_qa_pairs(self, chunk_text: str) -> List[Dict[str, str]]:
        system_prompt = (
            "Olet suomalaisen kirjanpidon asiantuntija. "
            "Tehtäväsi on luoda koulutusaineistoa Kilan (Kirjanpitolautakunta) tekstien pohjalta. "
            "Vastaa aina suomeksi."
        )
        prompt = f"""Lue alla oleva Kilan teksti huolellisesti.
Luo TÄSMÄLLEEN 1 tai 2 kysymys-vastaus -paria, joihin teksti antaa selkeän vastauksen.

Säännöt:
1. Kysymyksen ja vastauksen on oltava suomeksi.
2. Vastauksen on perustuttava VAIN annettuun tekstiin.
3. ÄLÄ KOSKAAN toista samaa kysymystä. Jokaisen parin on oltava erilainen.
4. LOPETA heti, kun olet luonut enintään 2 paria. Älä jatka listan luomista loputtomiin.
5. Älä lisää selityksiä tai johdantoja - palauta VAIN JSON.

Teksti:
{chunk_text}

Palauta AINOASTAAN JSON-lista tässä muodossa, ja sulje lista oikein haka- ja aaltosulkeilla:
[
    {{"instruction": "Kysymys 1 tähän?", "response": "Vastaus 1 tähän."}},
    {{"instruction": "Kysymys 2 tähän?", "response": "Vastaus 2 tähän."}}
]"""

        result = self.llm.generate_json(prompt, system_prompt=system_prompt)

        if isinstance(result, list):
            valid = [
                item for item in result
                if (isinstance(item, dict)
                    and isinstance(item.get("instruction"), str)
                    and isinstance(item.get("response"), str)
                    and len(item["instruction"].strip()) > 10
                    and len(item["response"].strip()) > 20)
            ]
            return valid[:2]   # HARD CAP — even if model hallucinates 15 pairs

        # Fallback: model returned a single dict instead of a list
        if (isinstance(result, dict)
                and len(str(result.get("instruction", "")).strip()) > 10
                and len(str(result.get("response", "")).strip()) > 20):
            return [result]

        return []

    @staticmethod
    def save_to_jsonl(qa_pairs: List[Dict[str, str]], filepath: str):
        """Appends pairs to a UTF-8 JSONL file (HuggingFace-compatible format)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
