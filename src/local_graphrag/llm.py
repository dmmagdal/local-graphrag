# llm.py
# This file contains functions for interacting with LLMs via Ollama.


import json
from typing import List, Dict

import requests


class LLM:
	def __init__(
		self, 
		llm_model: str,
		host: str = "http://localhost:11434",
	):
		pass


	def call_llm(self, prompt: str) -> str:
		return ""
	

	def extract_triplets(self, text: str) -> List[Dict[str, str]]:	
		return []
		

	def extract_entities(self, text: str) -> List[Dict[str, str]]:
		return []
	

	def generate_response(self, prompt: str) -> str:
		return ""


class OllamaLLM(LLM):
	def __init__(
		self, 
		llm_model: str,
		host: str = "http://localhost:11434", 
	):
		self.LLM_MODEL = llm_model
		self.OLLAMA_URL = host


	def call_llm(self, prompt: str, format: str = "") -> str:
		res = requests.post(
			f"{self.OLLAMA_URL}/generate", 
			json={
				"model": self.LLM_MODEL, 
				"prompt": prompt, 
				"stream": False, 
				"format": format
			}
		)
		return res.json()['response']
	

	def extract_triplets(self, text: str) -> List[Dict[str, str]]:
		prompt = f"""Extract entities and relationships from the following text as a JSON list of objects.
		Format: [{{"subject": "name", "relation": "description", "object": "name", "type": "category"}}]
		Text: {text}
		JSON:"""
		raw_json = self.call_llm(prompt, format="json")
		try: 
			return json.loads(raw_json)
		except: 
			return []


	def entity_extraction(self, text: str) -> List[Dict[str, str]]:
		prompt = f"""Extract entities from the following text as a JSON list of objects.
		Format: [{{"entity": "name"}}]
		Text: {text}
		JSON:"""
		raw_json = self.call_llm(prompt, format="json")
		try: 
			return json.loads(raw_json)
		except: 
			return []
		

	def generate_response(self, prompt: str) -> str:
		return self.call_llm(prompt)