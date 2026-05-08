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
		self.LLM_Model = llm_model
		self.OLLAMA_HOST = host.rstrip('/')
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
		self.OLLAMA_URL = host.rstrip("/")


	def call_llm(self, prompt: str, format: str = "") -> str:
		payload = {
			"model": self.LLM_MODEL, 
			"prompt": prompt, 
			"stream": False, 
		}
		if format != "":
			payload["format"] = format

		res = requests.post(
			f"{self.OLLAMA_URL}/api/generate", 
			json=payload
		)
		res.raise_for_status()
		return res.json()['response']
	

	def extract_triplets(self, text: str) -> List[Dict[str, str]]:
		# prompt = f"""Extract entities and relationships from the following text as a JSON list of objects.
		# Format: [{{"subject": "name", "relation": "description", "object": "name", "type": "category"}}]
		# Text: {text}
		# JSON:"""
		prompt = f"""You are an expert data extraction algorithm. Your task is to extract an exhaustive list of entities and their relationships from the given text.
		
		RULES:
		1. You MUST extract as many meaningful subject-relation-object triplets as possible.
		2. You MUST respond ONLY with a valid JSON array of objects.
		3. Each object MUST have exactly these four keys: "subject", "relation", "object", "type". Do not add any other keys.
		4. The "type" should be a broad category for the subject (e.g., "person", "location", "organization").

		EXAMPLE INPUT:
		John Smith works at Google. He lives in New York with his dog, Max.
		
		EXAMPLE OUTPUT:
		[
		  {{"subject": "John Smith", "relation": "works at", "object": "Google", "type": "person"}},
		  {{"subject": "John Smith", "relation": "lives in", "object": "New York", "type": "person"}},
		  {{"subject": "John Smith", "relation": "owns", "object": "Max", "type": "person"}},
		  {{"subject": "Max", "relation": "is a", "object": "dog", "type": "animal"}}
		]

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