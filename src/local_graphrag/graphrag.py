# graphrag.py


import json
from pathlib import Path

import pyarrow as pa

from .chunker import Chunker
from .graphdb import LadybugGraphDB
from .llm import OllamaLLM
from .vectordb import VectorDB


class GraphRAG:
	def __init__(self, 
		embed_model_id: str,
		vector_db_path: str,
		graph_db_path: str,
		llm_model: str,
		token_overlap: int = 128,													# Chunker + Embedder kwargs
		batch_size: int = 8,
		device: str = "cpu",
		use_binary: bool = False,
		query_metric: str = "cosine",
		model_save_root: str = Path.home() / ".cache" / "local-graphrag" / "models",
		host: str = "http://localhost:11434",   									# LLM kwargs
	):
		self.chunker = Chunker(
			model_id=embed_model_id,
			token_overlap=token_overlap,
			batch_size=batch_size,
			device=device,
			model_save_root=model_save_root,
		)
		self.vectordb = VectorDB(
			embed_model_id=embed_model_id,
			db_path=vector_db_path,
			token_overlap=token_overlap,
			batch_size=batch_size,
			device=device,
			use_binary=use_binary,
			model_save_root=model_save_root,
		)
		self.graphdb = LadybugGraphDB(
			db_path=graph_db_path,
		)
		self.llm = OllamaLLM(
			llm_model=llm_model,
			host=host,
		)

		self.metric = query_metric


	def get_dims(self) -> int:
		return self.vectordb.get_model_dims()


	def build_vector_table(self, table_name: str, schema: pa.Schema) -> None:
		self.vectordb.create_table(table_name, schema)


	def set_query_metric(self, query_metric: str) -> None:
		self.metric = query_metric


	def ingest(self, text: str, doc_id: str, table_name: str) -> None:
		# Step 1: Chunk & pass the documents to the vectordb.
		chunked_text = self.chunker.chunk_text(text)
		self.vectordb.add_document(
			doc_id=doc_id, 
			text=text, 
			table_name=table_name
		)

		# Step 2: Perform triplet extraction & build out graph in graphdb.
		required_keys = ["subject", "relation", "object", "type"]
		for chunk in chunked_text:
			subtext = text[chunk["text_idx"]: chunk["text_idx"] + chunk["text_len"]]
			triplets = self.llm.extract_triplets(subtext)

			print(subtext)
			print(json.dumps(triplets, indent=4))
			exit()

			# Save triplets to graphdb.
			for triplet in triplets:
				self.graphdb.add_triplet()

		pass


	def query(self, query: str, table_name: str, top_k: int = 5) -> str:
		# Step 1: Search vectordb for text chunks.
		results = self.vectordb.query(
			query, table_name=table_name, top_k=top_k, metric=self.metric
		)

		# Step 2: Extract entities from the question to query the 
		# graph.
		entities = []
		for result in results:
			pass

		# Step 3: Pull all relationships from the graph.

		# Step 4: Prompt the LLM for the final output.
		return ""