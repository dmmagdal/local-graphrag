# quickstart.py


import json
import os
import random
import shutil

import datasets
from tqdm import tqdm
import pyarrow as pa

from local_vectors import detect_device
from local_graphrag import GraphRAG


SEED = 1234
random.seed(SEED)

def main():
	# Load the dataset.
	target_dataset = "illuin-conteb/narrative-qa"
	cache_dir = f"./{target_dataset.replace('/', '_')}_cache"
	save_dir = f"./{target_dataset.replace('/', '_')}"
	subsets = ["documents", "queries"]

	# Download the dataset if it's not already available.
	if not os.path.exists(save_dir) or len(os.listdir(save_dir)) == 0:
		for subset in subsets:
			data = datasets.load_dataset(
				target_dataset,
				subset,
				cache_dir=cache_dir,
			)
			data.save_to_disk(os.path.join(save_dir, subset))

		# Clear the cache directory.
		shutil.rmtree(cache_dir)

	# Load the documents and queries.
	documents = datasets.load_from_disk(os.path.join(save_dir, "documents"))
	queries = datasets.load_from_disk(os.path.join(save_dir, "queries"))

	# Load the configuration information.
	with open("config.json", "r") as f:
		config = json.load(f)['graphrag']

	# Unpack and organized theh configuration data for each component 
	# of the graphrag.
	vector_config = config["vector"]
	graph_config = config["graph"]
	llm_config = config["llm"]

	# Clear any existing tables or databases.
	storage_artifacts = [
		vector_config["vector_db"],
		graph_config["graph_db"],
		"./graph"
	]
	for artifact in storage_artifacts:
		if os.path.exists(artifact):
			if os.path.isdir(artifact):
				shutil.rmtree(artifact)
			else:
				os.remove(artifact)

	# Detect GPU accelerators.
	device = detect_device(force_cpu=True)

	# Initialize graphrag with the configuration.
	graphrag = GraphRAG(
		embed_model_id=vector_config["model_id"],
		vector_db_path=vector_config["vector_db"],
		graph_db_path=graph_config["graph_db"],
		llm_model=llm_config["model_id"],
		token_overlap=vector_config["token_overlap"],
		batch_size=vector_config["batch_size"],
		device=device,
		use_binary=vector_config["use_binary"],
		query_metric=vector_config["metric"],
		model_save_root=vector_config["model_save_root"],
		host=llm_config["host"],  
	)

	# Define schema (this is heavily dependent upon the datasets) and
	# pass that to the graphrag so that the vectordb can build the 
	# table.
	schema = pa.schema([
		pa.field("id", pa.string()),
		pa.field("subtext", pa.string()),
		pa.field("text_len", pa.int32()),
		pa.field("text_idx", pa.int32()),
		pa.field("vector", pa.list_(
			pa.uint8() if vector_config["use_binary"] else pa.float32(), 
			graphrag.get_dims()
		)),
	])
	graphrag.build_vector_table(
		table_name=vector_config["table_name"],
		schema=schema
	)

	# Ingest and index the documents to the graphrag.
	for split_name, data in documents.items():
		for doc in tqdm(data, desc=f"Ingesting {split_name} split into Graph RAG"):
			graphrag.ingest(
				text=doc["chunk"], 
				doc_id=doc["chunk_id"],
				table_name=vector_config["table_name"]
			)

	# Perform a query on the graphrag.
	sampled_queries = queries.shuffle(seed=SEED).select(range(5))
	for query in sampled_queries:
		question, chunk_id, answer = query["og_query"], query["chunk_id"], query["answer"]
		graphrag_answer = graphrag.query(question)

		# Output results.
		print(f"Question: {question}")
		print(f"Expected answer: {answer} (chunk id {chunk_id})")
		print(f"Generated answer: {graphrag_answer}")
		print("-" * 72)


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()