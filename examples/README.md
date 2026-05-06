


### Notes

 - Several datasets were considered for testing GraphRAG at scale. 
     - [hotspotqa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
         - Decent sized dataset (~200K entries across all subsets).
         - Provides question, answer, and context as the key features for each entry.
             - Additional metadata is there too.
         - Cons: 
             - Doesn't allow for a general example of indexing a dataset at scale (it makes sense to index the context and query graphrag for the answer, vs index a whole knowledge base and query graphrag).
     - [MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue)
         - Good sized dataset (~22K entries across all subsets).
         - Provides paragraphs, question, and answer as the key features for each entry.
             - Additional metadata is there too.
         - Cons: 
             - Doesn't allow for a general example of indexing a dataset at scale (it makes sense to index the context and query graphrag for the answer, vs index a whole knowledge base and query graphrag).
     - [PubMedQA](https://huggingface.co/datasets/llamafactory/PubMedQA)
         - Good sized dataset (~11K entries across all subsets).
         - Provides instruction (which includes context), input (the question), and output as the key features for each entry.
             - Additional metadata is there too.
         - Cons:
             - Doesn't allow for a general example of indexing a dataset at scale (it makes sense to index the context and query graphrag for the answer, vs index a whole knowledge base and query graphrag).
     - [GraphRAG-Bench](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench)
         - Good sized dataset (~4K entries across all subsets).
         - Provides question, answer, and evidence as the key features for each entry.
             - Additional metadata is there too.
         - Provides a variety of question types (ie fact retrieval, complex reasoning, contextual summarize, etc).
         - Cons:
             - Doesn't allow for a general example of indexing a dataset at scale (it makes sense to index the context and query graphrag for the answer, vs index a whole knowledge base and query graphrag).
     - [Narrative-QA](https://huggingface.co/datasets/illuin-conteb/narrative-qa)
         - Good sized dataset (~8k rows for documents subset and 39k rows for the queries subset).
         - Provides chunk id and chunk as the key features for each document entry.
             - Provides the query, chunk id, and answer for each query entry.
             - Additional metadata is there too for the queries subset.
         - Cons:
             - Will have to account for storage requirements for the data.