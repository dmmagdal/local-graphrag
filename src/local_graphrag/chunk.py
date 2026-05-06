# chunk.py

from local_vectors.embedders import vector_preprocessing


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str):
        return vector_preprocessing(
            text, 
            self.chunk_size, 
            overlap=self.chunk_overlap
        )