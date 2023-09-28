from pydantic import BaseModel


class BulkEmbeddingsRequest(BaseModel):
    sentences: list[str]


class SimilarityRequest(BaseModel):
    sentence_1: str
    sentence_2: str
