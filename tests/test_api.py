import unittest
from modules import EmbeddingsService
import tensorflow_hub as hub
import os
from main import app
from starlette.testclient import TestClient
from unittest.mock import patch

os.environ['TFHUB_CACHE_DIR'] = f'{os.path.dirname(os.path.abspath(__file__))}/../src/tf_cache'
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


class TestEmbeddingsService(unittest.TestCase):

    def test_embed_sentence(self):
        embeddings = EmbeddingsService.embed_sentence("The quick brown fox", embed)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 512

    def test_embed_bulk_sentences(self):
        sentences = ["The quick brown fox", "The lazy dog"]
        embeddings = EmbeddingsService.embed_bulk_sentences(sentences, embed)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 512
        assert len(embeddings[1]) == 512

    def test_calculate_similarity(self):
        similarity = EmbeddingsService.calculate_similarity("The quick brown fox", "The lazy dog", embed)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0


class TestAPIMethods(unittest.TestCase):

    def test_get_embedding(self):
        with TestClient(app) as client:
            response = client.get("/embeddings?sentence=The+quick+brown+fox")
            assert response.status_code == 200
            data = response.json()
            assert "embedding" in data
            assert isinstance(data["embedding"], list)

    def test_get_embedding_empty_sentence(self):
        with TestClient(app) as client:
            response = client.get("/embeddings?sentence=")
            assert response.status_code == 400

    @patch('modules.EmbeddingsService.embed_sentence', side_effect=Exception("Custom exception message"))
    def test_get_embedding_internal_error(self, mock_embed_sentence):
        with TestClient(app) as client:
            response = client.get("/embeddings?sentence=The+quick+brown+fox")
            assert response.status_code == 500

    def test_get_bulk_embeddings(self):
        with TestClient(app) as client:
            request_data = {"sentences": ["The quick brown fox", "The lazy dog"]}
            response = client.post("/embeddings/bulk", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "embeddings" in data
            assert isinstance(data["embeddings"], list)

    def test_get_bulk_embeddings_empty_list(self):
        with TestClient(app) as client:
            request_data = {"sentences": []}
            response = client.post("/embeddings/bulk", json=request_data)
            assert response.status_code == 400

    @patch('modules.EmbeddingsService.embed_bulk_sentences', side_effect=Exception("Custom exception message"))
    def test_get_bulk_embeddings_internal_error(self, mock_embed_bulk_sentences):
        with TestClient(app) as client:
            request_data = {"sentences": ["The quick brown fox", "The lazy dog"]}
            response = client.post("/embeddings/bulk", json=request_data)
            assert response.status_code == 500

    def test_get_similarity(self):
        with TestClient(app) as client:
            request_data = {"sentence_1": "The quick brown fox", "sentence_2": "The lazy dog"}
            response = client.post("/embeddings/similarity", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert "similarity" in data
            assert isinstance(data["similarity"], float)

    def test_get_similarity_sentence1_empty(self):
        with TestClient(app) as client:
            request_data = {"sentence_1": "", "sentence_2": "The lazy dog"}
            response = client.post("/embeddings/similarity", json=request_data)
            assert response.status_code == 400

    def test_get_similarity_sentence2_empty(self):
        with TestClient(app) as client:
            request_data = {"sentence_1": "The lazy dog", "sentence_2": ""}
            response = client.post("/embeddings/similarity", json=request_data)
            assert response.status_code == 400

    @patch('modules.EmbeddingsService.embed_sentence', side_effect=Exception("Custom exception message"))
    def test_get_similarity_internal_error(self, mock_embed_sentence):
        with TestClient(app) as client:
            request_data = {"sentence_1": "The quick brown fox", "sentence_2": "The lazy dog"}
            response = client.post("/embeddings/similarity", json=request_data)
            assert response.status_code == 500
