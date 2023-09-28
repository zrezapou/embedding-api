from fastapi import FastAPI, HTTPException
import tensorflow_hub as hub
from contextlib import asynccontextmanager
import os
from loguru import logger
from schema import BulkEmbeddingsRequest, SimilarityRequest
from modules import EmbeddingsService


os.environ['TFHUB_CACHE_DIR'] = f'{os.path.dirname(os.path.abspath(__file__))}/tf_cache'
embed = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> asynccontextmanager:
    global embed
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    yield
    embed = None


def create_app() -> FastAPI:
    app: FastAPI = FastAPI(lifespan=lifespan)

    @app.get("/embeddings")
    async def get_embedding(sentence: str):
        if not sentence:
            raise HTTPException(status_code=400, detail="sentence cannot be empty")
        try:
            embeddings = EmbeddingsService.embed_sentence(sentence, embed)
            return {"embedding": embeddings}
        except Exception as e:
            logger.error(f"error happened in embed_sentence: {str(e)}")
            raise HTTPException(status_code=500, detail="internal happened when transforming sentence")

    @app.post("/embeddings/bulk")
    async def get_bulk_embeddings(request: BulkEmbeddingsRequest):
        if not request.sentences:
            raise HTTPException(status_code=400, detail="at least one sentence should be provided")
        try:
            embeddings = EmbeddingsService.embed_bulk_sentences(request.sentences, embed)
            return {"embeddings": embeddings}
        except Exception as e:
            logger.error(f"error happened in calculating bulk embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail="internal happened when calculating bulk embeddings")

    @app.post("/embeddings/similarity")
    async def get_similarity(request: SimilarityRequest):
        if not request.sentence_1:
            raise HTTPException(status_code=400, detail="sentence_1 cannot be empty")
        if not request.sentence_2:
            raise HTTPException(status_code=400, detail="sentence_2 cannot be empty")
        try:
            similarity = EmbeddingsService.calculate_similarity(request.sentence_1, request.sentence_2, embed)
            return {"similarity": similarity}
        except Exception as e:
            logger.error(f"error happened in calculating cosine similarity: {str(e)}")
            raise HTTPException(status_code=500, detail="internal happened when calculating cosine similarity")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
