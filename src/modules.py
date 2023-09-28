from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingsService:

    @staticmethod
    def embed_sentence(sentence, embed):
        embeddings = embed([sentence])
        emb = embeddings.numpy().tolist()
        return emb[0]

    @staticmethod
    def embed_bulk_sentences(sentences, embed):
        embeddings = embed(sentences)
        return embeddings.numpy().tolist()

    @staticmethod
    def calculate_similarity(sentence_1, sentence_2, embed):
        embeddings_1 = EmbeddingsService.embed_sentence(sentence_1, embed)
        embeddings_2 = EmbeddingsService.embed_sentence(sentence_2, embed)
        similarity = cosine_similarity([embeddings_1], [embeddings_2])[0][0]
        return similarity
