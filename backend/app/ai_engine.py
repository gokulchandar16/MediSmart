import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "medical_knowledge.json"
)


class MediSmartAI:
    def __init__(self):
        print("Loading MiniLM model...")

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Loading medical knowledge...")

        with open(DATA_PATH, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

        self.texts = [
            document["content"]
            for document in self.documents
        ]

        print("Creating embeddings...")

        embeddings = self.embedding_model.encode(
            self.texts,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype("float32")

        self.index = faiss.IndexFlatL2(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

        print("Loading FLAN-T5...")

        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-small"
        )

        print("MediSmart AI ready!")


    def search(self, query, top_k=3):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for index in indices[0]:

            if index < len(self.documents):
                results.append(
                    self.documents[index]
                )

        return results


    def answer(self, query):

        results = self.search(query)

        context = "\n".join(
            result["content"]
            for result in results
        )

        prompt = f"""
You are MediSmart, an educational medical information assistant.

Use only the provided context to answer the user's question.

Context:
{context}

Question:
{query}

Provide a concise and easy-to-understand answer.
Do not diagnose the user.
If the information is insufficient, say that the user should consult a qualified healthcare professional.

Answer:
"""

        response = self.generator(
            prompt,
            max_new_tokens=150,
            do_sample=False
        )

        return {
            "question": query,
            "answer": response[0]["generated_text"],
            "sources": results
        }
