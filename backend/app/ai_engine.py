import json
import os

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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

        print("Loading FLAN-T5 model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-small"
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small"
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

Use the following medical information to answer the question.

Medical information:
{context}

Question:
{query}

Give a concise, clear answer.
Do not diagnose the user.
If the information is insufficient, recommend consulting a qualified healthcare professional.

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150
            )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return {
            "question": query,
            "answer": answer,
            "sources": results
        }
