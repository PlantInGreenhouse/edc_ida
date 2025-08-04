from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy


class SchemaRetriever:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(
        self, target_schema_dict: dict, embedding_model, embedding_tokenizer, finetuned_e5mistral=False, use_openai_embedder=False
    ) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        self.target_schema_dict = target_schema_dict
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.use_openai_embedder = use_openai_embedder

        # Embed the target schema

        self.target_schema_embedding_dict = {}
        self.finetuned_e5mistral = finetuned_e5mistral

        for relation, relation_definition in target_schema_dict.items():
            if self.use_openai_embedder:
                embedding = llm_utils.get_embedding_openai(relation_definition)
            else:
                embedding = llm_utils.get_embedding_sts(
                    self.embedding_model,
                    relation_definition,
                    prompt="Instruct: Retrieve descriptions of relations that are present in the given text.\nQuery: ",
                )
            self.target_schema_embedding_dict[relation] = embedding

    def update_schema_embedding_dict(self):
        for relation, relation_definition in self.target_schema_dict.items():
            if relation in self.target_schema_embedding_dict:
                continue
            if self.finetuned_e5mistral:
                embedding = llm_utils.get_embedding_e5mistral(
                    self.embedding_model,
                    self.embedding_tokenizer,
                    relation_definition,
                )
            elif self.use_openai_embedder:
                embedding = llm_utils.get_embedding_openai(relation_definition)
            else:
                embedding = llm_utils.get_embedding_sts(
                    self.embedding_model,
                    relation_definition,
                )
            self.target_schema_embedding_dict[relation] = embedding

    def retrieve_relevant_relations(self, query_input_text: str, top_k=10):
        target_relation_list = list(self.target_schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.target_schema_embedding_dict.values())

        if self.finetuned_e5mistral:
            query_embedding = llm_utils.get_embedding_e5mistral(
                self.embedding_model,
                self.embedding_tokenizer,
                query_input_text,
                "Retrieve descriptions of relations that are present in the given text.",
            )
        elif self.use_openai_embedder:
            query_embedding = llm_utils.get_embedding_openai(query_input_text)
        else:
            query_embedding = llm_utils.get_embedding_sts(
                self.embedding_model,
                query_input_text,
                prompt="Instruct: Retrieve descriptions of relations that are present in the given text.\nQuery: ",
            )

        query_embedding = np.array(query_embedding)
        print(f"[DEBUG] query_embedding shape after np.array: {query_embedding.shape}")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            print(f"[DEBUG] query_embedding reshaped to: {query_embedding.shape}")

        target_relation_embedding_array = np.array(target_relation_embedding_list)
        print(f"[DEBUG] target_relation_embedding_array shape after np.array: {target_relation_embedding_array.shape}")
        if target_relation_embedding_array.ndim == 1:
            target_relation_embedding_array = target_relation_embedding_array.reshape(1, -1)
            print(f"[DEBUG] target_relation_embedding_array reshaped to: {target_relation_embedding_array.shape}")
        if target_relation_embedding_array.ndim == 3 and target_relation_embedding_array.shape[1] == 1:
            target_relation_embedding_array = target_relation_embedding_array.squeeze(1)
            print(f"[DEBUG] target_relation_embedding_array squeezed to: {target_relation_embedding_array.shape}")

        print(f"[DEBUG] Final query_embedding shape: {query_embedding.shape}")
        print(f"[DEBUG] Final target_relation_embedding_array shape: {target_relation_embedding_array.shape}")
        # Ensure embedding dimensions match
        if query_embedding.shape[1] != target_relation_embedding_array.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: query {query_embedding.shape}, target {target_relation_embedding_array.shape}")

        scores = query_embedding @ target_relation_embedding_array.T

        scores = scores[0]
        highest_score_indices = np.argsort(-scores)

        return [target_relation_list[idx] for idx in highest_score_indices[:top_k]]
