import re
import numpy as np
from rag.config import EMB_MODEL
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, indexData):
        self.emb_model = SentenceTransformer(EMB_MODEL)
        self.index = indexData.index
        self.parent_info = indexData.parent_info
        self.child_doc = indexData.child_doc
        self.parents = indexData.parents

    def emb_search(self, q, k = 12):
        q_emb = self.emb_model.encode(
            [q],
            normalize_embeddings=True
            )
        q_emb = np.array(q_emb).astype("float32")
        dis, indices = self.index.search(q_emb, k)
        parent_score = {}
        for idx, score in zip(indices[0], dis[0]):
            if idx == -1: continue
            child = self.child_doc[idx]
            parent_id = child["metadata"]["parent_id"]
            if parent_id not in parent_score:
                parent_score[parent_id] = score
            else:
                parent_score[parent_id] += score
            results = [
            {
                "doc" : self.parent_info[i],
                "score": s,
                "page_label":self.parent_info[i]["page_label"]
            }
            for i, s in parent_score.items()
            ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def extract_comparison_terms(self, query):
        q = query.strip()
        patterns = [
            r"difference between (.+)",
            r"compare (.+)",
            r"comparison between (.+)",
            r"distinguish between (.+)",
            r"(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, q, flags=re.IGNORECASE)
            if match:
                items = re.split(r",\s*|\s+and\s+|\s+vs\.?\s+|\s+versus\.?\s+", match.group(1))
                return items
        return None

    def expand_query(self, query):
        expanded = [query]
        terms = self.extract_comparison_terms(query)
        if terms:
            for term in terms:
                expanded.extend([
                    term,
                    f"{term} definition",
                    f"{term} characteristics",
                    f"{term} properties",
                    f"{term} examples",
                ])
        return list(dict.fromkeys(expanded))

    def retrieve_multi_query(self, query, k_per_query=6, final_k=6):
        merged = {}
        for expanded_query in self.expand_query(query):
            results = self.emb_search(expanded_query, k=k_per_query)
            for result in results:
                parent_key = result["page_label"] + "_" + result["doc"]["page_content"][:100]
                if parent_key not in merged:
                    merged[parent_key] = result
                else:
                    merged[parent_key]["score"] = max(
                        merged[parent_key]["score"],
                        result["score"]
                        )
        results = list(merged.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:final_k]
