import torch
from rag.config import (MODEL,
                MAX_SEQ,
                TEMP,
                MAX_TOKENS,
                SYSTEM_PROMPT
                )
class Generator:
    def __init__(self):
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = MAX_SEQ,
        load_in_4bit = True
        )
        FastLanguageModel.for_inference(self.model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def count_tokens(self, text):
        return len(
            self.tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]
        )

    def select_parent(self, results, max_token_set = 1100):
        selected = []
        curr_token_set = 0
        for result in results:
            doc_tokens = self.count_tokens(
                result["doc"]["page_content"]
            )
            if curr_token_set + doc_tokens > max_token_set:
                continue
            curr_token_set += doc_tokens
            selected.append(result)
        return selected

    def build_prompt(self, query, context, history):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":(
                    f"""
### Conversation History:
{history}
### Document Context
{context}
### User Question
{query}
Answer based only on the provided document context.
"""
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    
    def build_query_prompt(self, query, history):
        message = [
            {
                "role": "system",
                "content": """
You are a query rewriting assistant.

Your task is to rewrite the user's latest question into a complete, standalone question using only the conversation history.

Rules:
- Preserve the original meaning.
- Resolve references such as "it", "they", "this", "those", "these", and similar pronouns.
- Do not answer the question.
- Do not introduce information that is not present in the conversation history.
- If the current question is already complete and unambiguous, return it unchanged.
- Return only the rewritten question.
"""
            },
            {
                "role": "user",
                "content":f"""
###Conversation History:
{history}

###Current Question:
{query}
"""
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize = False,
            add_generation_prompt = True
        )
        return prompt

    def generate(self, prompt, DO_SAMPLE = True):
        inputs = self.tokenizer(
            prompt,
            truncation = True,
            max_length = MAX_SEQ,
            return_tensors = "pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = MAX_TOKENS,
                do_sample = DO_SAMPLE,
                temperature = TEMP,
                eos_token_id = self.tokenizer.eos_token_id,
                use_cache = True,
                )
        output = self.tokenizer.decode(
          outputs[0][inputs["input_ids"].shape[-1]:],
          skip_special_tokens = True
        )
        return output

    def generate_summary(self, parents, batch = 10):
        chunk_summary = []
        for i in range(0, len(parents), batch):
          batch_doc = parents[i:i+batch]
          combine_text = "\n".join(doc.page_content for doc in batch_doc)
          messages = [
        {
            "role": "user",
            "content": f"Summarize this document in 150 words:\n\n{combine_text}"
        }
    ]
          prompt = self.tokenizer.apply_chat_template(
              messages,
              tokenize = False,
              add_generation_prompt = True
          )
          generated = self.generate(prompt)
          chunk_summary.append(generated)
        combine = "\n".join(chunk_summary)
        messages = [
        {
            "role": "user",
            "content": f"Summarize this document in 150 words:\n\n{combine}"
        }
    ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )
        generated = self.generate(prompt)
        return generated

    def rewrite_query(self, query, history):
        if not history: return query
        prompt = self.build_query_prompt(query, history)
        output = self.generate(prompt, DO_SAMPLE=False)
        return output