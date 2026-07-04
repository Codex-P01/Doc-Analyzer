import torch
from unsloth import FastLanguageModel
from rag.config import (MODEL,
                MAX_SEQ,
                TEMP,
                MAX_TOKENS,
                SYSTEM_PROMPT
                )
class Generator:
    def __init__(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = MAX_SEQ,
        load_in_4bit = True
        )
        FastLanguageModel.for_inference(self.model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_prompt(self, query, context):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":(
                    f"""
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
