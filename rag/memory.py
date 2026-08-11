class ChatMemory:
    def __init__(self, tokenizer, maxHistory = 5, maxHistoryTokens=300):
        self.history = []
        self.maxHistory = maxHistory
        self.maxHistoryTokens = maxHistoryTokens
        self.tokenizer = tokenizer

    def count_tokens(self, text):
        return len(
            self.tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]
        )

    def add(self, query, answer):
        self.history.append({
            "User": query,
            "Assistant": answer
        })
        if len(self.history) > self.maxHistory:
            self.history.pop(0)

    def getHistory(self):
        if not self.history:
            return "No previous conversation."
        selected = []
        current_tokens = 0
        for item in reversed(self.history):
            turn_text = (
                f"User: {item['User']}\n"
                f"Assistant: {item['Assistant']}\n"
            )
            turn_tokens = self.count_tokens(turn_text)
            if current_tokens + turn_tokens > self.maxHistoryTokens:
                break
            selected.append(turn_text)
            current_tokens += turn_tokens
            if len(selected) >= self.maxHistory:
                break
        selected.reverse()
        return "\n".join(selected)

    def clear(self):
        self.history.clear()
        