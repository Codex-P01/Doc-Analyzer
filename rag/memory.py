class ChatMemory:
    def __init__(self, maxHistory = 5):
        self.history = []
        self.maxHistory = maxHistory

    def add(self, query, answer):
        self.history.append({
            "User": query,
            "Assistant": answer
        })
        if len(self.history) > self.maxHistory:
            self.history.pop(0)

    def formatHistory(self):
        lines = []
        for item in self.history:
            lines.append(f"User: {item['user']}")
            lines.append(f"Assistant: {item['assistant']}")
            lines.append("")
        return "\n".join(lines)

    def getHistory(self):
        if not self.history:
            return "No previous conversation."
        return self.formatHistory()

    def clear(self):
        self.history.clear()
        