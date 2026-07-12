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

    def getHistory(self):
        return self.history

    def clear(self):
        self.history.clear()
        