from langchain_core.chat_history import BaseChatMessageHistory


class InMemoryHistory(BaseChatMessageHistory):
    """Egyszerű memória alapú chat történet kezelő"""

    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = [] 