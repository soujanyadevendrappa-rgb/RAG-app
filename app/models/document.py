class Document:
    def __init__(self, id: str, title: str, content: str, metadata: dict = None):
        self.id = id
        self.title = title
        self.content = content
        self.metadata = metadata if metadata is not None else {}