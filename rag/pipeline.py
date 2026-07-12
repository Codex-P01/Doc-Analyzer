from rag.indexing import Indexer
from rag.retrieval import Retriever
from rag.generation import Generator
from rag.memory import ChatMemory

class RAGPipeline:
    def __init__(self):
        self.indexer = Indexer()
        self.generator = Generator()
        self.chatmemory = ChatMemory()
        self.indexData = None
        self.retriever = None

    def index_pdf(self, pdf_path):
        self.indexData = self.indexer.build_index(pdf_path)
        self.retriever = Retriever(self.indexData)

    def query(self, question):
        if self.retriever is None:
            raise RuntimeError("No document has been indexed")
        history = self.chatmemory.getHistory()
        rewrittenQuestion = self.generator.rewite_query(question, history)
        results = self.retriever.retrieve_multi_query(rewrittenQuestion)
        context = ""
        for result in results:
            context += f"""
Page {result["page_label"]}
{result["doc"]["page_content"]}
"""
        prompt = self.generator.build_prompt(question, context, history)
        answer = self.generator.generate(prompt, False)
        self.chatmemory.add(question, answer)
        return answer

    def summarize(self):
        if self.indexData is None:
            raise RuntimeError("No document has been indexed. ")
        return self.generator.generate_summary(self.indexData.parents)
