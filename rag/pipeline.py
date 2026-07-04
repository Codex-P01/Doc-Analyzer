from rag.indexing import Indexer
from rag.retrieval import Retriever
from rag.generation import Generator

class RAGPipeline:
    def __init__(self):
        self.indexer = Indexer()
        self.generator = Generator()
        self.indexData = None
        self.retriever = None

    def index_pdf(self, pdf_path):
        self.indexData = self.indexer.build_index(pdf_path)
        self.retriever = Retriever(self.indexData)

    def query(self, question):
        if self.retriever is None:
            raise RuntimeError("No document has been indexed")
        results = self.retriever.retrieve_multi_query(question)
        context = ""
        for result in results:
            context += f"""
Page {result["page_label"]}
{result["doc"]["page_content"]}
"""
        prompt = self.generator.build_prompt(question, context)
        answer = self.generator.generate(prompt, False)
        return answer

    def summarize(self):
        if self.indexData is None:
            raise RuntimeError("No document has been indexed. ")
        return self.generator.generate_summary(self.indexData.parents)
