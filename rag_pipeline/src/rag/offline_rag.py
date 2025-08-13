import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, text) -> str:
        if isinstance(text, str):
            return self.extract_answer(text)
        elif isinstance(text, dict):
            if "text" in text:
                return self.extract_answer(text["text"])
            elif "output" in text:
                return self.extract_answer(text["output"])
            elif "answer" in text:
                return self.extract_answer(text["answer"])
            else:
                return self.extract_answer(str(text))
        else:
            return self.extract_answer(str(text))


    def extract_answer(self, text_response: str, pattern: str = r"Answer:\s*(.*)") -> str:
        if not text_response or not isinstance(text_response, str):
            return str(text_response) if text_response is not None else "No answer generated"
        
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text if answer_text else text_response
        else:
            return text_response.strip()


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }

        rag_chain = (
            input_data | self.prompt | self.llm | self.str_parser
        )

        return rag_chain
    
    def format_docs(self, docs):
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for doc in docs:
            if hasattr(doc, 'page_content') and doc.page_content:
                formatted_docs.append(doc.page_content.strip())
        
        return "\n\n".join(formatted_docs) if formatted_docs else "No relevant documents found."
