from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import CONFIG


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag_chain(retriever, llm, cfg=None):
    cfg = cfg or CONFIG
    prompt = ChatPromptTemplate.from_template(cfg["prompt_template"])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG 체인 구성 완료")
    return rag_chain
