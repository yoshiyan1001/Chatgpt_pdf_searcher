import os
 
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import faiss
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import openai

os.environ["OPENAI_API_KEY"] = "MY_API_KEY"
reader = PdfReader('/Users/yoshi/Documents/Leetcode/data_essay.pdf')

raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text+= text

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()
docsearch = faiss.FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(ChatOpenAI(), chain_type="stuff")

query = "On which datasets does GPT-3 struggle??"
docs = docsearch.similarity_search(query, top_k=5)
answer = chain.invoke({"input_documents":docs, "question":query})
print(answer)
