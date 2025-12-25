from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams 
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from huggingface_hub import HfFolder

import gradio as gr

URL = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "skills-network"
MODEL_ID = "ibm/granite-3-2-8b-instruct"

def get_llm():
    parameters = {
        GenParams.TEMPERATURE:0.9,
        GenParams.MAX_NEW_TOKENS:512,
        GenParams.MIN_NEW_TOKENS:20
    }

    model = WatsonxLLM(
        model_id = MODEL_ID,
        url = URL,
        project_id = PROJECT_ID,
        params=parameters
    )

    return model

def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document


def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=100,
        length_function = len
    )

    chunks = text_splitter.split_documents(data)
    return chunks

def watsonx_embedding():
    params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS:3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS:{"input_text":True},
    }

    watsonx_embedding = WatsonxEmbeddings(
        model_id = "ibm/slate-125m-english-rtrvr-v2",
        url = URL,
        project_id = PROJECT_ID,
        params=params
    )

    return watsonx_embedding

def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

def retriver(file):
    docs = document_loaders(file)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriver(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False
    )

    response = qa.invoke(query)
    return response['result']

rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="auto",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Your Query", lines=2, placeholder="Enter your query here.")
    ],

    outputs=gr.Textbox(label="Response"),
    title="Question Answer Bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document"
)

rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)
