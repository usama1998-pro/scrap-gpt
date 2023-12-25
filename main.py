from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
import streamlit as st
import os

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GPT_MODEL = "gpt-3.5-turbo"
BASE_URL = "https://api.openai.com/v1"

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=GPT_MODEL, openai_api_base=BASE_URL)

embed = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT = """You are an intelligent AI assistant that helps user question their answer. Your name is 'Scrappy Bot'. 
    Remember the following information. do not make up things. Answer any random question a user ask. Do not answer out of context.
    if you don't know the answer just say i dont have the information.

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    """

prompt = PromptTemplate.from_template(PROMPT)

st.header(":robot_face: SCRAP-GPT")
URL = st.text_input("Enter URL")


if URL or "http://" not in URL:
    try:
        loader = WebBaseLoader([URL])
        document = loader.load()

        st.write("Title: " + document[0].metadata['title'])

        st.write("Splitting text...")
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0, length_function=len)
        docs = text_splitter.split_documents(document)

        st.write("Converting to vectors...")

        vecdb = Chroma.from_documents(docs, embed)

        user_question = st.text_input("Ask Anything")

        if user_question:
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(context=vecdb.similarity_search(user_question, k=3)[0], question=user_question)
            st.write(response)

    except Exception:
        st.write("There was a problem try again!")

else:
    st.write("You need to provide URL!")
