import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
st.title('AI QNA CHATBOT WITH HISTORY')
# GROQ_API_KEY=st.text_input('Enter your Groq key',type='password')
file_to_read=st.file_uploader('UPLOAD THE PDF FILE: ',type='pdf')
# url_to_read=st.text_area('Enter the URL: ')

input_question_ask=st.text_input("Enter your Query: ")

#now load all variables
os.environ['LANGCHAIN_API_KEY']='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'
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['HF_TOKEN']='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'
os.environ['LANGCHAIN_PROJECT']='QNA CHATBOT WITH HISTORY'


HF_TOKEN='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'
GROQ_API_KEY='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'

docs=''
from langchain_text_splitters import RecursiveCharacterTextSplitter

if file_to_read is not None:
    with open('uploaded_file.pdf','wb') as file:
        file.write(file_to_read.getbuffer())
    loader_pdf=PyPDFLoader('uploaded_file.pdf')
    docs=loader_pdf.load()
# elif url_to_read is not None:
#     loader_url=WebBaseLoader(web_path=url_to_read)
#     docs=loader_url.load()
else:
    st.write('Please provide the data either PDF file')
    
    
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
split_doc=text_splitter.split_documents(documents=docs)

#NOW LOAD MODEL
from langchain_groq import ChatGroq
llm_model=ChatGroq(model='Llama3-8b-8192',api_key=GROQ_API_KEY)


#now load webpage
# loader=WebBaseLoader(web_path='https://lilianweng.github.io/posts/2023-06-23-agent/')
# docs=loader.load()


#splitting the document


#now embedding the document & creating vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
HF_Embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
from langchain_community.vectorstores import FAISS
vectorstoreFaiss=FAISS.from_documents(documents=split_doc,embedding=HF_Embeddings)
Faiss_ret=vectorstoreFaiss.as_retriever()


#prompt template
system_prompt=(
    'you are an good assistant for question-answering tasks'
    'use the following pieces of retreived context to answer'
    'the question. if you dont know the answer then say you dont know'
    'use maximum 3 sentences, keep answer concise'
    '\n\n'
    '{context}'
)

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

chat_history=[]

Base_Prompt=ChatPromptTemplate.from_messages([
    ('system',system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{input}')
])

Context_qna_msg=(
    'Given a chat history and the latest user question'
    'which might reference context from the chat history,'
    'formulate a standalone question which can be understood'
    'without th chat history. Do not answer the question'
    'just reformulate it if needed and otherwise return it as is'
    
)

context_qna_prompt=ChatPromptTemplate.from_messages([
    ('system',Context_qna_msg),
    ('human','{input}')
])

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import AIMessage,HumanMessage

History_aware_ret=create_history_aware_retriever(llm=llm_model,retriever=Faiss_ret,prompt=context_qna_prompt)
Base_question_answer_chain=create_stuff_documents_chain(llm=llm_model,prompt=Base_Prompt)
History_rag_chain=create_retrieval_chain(retriever=History_aware_ret,combine_docs_chain=Base_question_answer_chain)



def question_answer(input_question_ask):
    response=History_rag_chain.invoke({'input':input_question_ask,'chat_history':chat_history})
    st.write(response['answer'])
    chat_history.extend([
        HumanMessage(content=input_question_ask),
        AIMessage(response['answer'])
    ])




if input_question_ask!="exit" or 'quit':
    question_answer(input_question_ask)






