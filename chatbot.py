# database creation
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import openai
import requests

# load_dotenv()

OPENAI_API_KEY= st.secrets['OPENAI_API_KEY']
PINE_API_KEY =  st.secrets['PINE_API_KEY']

# OPENAI_API_KEY=  os.getenv('OPENAI_API_KEY')
# PINE_API_KEY =  os.getenv('PINE_API_KEY')

pc = Pinecone(api_key= PINE_API_KEY)


# #-------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------

indicies = pc.list_indexes()

all_indicies_list = []

for index in indicies:
    all_indicies_list.append(index['name'])

# # -------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------------------

# #pdf index

if 'nvli-index-ebooks' not in all_indicies_list:
    pc.create_index(
    name="nvli-index-ebooks",
    dimension=1536, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-west-2"
        )   
    )
    pinecone_ebooks = pc.Index('nvli-index-ebooks')

    # response = requests.get('https://wzcc.nvli.in/system/files/nvli_pdfs/02.Indian%20seals%20Problem%20and%20Prospects.pdf')
    # data =  response.json()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000, chunk_overlap = 250)
    embeddings = OpenAIEmbeddings(api_key= OPENAI_API_KEY)

    # for ebook in data['rows']['search_results']:
    #     ebook_url = 'https://wzcc.nvli.in'+ebook['field_pdf_digital_file']

    # if response.status_code != 404:
    loader = PyMuPDFLoader('/home/janpath/Desktop/chatbot/nvli_project/chatbot/02.Indian seals Problem and Prospects (2) (1).pdf')
    pdf = loader.load()
    documents = text_splitter.split_documents(pdf)

    pinecone_ebooks = PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name = 'nvli-index-ebooks'
    )       
else:
    text = 'none'
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    pinecone_ebooks =  PineconeVectorStore.from_texts(
        text,
        embeddings,
        index_name = 'nvli-index-ebooks'
    )


#-------------------------------------------------------------------------------------------------------------------
# frontend

import streamlit as st
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import openai


parser = StrOutputParser()
model = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model= 'gpt-3.5-turbo') 

template = """
    Answer the question based on the context below. If you can't
    answer the question, reply No Books Available

    Context: {context}

    Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
    
chain_ebooks = (
    {'context': pinecone_ebooks.as_retriever(), 'question' : RunnablePassthrough()}
    | prompt
    | model
    | parser
)

def handle_query(user_input, word_limit):
    query = f"""You are a Indian Culture Portal chatbot. You are designed to be as helpful as possible while providing only factual information. 
                .Context information is below. Given the context information and not prior knowledge, answer the query.

                When the user enters search for pdfs/ebooks/provide information/data you should search the entire ebook index thorougly (do not miss details) and return the relevant information
                alongwith the title of book (that is written in the front of the book) and the page number. 

                Do not give any conclusions.
                
            an    The name of book and page number should be written at the end.

                Answers should be give in {word_limit} words.
                {user_input}
            """

    query_vec = openai.embeddings.create(
    input = user_input,
    model = 'text-embedding-ada-002'
    )

    ques = query_vec.data[0].embedding
    response_ebooks = chain_ebooks.invoke(query)
    return query, response_ebooks


def summarise_query(user_input, word_limit):
    summarise_query = f"""You are a Indian Culture Portal chatbot. You are designed to be as helpful as possible while providing only factual information. 
                .Context information is below. Given the context information and not prior knowledge, answer the query.

                When the user enters search for pdfs/ebooks/provide information/data you should search the entire ebook index thorougly (do not miss details) and return the relevant information
                and summarise the relevant book 

                Do not give any conclusions.
                
                The year of publication should also be written at the end after leaving a line.

                Answers should be give in {word_limit} words.

                {user_input}
            """
    response_ebooks = chain_ebooks.invoke(summarise_query)
    return summarise_query, response_ebooks


def main():
    st.set_page_config(page_title='NVLI Chatbot', page_icon=":robot_face:")

    with st.container():
        st.title('NVLI Chatbot')
        st.caption('Welcome! Ask me anything about NVLI.')
    
    st.markdown('---')

    st.write('Get started by writing...')
    sample_queries = [
        "Give me information about Microfilms",
        "Search images about Dance of Meghalaya",
        "Search videos about culture of Orissa"
    ]

    cols = st.columns(len(sample_queries))
    for col, query in zip(cols, sample_queries):
        with col:
            if st.button(f'Use sample: {query}'):
                st.session_state['info_query'] = query
                st.session_state['summarize_query'] = ""

    word_limit = st.number_input('Enter word limit',min_value= 150, max_value=500, value=250, step=30,
                                  help='Type the required word limit here')

    info_query = st.text_input("Enter your query for information:", value=st.session_state.get('info_query', ''), help="Type your question here")    
    submit_button = st.button('Submit query')

    summarize_query = st.text_input("Enter your query for summarization:", value=st.session_state.get('summarize_query', ''), help="Type your book name here")
    summarise_button = st.button('Summarise query')


    if submit_button and info_query:
        info_query = info_query.lower()
        if 'summarise' in info_query or 'summarize' in info_query or 'important points' in info_query:
            st.markdown('Please Check the Query Again')
        else:
            query, response_ebooks = handle_query(info_query, word_limit)
            display_responses(response_ebooks)

    if summarise_button and summarize_query:
        summarize_query = summarize_query.lower()
        if 'summarise' in summarize_query or 'summarize' in summarize_query or 'important points' in summarize_query:
            query, response_ebooks = summarise_query(summarize_query, word_limit)
            st.markdown('**Summarized Information:**')
            st.markdown(response_ebooks)
        else:
            st.markdown('Please Check the Query Again')


def display_responses(response_ebooks):
    st.markdown('**Ebooks Information:**')
    st.markdown(response_ebooks)

if __name__ == "__main__":
    main()


