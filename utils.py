import os, json
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever 
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.7)
memory = ConversationBufferWindowMemory(k=3, return_messages=True)

@st.cache_resource(show_spinner=False)
def get_vectorstore_from_url(url):      # on pressing submit, this function is called
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    doc_title = document[0].metadata['title']

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks 
    vector_store = Chroma.from_documents(
        document_chunks, 
        embedding=OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
        # persist_directory='embeddings'
    )
    return vector_store, doc_title


@st.cache_data(show_spinner=False)
def get_response(query:str, _vectorstore:Chroma=None):
    '''Generates a response to User query'''
    # llm = GoogleGenerativeAI(google_api_key=os.getenv('GOOGLE_API_KEY'), model='gemini-pro', temperature=0.7)
    llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.7)

    # initialize the vector store object
    retriever = _vectorstore.as_retriever(
        # search_type="similarity_score_threshold",       
        # search_kwargs={'score_threshold':0.8, 'k':5}
    )
    
    doc_retrieval_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # create a runnable that, when invoked, retrieves List[Docs] based on user_input and chat_history
    doc_retriever_runnable = create_history_aware_retriever(llm, retriever, doc_retrieval_prompt)

    elicit_response_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the human's questions based on the given context ONLY. But if you cannot find an answer based on the context, you should either request for additional context or, if it is a question, simply say - 'I have no idea.':\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    # create a runnable that, when invoked, appends retrieved List[Docs] to prompt and passes it on to the LLM as context for generating response to user_input
    context_to_response_runnable = create_stuff_documents_chain(llm, elicit_response_prompt)

    # chains up two runnables to yield the final output that would include user_input, chat_history, context and answer
    retrieval_chain_runnable =  create_retrieval_chain(doc_retriever_runnable, context_to_response_runnable)     # chains up two runnables to yield the final output that would include human_query, chat_history, context and answer

    response = retrieval_chain_runnable.invoke({
        "chat_history": memory.load_memory_variables({})['history'],
        "input": query
    })
    # since the memory is a buffer window, we append to the buffer the query and answer of the current conversation
    memory.save_context({"input": f"{response['input']}"}, {"output": f"{response['answer']}"})

    return response['answer']


def save_conversation_history():
    chat_history = {
        'messages': st.session_state.get('messages', [{"role":"assistant", "content":"Hello, there.\n How may I assist you?"}])
    }
    with open('chat_history.json', 'w') as f:
        json.dump(chat_history, f)


def load_conversation_history():
    try:
        with open('chat_history.json', 'r') as f:
            chat_history = json.load(f)
            st.session_state.messages = chat_history.get('messages', [{"role":"assistant", "content":"Hello, there.\n How may I assist you?"}])
    except FileNotFoundError:
        st.session_state.messages = [{"role":"assistant", "content":"Hello, there.\n How may I assist you?"}]