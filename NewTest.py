import streamlit as st
from llama_index.core import Settings
import logging
import sys
import os.path

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.core.service_context import set_global_service_context

# from llama_index.llms.llama_cpp import LlamaCPP
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
# from langchain.llms import HuggingFaceHub
from llama_index.core.prompts.chat_prompts import ChatPromptTemplate, ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine, ContextChatEngine, SimpleChatEngine
from llama_index.core.settings import llm_from_settings_or_context

# from llama_index.legacy.prompts import ChatPromptTemplate
# from llama_index.core.base.llms.types import ChatMessage, MessageRole
try:
    from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader, StorageContext, \
        load_index_from_storage
except ImportError:
    from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader, StorageContext, \
        load_index_from_storage

from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)
from huggingface_hub import login

login("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")

from transformers import AutoTokenizer

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

st.title("I2E Chatbot")
# st.info(
#     "Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)",
#     icon="📃")

from llama_index.core import PromptTemplate

prompt_template = """### System: Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Only return the helpful answer below and nothing else.
Helpful answer:
"""

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to I2E Assist, How can I help you!!"}
    ]

PERSIST_DIR = "./Folder"


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading the chatbot. Please wait for a while!!"):
        # reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        # docs = reader.load_data()
        llm = HuggingFaceInferenceAPI(
            generate_kwargs={"temperature": 0.0},
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        )
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embed_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        service_context = ServiceContext.from_defaults(
            chunk_size=1000,
            chunk_overlap=100,
            embed_model=embed_model,
            llm=llm
        )
        set_global_service_context(service_context)
        if not os.path.exists(PERSIST_DIR):
            reader = SimpleDirectoryReader(input_dir="./Sample", recursive=True)
            Settings.llm = llm
            Settings.embed_model = embed_model
            docs = reader.load_data()
            index = VectorStoreIndex.from_documents(documents=docs, service_context=service_context)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            # st.write("LoadEmbedding>>>", index)
            return index
        else:
            # load the existing index
            Settings.llm = llm
            Settings.embed_model = embed_model
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            # st.write("StoredEmbedding>>>", index)
            return index


index = load_data()

# def generate_text(prompt):
question = ("Hi, how are you!")

qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given only the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)

refine_prompt_str = (
    "We have the opportunity to refine the original answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question: {query_str}. "
    "If the context isn't useful, output the original answer again.\n"
    "Original Answer: {existing_answer}"
)

chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            prompt_template
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            qa_prompt_str
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

# Refine Prompt
chat_refine_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "If the context isn't helpful, just say I don't know. Don't any add informtion into the answer that is not available in the context"
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "New Context: {context_msg}\n"
            "Query: {query_str}\n"
            "Original Answer: {existing_answer}\n"
            "New Answer: "
        ),
    ),
]
refine_template = ChatPromptTemplate(chat_refine_msgs)

custom_prompt = PromptTemplate(
    """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)

# list of `ChatMessage` objects
custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="Hello assistant, we are having an insightful discussion about the given content and you are helping me understand the content by answering or summerizing and explaining me the content without changing its true meaning.",
    ),
    ChatMessage(role=MessageRole.ASSISTANT, content="Okay, sounds good."),
]


if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine

    query_engine = index.as_query_engine()
    from llama_index.core.memory import ChatMemoryBuffer

    # memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    #
    # retriever = index.as_retriever()
    #
    # chat_engine = ContextChatEngine.from_defaults(
    #     retriever=retriever,
    #     memory=memory,
    #     system_prompt=prompt_template,
    #     llm=Settings.llm,
    #     verbose=True
    #
    # )
    #
    # st.session_state.chat_engine = chat_engine

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    print("question--------------------->", prompt)

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print(st.session_state.messages)
            print("Step 1")
            # response = st.session_state.chat_engine.chat(prompt)
            query_engine = index.as_query_engine()
            print("Step 2")
            # prompt = "\n".join([f"{item['role']}: {item['content']}" for item in st.session_state.messages[-5:]])
            res = query_engine.query(prompt)
            st.write(res.response)
            message = {"role": "assistant", "content": res.response}
            st.session_state.messages.append(message)  # Add response to message history
