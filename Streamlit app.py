# from openai import OpenAI
import streamlit as st
from huggingface_hub import login
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

login("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")
from easyllm.clients import huggingface

st.title("Welcome to I2E")

# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "model" not in st.session_state:
    st.session_state["model"] = "meta-llama/Meta-Llama-3-8B-Instruct"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = huggingface.ChatCompletion.create(
            model=st.session_state["model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            temperature=0.0,
            top_p=0.6,
            max_tokens=1024,
            stop=["[/INST]", "USER:", "ADMIN:"]
        )

        response = st.write(response['choices'][0]['message']['content'])
    st.session_state.messages.append({"role": "assistant", "content": response})