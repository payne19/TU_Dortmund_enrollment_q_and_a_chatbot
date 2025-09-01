import streamlit as st
import os
from google import genai
from google.genai import types
import base64
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
import warnings
import re
import json

warnings.filterwarnings("ignore")

with open('config.json', 'r') as f:
    config = json.load(f)

with open('prompt_summary.txt', 'r') as f:
    prompt_guidelines = f.read()

os.environ["GOOGLE_API_KEY"] = config.get("api_key", "")

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

embedding_model = config.get("embedding_model", "models/gemini-embedding-001")
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
    )

db_directory = config.get("db_directory", "chroma_langchain_db")

if 'vectorstore' not in st.session_state:
    embeddings = st.session_state['embeddings']
    st.session_state['vectorstore'] = Chroma(
        persist_directory=db_directory,
        embedding_function=embeddings
    )

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_path = config.get("image_path", "background.jpg")
bg_ext = config.get("image_path", "background.jpg").split(".")[-1]
bg_base64 = get_base64_of_bin_file(bg_path)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/{bg_ext};base64,{bg_base64}");
    background-size: 150px 150px;
    background-repeat: repeat;
    background-position: top left;
    position: relative;
}}

/* Glass overlay */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
    background: rgba(255, 255, 255, 0.1); /* transparent overlay */
    backdrop-filter: blur(8px);           /* <-- glass blur */
    -webkit-backdrop-filter: blur(8px);   /* for Safari */
    z-index: 0;
}}

[data-testid="stAppViewContainer"] * {{
    position: relative;
    z-index: 1;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

prompt_for_summarizing = prompt_guidelines

def summary_generator(text, history, prompt_for_summarizing=prompt_for_summarizing):
    client = genai.Client(
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    model = config.get("summary_generate_model", "gemini-2.5-flash-lite")
    
    full_prompt = f"{prompt_for_summarizing}\n\n{history}\n\nCurrent Query and Retrieved Information:\n{text}"
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=full_prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=config.get("temperature", 0.25))

    summary_data = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,)
    return summary_data

def vector_db_generator(prompt):
    docs = st.session_state['vectorstore'].similarity_search(prompt, k=3)
    docs = [i.page_content for i in docs]
    summarized_info = str({'question': prompt, 'answers': docs})
    
    # Get the last 3 conversations from session state history
    history_context = ""
    if st.session_state.history:
        recent_history = st.session_state.history[-3:]  # Get last 3 conversations
        history_items = []
        for item in recent_history:
            history_items.append(f"User: {item['question']}\nAssistant: {item['answer']}")
        history_context = "Previous Conversation:\n" + "\n\n".join(history_items)
    
    summary = summary_generator(summarized_info, history_context)
    cleaned_response = re.sub(r'<[^>]+>', '', summary.text)
    return cleaned_response

st.set_page_config(page_title="TU Dortmund Q&A Chat", page_icon=":robot:", layout="wide")

with st.container(border=5):
    st.markdown(
        """
        <div style="
            background: rgba(255,255,255,0.7);
            border-radius: 10px;
            padding: 15px;
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
        ">
            <h1 style="color: black; margin: 0;">TU Dortmund Q&A Chat</h1>
            <p style="color: black; margin-top: 10px;">
                Welcome to the TU Dortmund Chat! Ask me anything about TU Dortmund University enrollment
                and other related topics, and I'll do my best to assist you.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.7);
                    color: black;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px 0;
                    backdrop-filter: blur(4px);
                    -webkit-backdrop-filter: blur(4px);
                ">
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.7);
                    color: black;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px 0;
                    backdrop-filter: blur(4px);
                    -webkit-backdrop-filter: blur(4px);
                ">
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.7);
                    color: black;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px 0;
                    backdrop-filter: blur(4px);
                    -webkit-backdrop-filter: blur(4px);
                ">
                    {prompt}
                </div>
                """,
                unsafe_allow_html=True
            )

    with st.chat_message("assistant"):
        stream = vector_db_generator(prompt)
        st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.7);
                    color: black;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px 0;
                    backdrop-filter: blur(4px);
                    -webkit-backdrop-filter: blur(4px);
                ">
                    {stream}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.session_state.messages.append({"role": "assistant", "content": stream})
    st.session_state.history.append({"question": prompt, "answer": stream})

reset_container = st.container()
with reset_container:
    left_spacer, right_column = st.columns([6, 2])
    with right_column:
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()