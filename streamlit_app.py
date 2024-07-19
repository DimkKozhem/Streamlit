import streamlit as st
from llama_cpp import Llama
import numpy as np
import requests
import os



# Кастомизация лейаута
st.set_page_config(page_title="Saiga", page_icon="🧠", layout="wide", )



# CSS стили
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#
# local_css("styles.css")


# Add elements to vertical setting menu
st.sidebar.title("Настройки пользователя")

# Add video source selection dropdown
source = st.sidebar.selectbox(
    "Модель",
    ("Saiga",'Mistral-Nemo'),
)
if source == "Saiga":
    n_ctx = 8192
    model_path = "model-q4_K.gguf"
    model_url = "https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf"
elif source == "Mistral-Nemo":
    n_ctx = 128000
    model_path = "Mistral-Nemo.gguf"
    model_url = "https://cdn-lfs-us-1.huggingface.co/repos/e4/70/e470da9d866fe2ec5de77638ec3e5b05d48a6a8c8d0ccc72570c0183fb691bb7/913722d032f33fbbe2f7374fe3c27c89f13c7cc52cf4ed2078b0297d9a6d9441?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27Mistral-Nemo-Instruct-2407-Q5_K_M.gguf%3B+filename%3D%22Mistral-Nemo-Instruct-2407-Q5_K_M.gguf%22%3B&Expires=1721681597&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyMTY4MTU5N319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zL2U0LzcwL2U0NzBkYTlkODY2ZmUyZWM1ZGU3NzYzOGVjM2U1YjA1ZDQ4YTZhOGM4ZDBjY2M3MjU3MGMwMTgzZmI2OTFiYjcvOTEzNzIyZDAzMmYzM2ZiYmUyZjczNzRmZTNjMjdjODlmMTNjN2NjNTJjZjRlZDIwNzhiMDI5N2Q5YTZkOTQ0MT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=Pq31M4xNOXHCfiD9n0ZRxOdYLkGZwrjJYQclc5FKI1HSDnpVRSAHzrg7zD0dzp0mN%7Ea%7ESzFl3uyKU%7EblDsm1vzEdpoacNjgqw5N4T5fyg3JwtcNgT4u4MjSVRlp8b%7EjhWWbNp0rMA-QFe4qymgMJ0tDKrFlFZCHLafL5g8sJAguKA%7EVoOskD-6Y388EkPdnHgsS2RtRcOep4wuZ-%7EVVpqGlLayEfUhE22BJTdLnHoBIfIGr26LjCFS-ktcoswjbtKIdyEoSoLZw6OVBL9KI0x8Rpas4USQOxzcVsVSoyqzT1BKMF4WZpjvONlufkvSYUW4K6LMTUp1-ctS-MCzyEiw__&Key-Pair-Id=K24J24Z295AEI9"



# Функция для загрузки модели
def download_model(model_url, save_path):
    with st.spinner("Загрузка модели..."):
        # Create a progress bar
        progress_bar = st.progress(0)

        # Stream the download and update progress bar
        response = requests.get(model_url, stream=True)

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Calculate progress and ensure it is between 0 and 100
                    progress = (downloaded_size / total_size)
                    progress_bar.progress(progress)

        # Close the progress bar once done
        st.success("Модель загружена!")

# Загрузка модели, если её нет
if not os.path.exists(model_path):
    download_model(model_url, model_path)



SYSTEM_PROMPT = ("""Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. 
                  """)
# Температура модели
temperature = st.sidebar.slider("Температура модели", 0.0, 1.0, 0.8, 0.01)
#функция генерация ответа
def interact(text,
    model_path="./model-q4_K.gguf",
    n_ctx=8192,
    top_k=30,
    top_p=0.9,
    temperature=0.8,
    repeat_penalty=1.1
):
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_parts=1,
        verbose=False,
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_message = text
    messages.append({"role": "user", "content": user_message})
    for part in model.create_chat_completion(
            messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stream=True,
        ):
            delta = part["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]


#Чат-бот
st.title('🐵🔗 Тестовый запуск LLM')

# Initialize chat history and text state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "text" not in st.session_state:
    st.session_state.text = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Initialize response variable
    response = ""

    # Placeholder for assistant's response
    assistant_message = st.chat_message("assistant")
    response_placeholder = assistant_message.empty()

    # Generate assistant response
    for resp in interact(prompt, temperature=temperature, n_ctx=n_ctx):
        response += resp
        response_placeholder.markdown(response, unsafe_allow_html=True)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    print(st.session_state.messages[-3:])


