import streamlit as st
from llama_cpp import Llama
import numpy as np
import requests
import os
from urllib.parse import urlencode
import zipfile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.faiss import FAISS
import torch

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
    ("Saiga",'Mistral-Nemo',"ruGPT-3.5-13B"),
)
if source == "Saiga":
    n_ctx = 8192 # контекстное окно
    model_path = "model-q4_K.gguf"
    model_url = "https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf"
    SYSTEM_PROMPT = ("""Ты — Сайга, русскоязычный автоматический Юрист. Ты разговариваешь с людьми и делаешь юридическую консультацию.""")

elif source == "Mistral-Nemo":
    n_ctx = 128000 # контекстное окно
    model_path = "Mistral-Nemo.gguf"
    model_url = "https://huggingface.co/second-state/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
    SYSTEM_PROMPT = ("""Ты — Юрист, русскоязычный автоматический Юрист. Ты разговариваешь с людьми и делаешь юридическую консультацию со сылками на нормы право.""")

elif source == "ruGPT-3.5-13B":
    n_ctx = 2040 # контекстное окно
    model_path = "ruGPT-3.5-13B.gguf"
    model_url = "https://huggingface.co/oblivious/ruGPT-3.5-13B-GGUF/resolve/main/ruGPT-3.5-13B-Q8_0.gguf"
    SYSTEM_PROMPT = ("""Ты — Юрист, русскоязычный автоматический Юрист. Ты разговариваешь с людьми и делаешь юридическую консультацию.""")



# Функция для загрузки модели
def download_model(model_url, save_path,text):

    with st.spinner(f"Загрузка {text}..."):
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
        st.success("Загрузка завершена!")

# Загрузка модели, если её нет
if not os.path.exists(model_path):
    text = 'модели'
    download_model(model_url, model_path, text)

#грузим модель из HuggingFace для создания эмбеддингов из тесктов, можно взять любую модель, включая GPT или GiGaChat
model_name=model_name_hug = 'sentence-transformers/all-MiniLM-L6-v2' # тут нужно подобрать  sentence sintisimilariti
model_kwargs = {'device': 'cuda'}
embeddings = HuggingFaceEmbeddings(model_name=model_name_hug, model_kwargs = model_kwargs )

name_base = 'faiss_index.zip'

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/wvdi-x-Nkt_hfA'  # Сюда вписываете вашу ссылку

# Загружаем и расспаковываем базу знаний
final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']
name_faile = 'faiss_index.zip'
download_response = requests.get(download_url)
with open(name_faile, 'wb') as f:   # Здесь укажите нужный путь к файлу
    f.write(download_response.content)
with zipfile.ZipFile(name_faile, 'r') as zip_ref:
    zip_ref.extractall()

# присоединяемся к БД
db = FAISS.load_local(
    '/home/dimk/langchain/Streamlit/faiss_index',
    embeddings,
    allow_dangerous_deserialization=True
)


# Температура модели
temperature = st.sidebar.slider("Температура модели", 0.0, 1.0, 0.5, 0.01)


#функция генерация ответа

def interact(text,
    model_path="./model-q4_K.gguf",
    n_ctx=4192,
    top_k=30,
    top_p=0.9,
    temperature=0.8,
    repeat_penalty=1.1
):

    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
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
if prompt := st.chat_input("Задайте свой вопрос?"):
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
    text_base = db.similarity_search_with_score(prompt, k=3)
    text_base = '\n\n'.join([f'Вопрос: {i[0].page_content}\nОтвет: {i[0].metadata["Ответ"]}' for i in text_base if i[1] < 0.2])
    if text_base:
        promp = f"""Ответь на вопрос: "{prompt}"
        На похожий вопрос отвечали следующим образом:\n{text_base}. 
        В конце ответа напиши источники на основании которых построен ответ"""
    else:
        promp = f"""Ответь на вопрос: "{prompt}".
        На похожий вопрос отвечали следующим образом:\n{text_base}. 
        В конце ответа напиши источники на основании которых построен ответ"""
    print()
    print(promp)
    for resp in interact(promp, temperature=temperature, n_ctx=n_ctx):
        response += resp
        response_placeholder.markdown(response, unsafe_allow_html=True)
    if text_base:
        response += '<h3><u>*При подготовке ответа использована база знаний*</u></h3>'
        response_placeholder.markdown(response, unsafe_allow_html=True)
    else:
        response += '<h3><u>*Модель отвечала без подсказок*</u></h3>'
        response_placeholder.markdown(response, unsafe_allow_html=True)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



