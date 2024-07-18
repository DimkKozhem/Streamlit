import streamlit as st
from llama_cpp import Llama

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
    ("Saiga",),
)
if source == "Saiga":
    model_path = "model-q4_K.gguf"
    model_url = "https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf"


# Функция для загрузки модели
def download_model(model_url, save_path):
    response = requests.get(model_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Загрузка модели, если её нет
if not os.path.exists(model_path):
    st.write("Загрузка модели...")
    download_model(model_url, model_path)
    st.write("Модель загружена.")


SYSTEM_PROMPT = ("""Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. 
                  """)



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



st.title('🐵🔗 Тестовый запуск LLM')




chat_container = st.empty()

with st.form('my_form'):
    # окно для ввода текста
    user_input = st.text_input("Введите сообщение:", "")
    submitted = st.form_submit_button('отправить')

    if submitted:
        if 'text' not in st.session_state:
            st.session_state.text = ''
        st.session_state.text += f"\nВы: {user_input}\n"

        st.session_state.text += f"\nSaiga: "

        for response in interact(user_input):
            st.session_state.text +=response
            chat_container.markdown(st.session_state.text, unsafe_allow_html=True)
        st.session_state.text +="\n"
        user_input = ""
