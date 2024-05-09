import streamlit as st
from llama_cpp import Llama
import requests

url = "https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf"
output_file = "model-q4_K.gguf"

response = requests.get(url)
if response.status_code == 200:
    with open(output_file, 'wb') as f:
        f.write(response.content)
    print("Файл успешно загружен.")
else:
    print("Не удалось загрузить файл.")


SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


def interact(text,
    model_path="./model-q4_K.gguf",
    n_ctx=8192,
    top_k=30,
    top_p=0.9,
    temperature=0.6,
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
    response = ""
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


# Кастомизация лейаута
st.set_page_config(page_title="DOCAI", page_icon="🤖", layout="wide", )
st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://api.unsplash.com/photos/?client_id=F3Lq5bwhNJdj_SbtRdcoMWaU2uW0Qd3iqeTrp9kXQOI"); 
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

st.title('🦜🔗 Тестовый запуск LLM')


with st.form('my_form'):
    text = st.text_area('Введите текст:', 'Привет!')
    submitted = st.form_submit_button('Старт')
    if submitted:
        st.write_stream(interact(text))
