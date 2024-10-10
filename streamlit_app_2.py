# Import necessary modules and libraries
import time
import pandas as pd
from fastapi_poe.types import ProtocolMessage
from fastapi_poe.client import get_bot_response
import json
import asyncio
from datetime import datetime
import requests
from PIL import Image
import os
import re
import streamlit as st
from requests.exceptions import MissingSchema
import google.generativeai as genai
from pytrends.request import TrendReq
from collections import Counter
from wordcloud import WordCloud
import random
from io import BytesIO
import matplotlib.pyplot as plt

# Configure Matplotlib to use 'Agg' backend for Streamlit compatibility
plt.switch_backend('Agg')

# Load API keys and configure models
api_key = st.secrets["api_keys"]["my_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
genai.configure(api_key=gemini_api_key)

gemini_bots = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
image_bots = ["DALL-E-3", "FLUX-pro"]
text_bots = gemini_bots + ["GPT-3.5-Turbo", "GPT-4", "Claude-3-Opus"]

painting_styles = ["", "油画", "水彩画", "水墨画", "素描", "丙烯画", "写实主义", "印象派", "梵高风格", "达芬奇风格", "莫奈风格", 
                   "毕加索风格", "德加风格", "雷诺阿风格", "米开朗基罗风格", "达利风格", "高更风格", "康定斯基风格", 
                   "塞尚风格", "爱德华·马奈风格", "齐白石风格", "张大千风格", "徐悲鸿风格", "吴冠中风格"]

csv_path = "aisetting.csv"
aisettings_df = pd.read_csv(csv_path)

# Extract language and thinking options
language_options = aisettings_df['a1'].dropna().tolist()
thinking_options = aisettings_df['a2'].dropna().tolist()

# Initialize session state
if "initial_vocabs" not in st.session_state:
    st.session_state["initial_vocabs"] = []
if "recursive_results" not in st.session_state:
    st.session_state["recursive_results"] = []

# General-purpose functions (same as before)
def generate_keywords_and_links(input_text, language, model):
    fixed_prompt_append = """
    Provide a list of 20 most related keywords, in the following format:
    - Keyword 1
    - Keyword 2
    - Keyword 3
    """
    final_prompt = f"{input_text}\n{fixed_prompt_append}"

    async def fetch_text_response():
        message = ProtocolMessage(role="user", content=final_prompt)
        reply = ""
        async for partial in get_bot_response(messages=[message], bot_name=model, api_key=api_key):
            response = json.loads(partial.raw_response["text"])
            reply += response["text"]
        return reply

    text_response = asyncio.run(fetch_text_response())
    if text_response:
        try:
            keywords = [line.strip()[2:] for line in text_response.splitlines() if line.startswith("-")]
            return keywords
        except Exception as e:
            st.error(f"Error processing keywords: {str(e)}")
            return []

def fetch_image_response(image_prompt, model):
    async def fetch():
        message = ProtocolMessage(role="user", content=image_prompt)
        reply = ""
        async for partial in get_bot_response(messages=[message], bot_name=model, api_key=api_key):
            response = json.loads(partial.raw_response["text"])
            reply += response["text"]
        return reply

    return asyncio.run(fetch())

def get_google_trends():
    pytrends = TrendReq(hl='en-US', tz=360)
    countries = ['united_states', 'japan', 'hong_kong', 'united_kingdom', 'taiwan', 'india', 'singapore', 'australia']
    trends_list = []
    for country in countries:
        trends = pytrends.trending_searches(pn=country)
        trends_list.extend(trends.values.tolist())
    return [item[0].strip() for item in trends_list]

def generate_wordcloud(keywords):
    font_path = 'NotoSansCJK-Regular.ttc'  # Update to your font path
    wordcloud = WordCloud(
        font_path=font_path,
        width=1600, 
        height=1600, 
        background_color="white",
        color_func=muted_color_func
    ).generate_from_frequencies(Counter(keywords))

    plt.figure(figsize=(10, 10), dpi=100)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf, wordcloud.words_

def muted_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    h = random.randint(180, 360)
    s = random.randint(50, 80)
    l = random.randint(40, 50)
    return f"hsl({h}, {s}%, {l}%)"


# Fetch Text Response Function
def fetch_text_response(prompt, model):
    async def fetch():
        message = ProtocolMessage(role="user", content=prompt)
        reply = ""
        async for partial in get_bot_response(messages=[message], bot_name=model, api_key=api_key):
            response = json.loads(partial.raw_response["text"])
            reply += response["text"]
        return reply
    return asyncio.run(fetch())

# Japanese Learning Page Block (updated to include the model)
def japanese_learning_page():
    st.header("日语学习")
    input_vocab_prompt = st.text_input("请输入日语学习相关的提示词")

    selected_text_model = st.selectbox("选择文本生成模型", text_bots)

    if st.button("生成词汇和解释"):
        with st.spinner("正在生成相关词汇..."):
            vocab_items = generate_japanese_vocab(input_vocab_prompt, selected_text_model)
            if vocab_items:
                st.session_state["initial_vocabs"] = vocab_items

    if "initial_vocabs" in st.session_state and st.session_state["initial_vocabs"]:
        st.subheader("初始词汇和解释")
        display_japanese_vocab(st.session_state["initial_vocabs"], selected_text_model, is_initial=True, context="initial")

    if "recursive_results" in st.session_state and st.session_state["recursive_results"]:
        for idx, result in enumerate(st.session_state["recursive_results"]):
            st.subheader(f"使用 '{result['input']}' 生成的新词汇")
            display_japanese_vocab(result['results'], selected_text_model, context=f"recursive_{idx}")

# Sidebar for navigation and main app structure
st.sidebar.title("导航")
page = st.sidebar.selectbox("选择页面", ["关键词提取", "词云生成", "图像生成", "文本生成", "日语学习"])

# Main function to switch between blocks/pages
def main():
    if page == "关键词提取":
        keyword_extraction_page()
    elif page == "词云生成":
        wordcloud_generation_page()
    elif page == "图像生成":
        image_generation_page()
    elif page == "文本生成":
        text_generation_page()
    elif page == "日语学习":
        japanese_learning_page()

# Run the app
if __name__ == "__main__":
    main()
