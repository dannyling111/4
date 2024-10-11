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
import openpyxl
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

# Load the Excel file (replace the file path with your .xlsx file)
xlsx_path = "aisetting.xlsx"  # Ensure the correct path to your Excel file
aisettings_df = pd.read_excel(xlsx_path)

# Extract language and thinking options
language_options = aisettings_df['a1'].dropna().tolist()
thinking_options = aisettings_df['a2'].dropna().tolist()

# General-purpose functions


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


# Page Block 1: Keyword Extraction
def keyword_extraction_page():
    st.header("关键词提取和搜索链接生成")
    input_text_prompt = st.text_input("请输入文本生成提示词")

    selected_language = st.selectbox("选择语言", language_options)
    selected_text_model = st.selectbox("选择文本生成模型", text_bots)
    
    # 从a5列读取选项并简化prompt选择逻辑
    fixed_prompt_options = aisettings_df['a5'].dropna().tolist()
    selected_fixed_prompt = st.selectbox("选择关键词生成模板", fixed_prompt_options)

    if st.button("生成关键词和链接"):
        with st.spinner("正在生成关键词和链接..."):
            keywords = generate_keywords_and_links(input_text_prompt, selected_language, selected_text_model, selected_fixed_prompt)
            if keywords:
                display_keywords_and_links(keywords)

# 修改后的generate_keywords_and_links函数
def generate_keywords_and_links(input_text, language, model, fixed_prompt_append):
    # 构造包含语言选项的最终提示词
    final_prompt = f"{input_text}\n{fixed_prompt_append}\nLanguage: {language}" if language else f"{input_text}\n{fixed_prompt_append}"

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

    text_response = asyncio.run(fetch_text_response())
    if text_response:
        try:
            keywords = [line.strip()[2:] for line in text_response.splitlines() if line.startswith("-")]
            return keywords
        except Exception as e:
            st.error(f"Error processing keywords: {str(e)}")
            return []

def display_keywords_and_links(keywords):
    for keyword in keywords:
        google_search = f"https://www.google.com/search?q={keyword}"
        youtube_search = f"https://www.youtube.com/results?search_query={keyword}"
        bilibili_search = f"https://search.bilibili.com/all?keyword={keyword}"
        st.markdown(f"- **{keyword}**: [Google Search]({google_search}) | [YouTube Search]({youtube_search}) | [Bilibili Search]({bilibili_search})")
# Page Block 2: Image Generation
def image_generation_page():
    st.header("图像生成")
    input_image_prompt = st.text_input("请输入图像生成提示词")
    selected_style = st.selectbox("选择绘画风格", painting_styles, index=0)
    selected_image_model = st.selectbox("选择图像生成模型", image_bots)

    if st.button("生成图像"):
        with st.spinner("正在生成图像..."):
            image_prompt = f"{input_image_prompt}，风格为{selected_style}" if selected_style else input_image_prompt
            image_response = fetch_image_response(image_prompt, selected_image_model)

            image_url = re.search(r'\((.*?)\)', image_response).group(1)
            try:
                image_data = requests.get(image_url).content
                st.image(image_data, caption="生成的图像", use_column_width=True)
            except Exception as e:
                st.error(f"无法加载图像: {str(e)}")

# Page Block 3: Text Generation
def text_generation_page():
    st.header("文本生成")
    input_text_prompt = st.text_input("请输入文本生成提示词")
    selected_language = st.selectbox("选择语言", [''] + language_options)
    selected_thinking = st.selectbox("选择思维方式", [''] + thinking_options)
    selected_a3_items = st.multiselect("选择附加项 (a3 列)", aisettings_df['a3'].dropna().tolist())
    selected_text_model = st.selectbox("选择文本生成模型", text_bots)

    message_content = input_text_prompt
    if selected_language:
        message_content += f"\nLanguage: {selected_language}"
    if selected_thinking:
        message_content += f"\nThinking: {selected_thinking}"
    if selected_a3_items:
        message_content += f"\nAdditional: {', '.join(selected_a3_items)}"

    st.subheader("AI 生成的最终提示词：")
    st.write(f"**Prompt Input:**\n{message_content}")

    if st.button("生成文本"):
        with st.spinner("正在生成文本..."):
            text_response = fetch_text_response(message_content, selected_text_model)
            if text_response:
                st.success("生成的文本：")
                st.write(text_response)

# Page Block 4: Word Cloud Generation
def wordcloud_generation_page():
    st.header("生成 Google Trends 词云")
    if st.button("生成词云"):
        with st.spinner("正在生成词云..."):
            trend_keywords = get_google_trends()
            wordcloud_image, keyword_frequencies = generate_wordcloud(trend_keywords)

            st.image(wordcloud_image, caption="Google Trends 词云", use_column_width=True)

            sorted_keywords = sorted(keyword_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
            st.subheader("关键词搜索链接")
            for keyword, _ in sorted_keywords:
                google_search_link = f"https://www.google.com/search?q={keyword}"
                google_news_link = f"https://news.google.com/search?q={keyword}"
                youtube_link = f"https://www.youtube.com/results?search_query={keyword}"
                st.markdown(f"- **{keyword}**: [Google Search]({google_search_link}) | [Google News]({google_news_link}) | [YouTube]({youtube_link})")

def fetch_text_response(message_content, model):
    async def fetch():
        message = ProtocolMessage(role="user", content=message_content)
        reply = ""
        async for partial in get_bot_response(messages=[message], bot_name=model, api_key=api_key):
            response = json.loads(partial.raw_response["text"])
            reply += response["text"]
        return reply

    return asyncio.run(fetch())

# Page Block 5: Japanese Learning Page
def japanese_learning_page():
    st.header("学习日语")
    input_text_prompt = st.text_input("请输入与学习日语相关的提示词")
    
    # Extract options for a4 column (附加项) and change to single select (单选)
    selected_a4_item = st.selectbox("选择附加项 (a4 列)", [''] + aisettings_df['a4'].dropna().tolist())  # 改为单选

    # Select language (单选)
    selected_language = st.selectbox("选择语言", [''] + language_options)  # 单选语言
    
    # Select text generation model
    selected_text_model = st.selectbox("选择文本生成模型", text_bots)

    # Construct final prompt
    message_content = input_text_prompt
    if selected_language:
        message_content += f"\nLanguage: {selected_language}"
    if selected_a4_item:
        message_content += f"\n附加项: {selected_a4_item}"  # 单选附加项
    
    st.subheader("AI 生成的最终提示词：")
    st.write(f"**Prompt Input:**\n{message_content}")

    if st.button("生成文本"):
        with st.spinner("正在生成文本..."):
            text_response = fetch_text_response(message_content, selected_text_model)
            if text_response:
                st.success("生成的文本：")
                st.write(text_response)

# Sidebar for navigation and main app structure
st.sidebar.title("导航")
page = st.sidebar.selectbox("选择页面", ["关键词提取", "词云生成", "图像生成", "文本生成", "学习日语"])

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
    elif page == "学习日语":
        japanese_learning_page()

# Run the app
if __name__ == "__main__":
    main()
