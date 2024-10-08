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

# Load the API key from Streamlit secrets
api_key = st.secrets["api_keys"]["my_api_key"]

# Configure Gemini API
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
genai.configure(api_key=gemini_api_key)

# Gemini bot names
gemini_bots = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]

# Select bot models from available options
image_bots = ["DALL-E-3", "FLUX-pro"]
text_bots = gemini_bots+["GPT-3.5-Turbo", "GPT-4", "Claude-3-Opus"]
bot_models = text_bots + image_bots

# List of painting styles
painting_styles = ["", "油画", "水彩画", "水墨画", "素描", "丙烯画", 
    "写实主义", "印象派", "梵高风格", "达芬奇风格", "莫奈风格",
    "毕加索风格", "德加风格", "雷诺阿风格", "米开朗基罗风格",
    "达利风格", "高更风格", "康定斯基风格", "塞尚风格", 
    "爱德华·马奈风格", "齐白石风格", "张大千风格", 
    "徐悲鸿风格", "吴冠中风格"]

# Load CSV data for 'Language' and 'Thinking' from 'aisetting.csv'
csv_path = "aisetting.csv"
aisettings_df = pd.read_csv(csv_path)

# Extract 'Language' and 'Thinking' options from columns 'a1' and 'a2'
language_options = aisettings_df['a1'].dropna().tolist()
thinking_options = aisettings_df['a2'].dropna().tolist()

# Google Trends and WordCloud functions
font_path = 'NotoSansCJK-Regular.ttc'  # Update to your font path

def get_google_trends():
    pytrends = TrendReq(hl='en-US', tz=360)
    countries = ['united_states', 'japan', 'hong_kong', 'united_kingdom', 'taiwan', 'india', 'singapore', 'australia']
    trends_list = []
    for country in countries:
        trends = pytrends.trending_searches(pn=country)
        trends_list.extend(trends.values.tolist())
    return [item[0].strip() for item in trends_list]

def muted_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    h = random.randint(180, 360)  # Muted colors in blue/purple range
    s = random.randint(50, 80)  # Moderate saturation
    l = random.randint(40, 50)  # Medium lightness for readability
    return f"hsl({h}, {s}%, {l}%)"

def generate_wordcloud(keywords):
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

# Sidebar for navigation
st.sidebar.title("导航")
page = st.sidebar.selectbox("选择页面", ["图像生成", "文本生成", "词云生成", "关键词提取"])

# Main header and description
st.title("AI 图像、文本和词云生成器")

# State to store initial and recursive results
if "recursive_results" not in st.session_state:
    st.session_state["recursive_results"] = []
if "initial_keywords" not in st.session_state:
    st.session_state["initial_keywords"] = []

# Function to generate keywords and links from a given input
def generate_keywords_and_links(input_text):
    fixed_prompt_append = f"""
    Provide a list of 20 most related keywords, in the following format:

    - Keyword 1
    - Keyword 2
    - Keyword 3
    ...

    Provide the result only in the list format as shown above. Respond in {language_options[selected_language]}.
    """
    final_prompt = f"{input_text}\n{fixed_prompt_append}"

    async def fetch_text_response():
        message = ProtocolMessage(role="user", content=final_prompt)
        reply = ""
        async for partial in get_bot_response(messages=[message], bot_name=selected_text_model, api_key=api_key):
            response = json.loads(partial.raw_response["text"])
            reply += response["text"]
        return reply

    # Fetch the AI response (expected in list format)
    text_response = asyncio.run(fetch_text_response())
    
    if text_response:
        try:
            # Split the list by new lines and dashes (since AI will return a list in the format `- Keyword`)
            keywords = [line.strip()[2:] for line in text_response.splitlines() if line.startswith("-")]
            return keywords
        except Exception as e:
            st.error(f"处理关键词列表时出现错误: {str(e)}")
            return []

# Function to display keywords and links on the page with buttons
def display_keywords_and_links(keywords, is_initial=False, context=""):
    for index, keyword in enumerate(keywords):
        google_search = f"https://www.google.com/search?q={keyword}"
        youtube_search = f"https://www.youtube.com/results?search_query={keyword}"
        bilibili_search = f"https://search.bilibili.com/all?keyword={keyword}"
        
        # Display keyword and links
        st.markdown(f"- **{keyword}**: [Google Search]({google_search}) | [YouTube Search]({youtube_search}) | [Bilibili Search]({bilibili_search})")
        
        # Add a general button to generate new keywords, with a unique key
        if st.button("生成新关键词", key=f"{context}_{index}"):
            new_keywords = generate_keywords_and_links(keyword)
            if new_keywords:
                # Store new results in session_state to keep them across rerenders
                st.session_state["recursive_results"].append({
                    "input": keyword,
                    "results": new_keywords
                })

if page == "关键词提取":
    st.header("关键词提取和搜索链接生成")
    input_text_prompt = st.text_input("请输入文本生成提示词")

    # Language selection
    language_options = {
        "中文": "Chinese",
        "英文": "English",
        "日文": "Japanese"
    }
    selected_language = st.selectbox("选择语言", list(language_options.keys()))

    # Bot selection (same as in other pages)
    selected_text_model = st.selectbox("选择文本生成模型", text_bots)

    # Button to trigger text generation
    if st.button("生成关键词和链接"):
        with st.spinner("正在生成关键词和链接..."):
            # Generate initial keywords based on the input prompt
            keywords = generate_keywords_and_links(input_text_prompt)
            if keywords:
                st.session_state["initial_keywords"] = keywords

    # Display initial generated keywords if available
    if st.session_state["initial_keywords"]:
        st.subheader("初始关键词和搜索链接")
        display_keywords_and_links(st.session_state["initial_keywords"], is_initial=True, context="initial")

    # Display any recursive results
    if st.session_state["recursive_results"]:
        for idx, result in enumerate(st.session_state["recursive_results"]):
            st.subheader(f"使用 '{result['input']}' 生成的新关键词")
            display_keywords_and_links(result['results'], context=f"recursive_{idx}")
elif page == "图像生成":
    st.header("图像生成")

    input_image_prompt = st.text_input("请输入图像生成提示词")
    selected_style = st.selectbox("选择绘画风格", painting_styles, index=0)
    selected_image_model = st.selectbox("选择图像生成模型", image_bots)

    if st.button("生成图像"):
        with st.spinner("正在生成图像..."):
            image_prompt = f"{input_image_prompt}，风格为{selected_style}" if selected_style else input_image_prompt

            async def fetch_image_response():
                message = ProtocolMessage(role="user", content=image_prompt)
                reply = ""
                async for partial in get_bot_response(messages=[message], bot_name=selected_image_model, api_key=api_key):
                    response = json.loads(partial.raw_response["text"])
                    reply += response["text"]
                return reply

            image_response = asyncio.run(fetch_image_response())
            image_url = image_response.split("(")[-1].split(")")[0]

            try:
                image_data = requests.get(image_url).content
                image_path = f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                with open(image_path, "wb") as f:
                    f.write(image_data)
                st.image(image_data, caption="生成的图像", use_column_width=True)
            except MissingSchema:
                st.error(f"无效的图像链接: {image_url}")
            except Exception as e:
                st.error(f"无法加载图像: {str(e)}")

elif page == "文本生成":
    st.header("文本生成")

    input_text_prompt = st.text_input("请输入文本生成提示词")
    selected_language = st.selectbox("选择语言", [''] + language_options)
    selected_thinking = st.selectbox("选择思维方式", [''] + thinking_options)
    a3_options = aisettings_df['a3'].dropna().tolist()  # Assuming 'a3' exists in the CSV
    selected_a3_items = st.multiselect("选择附加项 (a3 列)", a3_options, default=[])
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

            async def fetch_text_response():
                message = ProtocolMessage(role="user", content=message_content)
                reply = ""
                if selected_text_model in gemini_bots:
                    try:
                        model = genai.GenerativeModel(selected_text_model)
                        response = model.generate_content(message.content)
                        reply = response.text
                    except Exception as e:
                        st.error(f"Error getting response from Gemini: {str(e)}")
                        return None
                else:
                    async for partial in get_bot_response(messages=[message], bot_name=selected_text_model, api_key=api_key):
                        response = json.loads(partial.raw_response["text"])
                        reply += response["text"]
                return reply

            text_response = asyncio.run(fetch_text_response())
            if text_response:
                st.success("生成的文本：")
                st.write(text_response)

elif page == "词云生成":
    st.header("生成 Google Trends 词云")

    if st.button("生成词云"):
        with st.spinner("正在生成词云..."):
            trend_keywords = get_google_trends()
            wordcloud_image, keyword_frequencies = generate_wordcloud(trend_keywords)

            # Display the word cloud image
            st.image(wordcloud_image, caption="Google Trends 词云", use_column_width=True)

            # Sort keywords by frequency and get the top 20
            sorted_keywords = sorted(keyword_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]

            st.subheader("关键词搜索链接")
            for keyword, _ in sorted_keywords:  # Display the top 20 keywords
                google_search_link = f"https://www.google.com/search?q={keyword}"
                google_news_link = f"https://news.google.com/search?q={keyword}"
                youtube_link = f"https://www.youtube.com/results?search_query={keyword}"
                st.markdown(f"- **{keyword}**: [Google Search]({google_search_link}) | [Google News]({google_news_link}) | [YouTube]({youtube_link})")

