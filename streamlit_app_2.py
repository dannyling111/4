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
import urllib.parse

# Configure Matplotlib to use 'Agg' backend for Streamlit compatibility
plt.switch_backend('Agg')

# Load API keys and configure models
api_key = st.secrets["api_keys"]["my_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
genai.configure(api_key=gemini_api_key)

gemini_bots = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
text_bots = gemini_bots + ["GPT-3.5-Turbo", "GPT-4", "Claude-3-Opus"]

# Load the Excel file (replace the file path with your .xlsx file)
xlsx_path = "aisetting.xlsx"  # Ensure the correct path to your Excel file
aisettings_df = pd.read_excel(xlsx_path)

# Extract language options
language_options = aisettings_df['a1'].dropna().tolist()

# General-purpose functions

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

# Functions used in Analysis Generation

def generate_keywords_and_links(input_text, language, model, fixed_prompt_append):
    # Construct the final prompt including the language option
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

def display_analysis_keywords(keywords, selected_language, selected_text_model, round_idx, generate_links):
    # Get options from a7 column
    a7_options = ['请选择命令'] + aisettings_df['a7'].dropna().tolist()
    # Second dropdown: options from a6 column
    fixed_prompt_options_a6 = ['请选择模板'] + aisettings_df['a6'].dropna().tolist()

    for idx, keyword in enumerate(keywords):
        col1, col2 = st.columns([3, 2])  # Two columns: keyword | dropdowns stacked vertically

        with col1:
            st.markdown(f"**{keyword}**")
            if generate_links:
                encoded_keyword = urllib.parse.quote(keyword)
                google_search = f"https://www.google.com/search?q={encoded_keyword}"
                youtube_search = f"https://www.youtube.com/results?search_query={encoded_keyword}"
                bilibili_search = f"https://search.bilibili.com/all?keyword={encoded_keyword}"
                st.markdown(f"[Google]({google_search}) | [YouTube]({youtube_search}) | [Bilibili]({bilibili_search})")

        # Stack the two dropdowns vertically within the same column
        with col2:
            # Dropdown 1: Command selection
            select_a7_key = f"a7_template_select_{round_idx}_{idx}"
            selected_a7_option = st.selectbox(
                "选择命令", a7_options, key=select_a7_key
            )

            # Initialize previous selection
            prev_select_a7_key = f"prev_{select_a7_key}"
            if prev_select_a7_key not in st.session_state:
                st.session_state[prev_select_a7_key] = a7_options[0]

            # Check if selection changed
            if selected_a7_option != st.session_state[prev_select_a7_key]:
                st.session_state[prev_select_a7_key] = selected_a7_option
                if selected_a7_option != '请选择命令':
                    handle_selection(keyword, selected_a7_option, '请选择模板', selected_language, selected_text_model, generate_links)

            # Dropdown 2: Template selection
            select_fixed_prompt_key = f"fixed_prompt_select_{round_idx}_{idx}"
            selected_fixed_prompt = st.selectbox(
                "选择模板", fixed_prompt_options_a6, key=select_fixed_prompt_key
            )

            # Initialize previous selection
            prev_select_fixed_prompt_key = f"prev_{select_fixed_prompt_key}"
            if prev_select_fixed_prompt_key not in st.session_state:
                st.session_state[prev_select_fixed_prompt_key] = fixed_prompt_options_a6[0]

            # Check if selection changed
            if selected_fixed_prompt != st.session_state[prev_select_fixed_prompt_key]:
                st.session_state[prev_select_fixed_prompt_key] = selected_fixed_prompt
                if selected_fixed_prompt != '请选择模板':
                    handle_selection(keyword, '请选择命令', selected_fixed_prompt, selected_language, selected_text_model, generate_links)


def handle_selection(keyword, a7_option, fixed_prompt, language, model, generate_links):
    # If an a7 option is selected, generate an article
    if a7_option != '请选择命令':
        with st.spinner(f"生成关于 {keyword} 的文章..."):
            article = generate_article(keyword, a7_option, language, model)
            if article:
                st.session_state.analysis_rounds.append({
                    'type': 'article',
                    'content': article
                })
                st.success(f"成功生成关于 {keyword} 的文章！")

    # If a fixed prompt is selected, generate more keywords
    if fixed_prompt != '请选择模板':
        with st.spinner(f"根据模板 {fixed_prompt} 生成更多关键词..."):
            new_keywords = generate_keywords_and_links(keyword, language, model, fixed_prompt)
            if new_keywords:
                st.session_state.analysis_rounds.append({
                    'type': 'keywords',
                    'content': new_keywords,
                    'generate_links': generate_links  # Use the current generate_links setting
                })
                st.success("成功生成更多关键词！")

def generate_article(keyword, command, language, model):
    prompt = f"关键词: {keyword}\n命令: {command}\n语言: {language}"
    return fetch_text_response(prompt, model)

def fetch_text_response(prompt, model):
    async def fetch():
        message = ProtocolMessage(role="user", content=prompt)
        reply = ""
        async for partial in get_bot_response(messages=[message], bot_name=model, api_key=api_key):
            response = json.loads(partial.raw_response["text"])
            reply += response["text"]
        return reply

    return asyncio.run(fetch())

# Page Block: Word Cloud Generation
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
                encoded_keyword = urllib.parse.quote(keyword)  # URL-encode the keyword
                google_search_link = f"https://www.google.com/search?q={encoded_keyword}"
                google_news_link = f"https://news.google.com/search?q={encoded_keyword}"
                youtube_link = f"https://www.youtube.com/results?search_query={encoded_keyword}"
                st.markdown(f"- {keyword}: [Google Search]({google_search_link}) | [Google News]({google_news_link}) | [YouTube]({youtube_link})")

def analysis_generation_page():
    st.header("主题分析生成")

    if 'input_text_prompt_analysis' not in st.session_state:
        st.session_state.input_text_prompt_analysis = ''
    if 'analysis_rounds' not in st.session_state:
        st.session_state.analysis_rounds = []

    input_text_prompt_analysis = st.text_input(
        "请输入文本生成提示词", value=st.session_state.input_text_prompt_analysis)

    selected_language = st.selectbox("选择语言", language_options)
    selected_text_model = st.selectbox("选择文本生成模型", text_bots)

    fixed_prompt_options_a6 = aisettings_df['a6'].dropna().tolist()
    selected_fixed_prompt_a6 = st.selectbox("选择关键词生成模板", fixed_prompt_options_a6)

    generate_links = st.checkbox("是否生成关键词相关的搜索链接", value=True)

    if st.button("生成关键词"):
        if input_text_prompt_analysis.strip():
            with st.spinner("正在生成关键词..."):
                new_keywords = generate_keywords_and_links(
                    input_text_prompt_analysis, selected_language, selected_text_model, selected_fixed_prompt_a6
                )
                if new_keywords:
                    st.session_state.analysis_rounds.append({
                        'type': 'keywords',
                        'content': new_keywords,
                        'generate_links': generate_links  # Ensure correct setting
                    })
        else:
            st.warning("请输入文本生成提示词！")

    if st.button("清除结果"):
        st.session_state.analysis_rounds = []
        st.session_state.input_text_prompt_analysis = ''
        st.success("所有结果已清除！")

    for round_idx, round_data in enumerate(st.session_state.analysis_rounds):
        if round_data['type'] == 'keywords':
            st.subheader(f"第 {round_idx + 1} 轮生成的主题关键词")
            display_analysis_keywords(
                round_data['content'], selected_language, selected_text_model, round_idx, round_data['generate_links']
            )
        elif round_data['type'] == 'article':
            st.subheader(f"分析文章：第 {round_idx + 1} 轮")
            st.write(round_data['content'])

# Page Block: Excel File Reading and Editing
def excel_page():
    st.header("Excel 文件读取与编辑")

    # Excel file path (fixed path)
    xlsx_path = "aisetting.xlsx"

    try:
        # Read all sheets from the Excel file
        df = pd.read_excel(xlsx_path, sheet_name=None)  
        sheet_names = list(df.keys())  # Get all sheet names
        selected_sheet = st.selectbox("选择工作表", sheet_names)  # Select sheet

        # Get data from the current sheet
        data = df[selected_sheet]
        st.write(f"**当前显示的表：{selected_sheet}**")

        # Display editable table
        edited_data = st.data_editor(data, use_container_width=True)  # Use st.data_editor

        # Button to save edited content
        if st.button("保存编辑后的文件"):
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                # Iterate over all sheets and save edited content
                for sheet_name, sheet_data in df.items():
                    if sheet_name == selected_sheet:
                        edited_data.to_excel(writer, index=False, sheet_name=sheet_name)  # Save edited data
                    else:
                        sheet_data.to_excel(writer, index=False, sheet_name=sheet_name)  # Retain unedited data

            st.success(f"已成功保存编辑后的内容到 {xlsx_path}")

    except Exception as e:
        st.error(f"读取或保存 Excel 文件时出错: {e}")

def main():
    st.sidebar.title("导航")
    page = st.sidebar.selectbox("选择页面", ["词云生成", "主题分析生成", "Excel"])

    if page == "词云生成":
        wordcloud_generation_page()
    elif page == "主题分析生成":
        analysis_generation_page()
    elif page == "Excel":
        excel_page()

# Run the app
if __name__ == "__main__":
    main()
