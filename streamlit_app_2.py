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

# Load the Excel file
xlsx_path = "aisetting.xlsx"  # Ensure the correct path to your Excel file
aisettings_df = pd.read_excel(xlsx_path)

# Extract language options
language_options = aisettings_df['a1'].dropna().tolist()

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

def handle_selection(keyword, a7_option, fixed_prompt, language, model, generate_links):
    # Ensure 'analysis_rounds' is initialized
    if 'analysis_rounds' not in st.session_state or not isinstance(st.session_state.analysis_rounds, dict):
        st.session_state.analysis_rounds = {}

    # If a command is selected, generate an article
    if a7_option != '请选择命令':
        with st.spinner(f"生成关于 {keyword} 的文章..."):
            article = generate_article(keyword, a7_option, language, model)
            if article:
                if keyword not in st.session_state.analysis_rounds:
                    st.session_state.analysis_rounds[keyword] = {'articles': [], 'keywords': [], 'generate_links': generate_links}
                st.session_state.analysis_rounds[keyword]['articles'].append(article)
                st.success(f"成功生成关于 {keyword} 的文章！")

    # If a fixed prompt is selected, generate more keywords
    if fixed_prompt != '请选择模板':
        with st.spinner(f"根据模板 {fixed_prompt} 生成更多关键词..."):
            new_keywords = generate_keywords_and_links(keyword, language, model, fixed_prompt)
            if new_keywords:
                if keyword not in st.session_state.analysis_rounds:
                    st.session_state.analysis_rounds[keyword] = {'articles': [], 'keywords': [], 'generate_links': generate_links}
                st.session_state.analysis_rounds[keyword]['keywords'].extend(new_keywords)
                st.success("成功生成更多关键词！")

def display_analysis_keywords(keywords, selected_language, selected_text_model, round_idx, generate_links):
    # Define colors
    round_colors = ["#AED6F1", "#A9DFBF", "#F5B7B1", "#F9E79F", "#D7BDE2"]
    selected_color = round_colors[round_idx % len(round_colors)]

    # Get command and template options
    a7_options = ['请选择命令'] + aisettings_df['a7'].dropna().tolist()
    fixed_prompt_options_a6 = ['请选择模板'] + aisettings_df['a6'].dropna().tolist()

    for idx, keyword in enumerate(keywords):
        # Use container style
        container_style = f"""
            <div style="
                background-color: {selected_color};
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 8px;
                border: 1px solid #d0d0d0;">
        """
        st.markdown(container_style, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])

        with col1:
            # Display keyword and related links
            st.markdown(f"**{keyword}**")
            if generate_links:
                encoded_keyword = urllib.parse.quote(keyword)
                google_search = f"https://www.google.com/search?q={encoded_keyword}"
                youtube_search = f"https://www.youtube.com/results?search_query={encoded_keyword}"
                bilibili_search = f"https://search.bilibili.com/all?keyword={encoded_keyword}"
                st.markdown(f"[Google]({google_search}) | [YouTube]({youtube_search}) | [Bilibili]({bilibili_search})")

            # Check if the keyword is in 'analysis_rounds'
            if 'analysis_rounds' in st.session_state and keyword in st.session_state.analysis_rounds:
                # Display related articles
                for article in st.session_state.analysis_rounds[keyword].get('articles', []):
                    st.write(article)

                # Display new keywords
                sub_keywords = st.session_state.analysis_rounds[keyword].get('keywords', [])
                if sub_keywords:
                    st.markdown("**生成的新关键词：**")
                    for new_keyword in sub_keywords:
                        st.markdown(f"- {new_keyword}")

        with col2:
            # Command selection dropdown
            select_a7_key = f"a7_template_select_{round_idx}_{idx}"
            selected_a7_option = st.selectbox("选择命令", a7_options, key=select_a7_key)

            # Handle command selection
            if selected_a7_option != '请选择命令':
                handle_selection(keyword, selected_a7_option, '请选择模板', selected_language, selected_text_model, generate_links)
                st.experimental_rerun()

            # Fixed prompt selection dropdown
            select_fixed_prompt_key = f"fixed_prompt_select_{round_idx}_{idx}"
            selected_fixed_prompt = st.selectbox("选择模板", fixed_prompt_options_a6, key=select_fixed_prompt_key)

            # Handle fixed prompt selection
            if selected_fixed_prompt != '请选择模板':
                handle_selection(keyword, '请选择命令', selected_fixed_prompt, selected_language, selected_text_model, generate_links)
                st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

def analysis_generation_page():
    st.header("主题分析生成")

    # Ensure 'analysis_rounds' is initialized
    if 'analysis_rounds' not in st.session_state or not isinstance(st.session_state.analysis_rounds, dict):
        st.session_state.analysis_rounds = {}

    # Input text prompt
    input_text_prompt_analysis = st.text_input(
        "请输入文本生成提示词", key='input_text_prompt_analysis'
    )

    selected_language = st.selectbox("选择语言", language_options, key='selected_language')
    selected_text_model = st.selectbox("选择文本生成模型", text_bots, key='selected_text_model')

    a7_options = ['请选择命令'] + aisettings_df['a7'].dropna().tolist()
    selected_command = st.selectbox("选择命令", a7_options, key='selected_command')

    fixed_prompt_options_a6 = ['请选择模板'] + aisettings_df['a6'].dropna().tolist()
    selected_template = st.selectbox("选择关键词生成模板", fixed_prompt_options_a6, key='selected_template')

    generate_links = st.checkbox("是否生成关键词相关的搜索链接", value=True, key='generate_links')

    if st.button("生成内容"):
        content_generated = False

        if selected_command != '请选择命令':
            with st.spinner(f"正在生成关于命令 '{selected_command}' 的内容..."):
                article = generate_article(
                    input_text_prompt_analysis,
                    selected_command,
                    selected_language,
                    selected_text_model
                )
                if article:
                    if input_text_prompt_analysis not in st.session_state.analysis_rounds:
                        st.session_state.analysis_rounds[input_text_prompt_analysis] = {'articles': [], 'keywords': [], 'generate_links': generate_links}
                    st.session_state.analysis_rounds[input_text_prompt_analysis]['articles'].append(article)
                    st.success("命令内容生成成功！")
                    content_generated = True

        if selected_template != '请选择模板':
            with st.spinner(f"根据模板 '{selected_template}' 生成关键词..."):
                new_keywords = generate_keywords_and_links(
                    input_text_prompt_analysis,
                    selected_language,
                    selected_text_model,
                    selected_template
                )
                if new_keywords:
                    if input_text_prompt_analysis not in st.session_state.analysis_rounds:
                        st.session_state.analysis_rounds[input_text_prompt_analysis] = {'articles': [], 'keywords': [], 'generate_links': generate_links}
                    st.session_state.analysis_rounds[input_text_prompt_analysis]['keywords'].extend(new_keywords)
                    st.success("关键词内容生成成功！")
                    content_generated = True

        if not content_generated:
            st.warning("请确保至少选择一个命令或模板。")

    if st.button("清除结果"):
        st.session_state.analysis_rounds = {}
        st.session_state.input_text_prompt_analysis = ''
        st.session_state.selected_command = '请选择命令'
        st.session_state.selected_template = '请选择模板'
        st.success("所有结果已清除！")
        st.experimental_rerun()

    # Ensure 'analysis_rounds' is a non-empty dictionary
    if st.session_state.analysis_rounds:
        for round_idx, (keyword, round_data) in enumerate(st.session_state.analysis_rounds.items()):
            if round_data.get('articles'):
                st.subheader(f"分析文章：关于关键词 '{keyword}'")
                for article in round_data.get('articles', []):
                    st.write(article)

            if round_data.get('keywords'):
                st.subheader(f"关于关键词 '{keyword}' 生成的关键词")
                display_analysis_keywords(
                    round_data['keywords'], st.session_state.selected_language, st.session_state.selected_text_model,
                    round_idx, round_data.get('generate_links', True)
                )

def main():
    st.sidebar.title("导航")
    page = st.sidebar.selectbox("选择页面", ["词云生成", "主题分析生成", "Excel"])

    if page == "词云生成":
        wordcloud_generation_page()
    elif page == "主题分析生成":
        analysis_generation_page()
    elif page == "Excel":
        excel_page()

if __name__ == "__main__":
    main()
