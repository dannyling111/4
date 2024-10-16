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
            reply += partial.text
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
            reply += partial.text
        return reply

    return asyncio.run(fetch())

def handle_selection(keyword_path, a7_option, fixed_prompt, language, model, generate_links):
    # 获取当前关键词的节点引用
    node = st.session_state.analysis_data
    for key in keyword_path:
        if 'sub_keywords' not in node:
            node['sub_keywords'] = {}
        if key not in node['sub_keywords']:
            node['sub_keywords'][key] = {
                'articles': [],
                'sub_keywords': {},
                'generate_links': generate_links
            }
        node = node['sub_keywords'][key]

    # 如果选择了命令，生成文章
    if a7_option != '请选择命令':
        with st.spinner(f"生成关于 {keyword_path[-1]} 的文章..."):
            article = generate_article(keyword_path[-1], a7_option, language, model)
            if article:
                node['articles'].append(article)
                st.success(f"成功生成关于 {keyword_path[-1]} 的文章！")

    # 如果选择了模板，生成更多关键词
    if fixed_prompt != '请选择模板':
        with st.spinner(f"根据模板 {fixed_prompt} 生成更多关键词..."):
            new_keywords = generate_keywords_and_links(keyword_path[-1], language, model, fixed_prompt)
            if new_keywords:
                if 'sub_keywords' not in node:
                    node['sub_keywords'] = {}
                for kw in new_keywords:
                    if kw not in node['sub_keywords']:
                        node['sub_keywords'][kw] = {
                            'articles': [],
                            'sub_keywords': {},
                            'generate_links': generate_links
                        }
                st.success("成功生成更多关键词！")


def display_analysis_keywords(node, keyword_path, selected_language, selected_text_model, level=0):
    # 定义颜色
    round_colors = ["#AED6F1", "#A9DFBF", "#F5B7B1", "#F9E79F", "#D7BDE2"]
    selected_color = round_colors[level % len(round_colors)]

    # 获取命令和模板选项
    a7_options = ['请选择命令'] + aisettings_df['a7'].dropna().tolist()
    fixed_prompt_options_a6 = ['请选择模板'] + aisettings_df['a6'].dropna().tolist()

    keyword = keyword_path[-1]

    # 使用容器样式
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
        # 显示缩进
        indent = "&nbsp;" * (level * 4)
        # 展示关键词及其相关链接
        st.markdown(f"{indent}**{keyword}**", unsafe_allow_html=True)
        if node.get('generate_links', True):
            encoded_keyword = urllib.parse.quote(keyword)
            google_search = f"https://www.google.com/search?q={encoded_keyword}"
            youtube_search = f"https://www.youtube.com/results?search_query={encoded_keyword}"
            bilibili_search = f"https://search.bilibili.com/all?keyword={encoded_keyword}"
            st.markdown(f"{indent}[Google]({google_search}) | [YouTube]({youtube_search}) | [Bilibili]({bilibili_search})", unsafe_allow_html=True)

        # 展示与关键词相关的文章
        for article in node.get('articles', []):
            st.write(article)

   with col2:
        # 命令选择下拉菜单
        select_a7_key = '_'.join(['a7'] + keyword_path)
        
        def on_a7_change():
            selected_a7_option = st.session_state[select_a7_key]
            if selected_a7_option != '请选择命令':
                handle_selection(keyword_path, selected_a7_option, '请选择模板', selected_language, selected_text_model, node.get('generate_links', True))
                # 重置选项
                st.session_state[select_a7_key] = '请选择命令'
                st.experimental_rerun()
        
        st.selectbox(
            "选择命令",
            options=a7_options,
            key=select_a7_key,
            on_change=on_a7_change
        )

        # 模板选择下拉菜单
        select_fixed_prompt_key = '_'.join(['a6'] + keyword_path)
        
        def on_a6_change():
            selected_fixed_prompt = st.session_state[select_fixed_prompt_key]
            if selected_fixed_prompt != '请选择模板':
                handle_selection(keyword_path, '请选择命令', selected_fixed_prompt, selected_language, selected_text_model, node.get('generate_links', True))
                # 重置选项
                st.session_state[select_fixed_prompt_key] = '请选择模板'
                st.experimental_rerun()
        
        st.selectbox(
            "选择模板",
            options=fixed_prompt_options_a6,
            key=select_fixed_prompt_key,
            on_change=on_a6_change
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # 递归显示子关键词
    sub_keywords = node.get('sub_keywords', {})
    for sub_kw in sub_keywords:
        display_analysis_keywords(sub_keywords[sub_kw], keyword_path + [sub_kw], selected_language, selected_text_model, level + 1)

def analysis_generation_page():
    st.header("主题分析生成")

    # 初始化 session_state
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {}

    # 输入框
    input_text_prompt_analysis = st.text_input(
        "请输入文本生成提示词", key='input_text_prompt_analysis'
    )

    # 语言和文本模型选择
    selected_language = st.selectbox("选择语言", language_options, key='selected_language')
    selected_text_model = st.selectbox("选择文本生成模型", text_bots, key='selected_text_model')

    # 命令选择 Dropdown
    a7_options = ['请选择命令'] + aisettings_df['a7'].dropna().tolist()
    selected_command = st.selectbox(
        "选择命令", a7_options, key='selected_command'
    )

    # 关键词生成模板 Dropdown
    fixed_prompt_options_a6 = ['请选择模板'] + aisettings_df['a6'].dropna().tolist()
    selected_template = st.selectbox(
        "选择关键词生成模板", fixed_prompt_options_a6, key='selected_template'
    )

    # 是否生成关键词相关链接的 Checkbox
    generate_links = st.checkbox("是否生成关键词相关的搜索链接", value=True, key='generate_links')

    # 按钮：生成内容
    if st.button("生成内容"):
        content_generated = False  # 标记是否有内容生成

        root_keyword = input_text_prompt_analysis.strip()
        if not root_keyword:
            st.warning("请输入文本生成提示词。")
        else:
            # 初始化根节点
            if root_keyword not in st.session_state.analysis_data:
                st.session_state.analysis_data[root_keyword] = {
                    'articles': [],
                    'sub_keywords': {},
                    'generate_links': generate_links
                }

            keyword_path = [root_keyword]  # 根关键词的路径

            # 如果选择了命令，则生成文章内容
            if selected_command != '请选择命令':
                with st.spinner(f"正在生成关于命令 '{selected_command}' 的内容..."):
                    article = generate_article(
                        root_keyword,
                        selected_command,
                        selected_language,
                        selected_text_model
                    )
                    if article:
                        st.session_state.analysis_data[root_keyword]['articles'].append(article)
                        st.success("命令内容生成成功！")
                        content_generated = True

            # 如果选择了模板，则生成关键词内容
            if selected_template != '请选择模板':
                with st.spinner(f"根据模板 '{selected_template}' 生成关键词..."):
                    new_keywords = generate_keywords_and_links(
                        root_keyword,
                        selected_language,
                        selected_text_model,
                        selected_template
                    )
                    if new_keywords:
                        for kw in new_keywords:
                            if kw not in st.session_state.analysis_data[root_keyword]['sub_keywords']:
                                st.session_state.analysis_data[root_keyword]['sub_keywords'][kw] = {
                                    'articles': [],
                                    'sub_keywords': {},
                                    'generate_links': generate_links
                                }
                        st.success("关键词内容生成成功！")
                        content_generated = True

            # 如果没有生成任何内容，显示警告
            if not content_generated:
                st.warning("请确保至少选择一个命令或模板。")

    # 按钮：清除结果
    if st.button("清除结果"):
        st.session_state.analysis_data = {}
        st.session_state.input_text_prompt_analysis = ''
        st.session_state.selected_command = '请选择命令'
        st.session_state.selected_template = '请选择模板'
        st.success("所有结果已清除！")
        st.experimental_rerun()

    # 显示分析数据
    for root_kw in st.session_state.analysis_data:
        display_analysis_keywords(st.session_state.analysis_data[root_kw], [root_kw], selected_language, selected_text_model)

def main():
    st.sidebar.title("导航")
    page = st.sidebar.selectbox("选择页面", ["主题分析生成"])  # 如果需要，添加其他页面

    if page == "主题分析生成":
        analysis_generation_page()
    # elif page == "其他页面":
    #     other_page_function()

if __name__ == "__main__":
    main()
