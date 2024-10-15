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

# 在此处定义下拉菜单的选项列表
a7_options = ['请选择命令'] + aisettings_df['a7'].dropna().tolist()
fixed_prompt_options_a6 = ['请选择模板'] + aisettings_df['a6'].dropna().tolist()

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
    final_prompt = f"{input_text}\n{fixed_prompt_append}\nLanguage: {language}"

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

def generate_label(path, idx):
    """根据层级和索引生成编号标签，例如：1, 1.1, 1.1.1"""
    parts = [str(i) for i in path + [idx + 1]]
    return ".".join(parts)


def display_analysis_keywords(
    keywords, selected_language, selected_text_model, round_idx, generate_links, depth=1, path=None
):
    MAX_DEPTH = 3  # 限制递归层级

    if depth > MAX_DEPTH:
        return  # 超出最大层级时停止递归

    if path is None:
        path = []

    # 每轮使用不同的颜色
    round_colors = ['#e6f7ff', '#fff1f0', '#f6ffed', '#fff7e6', '#f9f0ff']
    background_color = round_colors[round_idx % len(round_colors)]

    for idx, keyword in enumerate(keywords):
        label = generate_label(path, idx)

        # 定义容器样式
        container_style = f"""
            <div style="
                background-color: {background_color};
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 8px;
                border: 1px solid #d0d0d0;">
        """
        st.markdown(container_style, unsafe_allow_html=True)
        st.markdown(f"**{label} {keyword}**")

        if generate_links:
            encoded_keyword = urllib.parse.quote(keyword)
            st.markdown(
                f"[Google](https://www.google.com/search?q={encoded_keyword}) | "
                f"[YouTube](https://www.youtube.com/results?search_query={encoded_keyword})"
            )

        # 更新路径并生成唯一键
        current_path = path + [idx]
        path_str = "_".join(map(str, current_path))

        # Dropdown 选择
        select_a7_key = f"a7_{path_str}"
        select_fixed_prompt_key = f"fixed_{path_str}"

        selected_a7_option = st.selectbox(
            "选择命令", a7_options, key=select_a7_key, on_change=handle_selection,
            args=(keyword, "a7", path_str)
        )
        selected_fixed_prompt = st.selectbox(
            "选择模板", fixed_prompt_options_a6, key=select_fixed_prompt_key, on_change=handle_selection,
            args=(keyword, "fixed", path_str)
        )

        # 定义状态键
        content_key = f"content_{path_str}"
        article_key = f"article_{path_str}"

        # 显示生成的关键词
        if content_key in st.session_state:
            st.markdown("**生成的关键词：**")
            display_analysis_keywords(
                st.session_state[content_key], selected_language, selected_text_model,
                round_idx, generate_links, depth + 1, current_path
            )

        # 显示生成的文章
        if article_key in st.session_state:
            st.markdown("**生成的内容：**")
            st.write(st.session_state[article_key])

        st.markdown("</div>", unsafe_allow_html=True)


def handle_selection(keyword, option_type, path_str):
    """根据用户的下拉菜单选择执行相应逻辑。"""
    content_key = f"content_{path_str}"
    article_key = f"article_{path_str}"

    if option_type == "a7":
        selected_command = st.session_state[f"a7_{path_str}"]
        if selected_command != "请选择命令" and article_key not in st.session_state:
            with st.spinner(f"生成关于 '{keyword}' 的文章..."):
                article = generate_article(
                    keyword, selected_command, selected_language, selected_text_model
                )
                if article:
                    st.session_state[article_key] = article

    elif option_type == "fixed":
        selected_template = st.session_state[f"fixed_{path_str}"]
        if selected_template != "请选择模板" and content_key not in st.session_state:
            with st.spinner(f"根据模板 '{selected_template}' 生成关键词..."):
                new_keywords = generate_keywords_and_links(
                    keyword, selected_language, selected_text_model, selected_template
                )
                if new_keywords:
                    st.session_state[content_key] = new_keywords






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

    # 初始化 session_state
    if 'input_text_prompt_analysis' not in st.session_state:
        st.session_state.input_text_prompt_analysis = ''
    if 'selected_command' not in st.session_state:
        st.session_state.selected_command = None  # 初始化命令选择状态
    if 'selected_template' not in st.session_state:
        st.session_state.selected_template = None  # 初始化模板选择状态
    if 'analysis_rounds' not in st.session_state:
        st.session_state.analysis_rounds = []

    # 输入框
    input_text_prompt_analysis = st.text_input(
        "请输入文本生成提示词", value=st.session_state.input_text_prompt_analysis
    )

    # 语言和文本模型选择
    selected_language = st.selectbox("选择语言", language_options)
    selected_text_model = st.selectbox("选择文本生成模型", text_bots)

    # 命令选择 Dropdown
    a7_options = ['请选择命令'] + aisettings_df['a7'].dropna().tolist()
    st.session_state.selected_command = st.selectbox(
        "选择命令", a7_options, index=0  # 默认显示为“请选择命令”
    )

    # 关键词生成模板 Dropdown
    fixed_prompt_options_a6 = ['请选择模板'] + aisettings_df['a6'].dropna().tolist()
    st.session_state.selected_template = st.selectbox(
        "选择关键词生成模板", fixed_prompt_options_a6, index=0
    )

    # 是否生成关键词相关链接的 Checkbox
    generate_links = st.checkbox("是否生成关键词相关的搜索链接", value=True)

    # 按钮：生成内容
    if st.button("生成内容"):
        content_generated = False  # 标记是否有内容生成
        round_data = {}

        # 如果选择了命令，则生成文章内容
        if (st.session_state.selected_command and 
            st.session_state.selected_command != '请选择命令'):
            with st.spinner(f"正在生成关于命令 '{st.session_state.selected_command}' 的内容..."):
                article = generate_article(
                    input_text_prompt_analysis,
                    st.session_state.selected_command,
                    selected_language,
                    selected_text_model
                )
                if article:
                    round_data['article'] = article
                    content_generated = True

        # 如果选择了模板，则生成关键词内容
        if (st.session_state.selected_template and 
            st.session_state.selected_template != '请选择模板'):
            with st.spinner(f"根据模板 '{st.session_state.selected_template}' 生成关键词..."):
                new_keywords = generate_keywords_and_links(
                    input_text_prompt_analysis,
                    selected_language,
                    selected_text_model,
                    st.session_state.selected_template
                )
                if new_keywords:
                    round_data['keywords'] = new_keywords
                    round_data['generate_links'] = generate_links  # 记录链接生成设置
                    content_generated = True

        # 保存生成的内容到分析轮次
        if content_generated:
            st.session_state.analysis_rounds.append(round_data)
            st.success("内容生成成功！")
        else:
            st.warning("请确保至少选择一个命令或模板。")

    # 清除结果的按钮
    if st.button("清除结果"):
        st.session_state.analysis_rounds = []
        st.session_state.input_text_prompt_analysis = ''
        st.session_state.selected_command = None
        st.session_state.selected_template = None
        st.success("所有结果已清除！")

    # 展示生成的关键词和文章内容
    for round_idx, round_data in enumerate(st.session_state.analysis_rounds):
        # 如果有关键词内容，优先展示关键词和链接
        if 'keywords' in round_data:
            st.subheader(f"第 {round_idx + 1} 轮生成的关键词")
            display_analysis_keywords(
                round_data['keywords'], selected_language, selected_text_model,
                round_idx, round_data['generate_links']
            )

        # 如果有文章内容，直接在关键词内容之后展示
        if 'article' in round_data:
            st.subheader(f"分析文章：第 {round_idx + 1} 轮")
            st.write(round_data['article'])

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
