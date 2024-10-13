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
import urllib.parse

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

    # Check if we need to load a keyword as input (from a previous keyword click)
    if 'input_text_prompt' not in st.session_state:
        st.session_state.input_text_prompt = ''  # Initialize if not present

    # Set the input text from session state (this could come from a keyword button click)
    input_text_prompt = st.text_input("请输入文本生成提示词", value=st.session_state.input_text_prompt)
    
    selected_language = st.selectbox("选择语言", language_options)
    selected_text_model = st.selectbox("选择文本生成模型", text_bots)
    fixed_prompt_options = aisettings_df['a5'].dropna().tolist()
    selected_fixed_prompt = st.selectbox("选择关键词生成模板", fixed_prompt_options)

    # Initialize session state to hold multiple rounds of keyword generation
    if 'keywords_rounds' not in st.session_state:
        st.session_state.keywords_rounds = []  # Empty list to hold each round of keywords

    # Button to generate new round of keywords
    if st.button("生成关键词和链接"):
        with st.spinner("正在生成关键词和链接..."):
            # Generate new keywords
            new_keywords = generate_keywords_and_links(
                input_text_prompt, selected_language, selected_text_model, selected_fixed_prompt
            )

            # Append the new keywords to session state (multiple rounds of keywords)
            st.session_state.keywords_rounds.append(new_keywords)

    # Button to clear results
    if st.button("清除结果"):
        st.session_state.keywords_rounds = []  # Clear all previous rounds of keywords
        st.session_state.input_text_prompt = ''  # Clear the input text
        st.success("所有结果已清除！")

    # Display all rounds of keywords
    if st.session_state.keywords_rounds:
        for round_idx, keywords in enumerate(st.session_state.keywords_rounds, 1):
            st.subheader(f"第 {round_idx} 轮生成的关键词")
            display_keywords_and_links(
                keywords, input_text_prompt, selected_language, selected_text_model, selected_fixed_prompt, round_idx
            )


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

def display_keywords_and_links(keywords, input_text, selected_language, selected_text_model, fixed_prompt_append, round_idx):
    for idx, keyword in enumerate(keywords):
        col1, col2, col3 = st.columns([3, 1, 1])  # Layout with 3 columns

        with col1:
            st.markdown(f"{keyword}")
        
        with col2:
            google_search = f"https://www.google.com/search?q={keyword}"
            youtube_search = f"https://www.youtube.com/results?search_query={keyword}"
            bilibili_search = f"https://search.bilibili.com/all?keyword={keyword}"
            st.markdown(f"[Google]({google_search}) | [YouTube]({youtube_search}) | [Bilibili]({bilibili_search})")
        
        with col3:
            # Ensure unique key by combining round index, loop index, and the keyword
            button_key = f"regen_{round_idx}_{idx}_{keyword}"
            if st.button(f"🔄 重新生成 {keyword}", key=button_key):
                # Use the clicked keyword as the new input and regenerate
                st.session_state.input_text_prompt = keyword

                # Generate new keywords from the clicked keyword (without rerun)
                new_keywords = generate_keywords_and_links(
                    input_text=keyword,  # Use current keyword as new input
                    language=selected_language,  # Reuse existing language setting
                    model=selected_text_model,  # Reuse existing model setting
                    fixed_prompt_append=fixed_prompt_append  # Reuse existing prompt append
                )
                if new_keywords:
                    st.session_state.keywords_rounds.append(new_keywords)  # Add to session state for history
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
    st.write(f"Prompt Input:\n{message_content}")

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
                st.markdown(f"- {keyword}: [Google Search]({google_search_link}) | [Google News]({google_news_link}) | [YouTube]({youtube_link})")



# 在 display_analysis_keywords 函数中添加模板选择下拉框
def display_analysis_keywords(keywords, selected_language, selected_text_model, fixed_prompt_options_a6, round_idx, generate_links, fixed_prompt_used):
    if not keywords:
        st.error("No keywords provided.")
        return

    for idx, keyword in enumerate(keywords):
        col1, col2, col3 = st.columns([3, 2, 3])  # 调整列宽

        with col1:
            st.markdown(f"**{keyword}**")
            if generate_links:
                encoded_keyword = urllib.parse.quote(keyword)
                google_search = f"https://www.google.com/search?q={encoded_keyword}"
                youtube_search = f"https://www.youtube.com/results?search_query={encoded_keyword}"
                bilibili_search = f"https://search.bilibili.com/all?keyword={encoded_keyword}"
                st.markdown(f"[Google]({google_search}) | [YouTube]({youtube_search}) | [Bilibili]({bilibili_search})")

        with col2:
            action_key = f"action_select_{round_idx}_{idx}"
            action_options = ["请选择操作", "🔄 重新生成关键词", "📝 生成分析文章"]
            selected_action = st.selectbox(
                "选择操作",
                options=action_options,
                key=action_key
            )
            action_processed_key = f"action_processed_{round_idx}_{idx}"

            if selected_action != "请选择操作" and not st.session_state.get(action_processed_key, False):
                if selected_action == "🔄 重新生成关键词":
                    with st.spinner(f"正在使用关键词 {keyword} 重新生成..."):
                        new_keywords = generate_keywords_and_links(
                            input_text=keyword,
                            language=selected_language,
                            model=selected_text_model,
                            fixed_prompt_append=fixed_prompt_used  # 使用当前轮次的模板
                        )
                        if new_keywords:
                            st.session_state.analysis_rounds.append({
                                'type': 'keywords',
                                'content': new_keywords,
                                'generate_links': generate_links,
                                'fixed_prompt': fixed_prompt_used  # 继续传递使用的模板
                            })
                            st.session_state[action_processed_key] = True
                            st.experimental_rerun()
                elif selected_action == "📝 生成分析文章":
                    with st.spinner(f"正在生成关于 {keyword} 的分析文章..."):
                        analysis_prompt = f"写一篇关于{keyword}的分析文章。语言: {selected_language}"
                        analysis_article = fetch_text_response(analysis_prompt, selected_text_model)
                        if analysis_article:
                            st.session_state.analysis_rounds.append({
                                'type': 'article',
                                'content': analysis_article
                            })
                            st.session_state[action_processed_key] = True
                            st.experimental_rerun()

        with col3:
            # 添加模板选择下拉框
            template_key = f"template_select_{round_idx}_{idx}"
            previous_template_key = f"previous_template_{round_idx}_{idx}"

            selected_template = st.selectbox(
                "选择关键词生成模板",
                options=['请选择模板'] + fixed_prompt_options_a6,
                key=template_key
            )

            previous_template = st.session_state.get(previous_template_key, '请选择模板')

            if selected_template != '请选择模板' and selected_template != previous_template:
                with st.spinner(f"正在使用关键词 {keyword} 和模板 {selected_template} 重新生成..."):
                    new_keywords = generate_keywords_and_links(
                        input_text=keyword,
                        language=selected_language,
                        model=selected_text_model,
                        fixed_prompt_append=selected_template
                    )
                    if new_keywords:
                        st.session_state.analysis_rounds.append({
                            'type': 'keywords',
                            'content': new_keywords,
                            'generate_links': generate_links,
                            'fixed_prompt': selected_template  # 存储新选择的模板
                        })
                        st.session_state[previous_template_key] = selected_template
                        st.experimental_rerun()

def analysis_generation_page():
    st.header("主题分析生成")

    if 'input_text_prompt_analysis' not in st.session_state:
        st.session_state.input_text_prompt_analysis = ''
    if 'analysis_rounds' not in st.session_state:
        st.session_state.analysis_rounds = []
    if 'trigger_rerun' not in st.session_state:
        st.session_state.trigger_rerun = False

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
                new_analysis_keywords = generate_keywords_and_links(
                    input_text_prompt_analysis, selected_language, selected_text_model, selected_fixed_prompt_a6)

                if new_analysis_keywords:
                    st.session_state.analysis_rounds.append({
                        'type': 'keywords',
                        'content': new_analysis_keywords,
                        'generate_links': generate_links,
                        'fixed_prompt': selected_fixed_prompt_a6  # 存储使用的模板
                    })
                    st.session_state.trigger_rerun = True
        else:
            st.warning("请输入文本生成提示词！")

    if st.button("清除结果"):
        st.session_state.analysis_rounds = []
        st.session_state.input_text_prompt_analysis = ''
        st.session_state.trigger_rerun = True
        st.success("所有结果已清除！")

    for round_idx, round_data in enumerate(st.session_state.analysis_rounds):
        if round_data['type'] == 'keywords':
            st.subheader(f"第 {round_idx + 1} 轮生成的主题关键词")
            display_analysis_keywords(
                round_data['content'],
                selected_language,
                selected_text_model,
                fixed_prompt_options_a6,  # 传递模板选项列表
                round_idx,
                round_data['generate_links'],
                round_data['fixed_prompt']  # 传递当前轮次使用的模板
            )
        elif round_data['type'] == 'article':
            st.subheader(f"分析文章：第 {round_idx + 1} 轮")
            st.write(round_data['content'])

    # Check if a rerun is needed and reset the trigger
    if st.session_state.trigger_rerun:
        st.session_state.trigger_rerun = False
        st.experimental_rerun()


def rerun_with_keyword(keyword, selected_language, selected_text_model, fixed_prompt_append):
    with st.spinner(f"正在使用关键词 {keyword} 重新生成..."):
        new_keywords = generate_keywords_and_links(keyword, selected_language, selected_text_model, fixed_prompt_append)
        if new_keywords:
            st.session_state.analysis_rounds.append({
                'type': 'keywords',
                'content': new_keywords,
                'generate_links': True  # Assuming we want links for rerun keywords
            })
          
# Fetch the analysis article using the API call
def fetch_text_response(message_content, model):
    async def fetch():
        message = ProtocolMessage(role="user", content=message_content)
        reply = ""
        async for partial in get_bot_response(messages=[message], bot_name=model, api_key=api_key):
            response = json.loads(partial.raw_response["text"])
            reply += response["text"]
        return reply

    return asyncio.run(fetch())


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

def excel_page():
    st.header("Excel 文件读取与编辑")

    # Excel 文件路径（固定路径）
    xlsx_path = "aisetting.xlsx"

    try:
        # 读取 Excel 文件中的所有工作表
        df = pd.read_excel(xlsx_path, sheet_name=None)  
        sheet_names = list(df.keys())  # 获取所有工作表名称
        selected_sheet = st.selectbox("选择工作表", sheet_names)  # 选择工作表

        # 获取当前工作表的数据
        data = df[selected_sheet]
        st.write(f"**当前显示的表：{selected_sheet}**")

        # 显示可编辑表格
        edited_data = st.data_editor(data, use_container_width=True)  # 使用 `st.data_editor`

        # 按钮保存编辑后的内容
        if st.button("保存编辑后的文件"):
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                # 遍历所有工作表，保存编辑后的内容
                for sheet_name, sheet_data in df.items():
                    if sheet_name == selected_sheet:
                        edited_data.to_excel(writer, index=False, sheet_name=sheet_name)  # 保存编辑的数据
                    else:
                        sheet_data.to_excel(writer, index=False, sheet_name=sheet_name)  # 保留未编辑的数据

            st.success(f"已成功保存编辑后的内容到 {xlsx_path}")

    except Exception as e:
        st.error(f"读取或保存 Excel 文件时出错: {e}")




def main():
    st.sidebar.title("导航")
    page = st.sidebar.selectbox("选择页面", ["关键词提取", "词云生成", "图像生成", "文本生成", "学习日语", "主题分析生成", "Excel"])

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
    elif page == "主题分析生成":
        analysis_generation_page()
    elif page == "Excel":
        excel_page()  # 调用新的Excel页面函数

# Run the app
if __name__ == "__main__":
    main()
