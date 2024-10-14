import pandas as pd
import streamlit as st
from collections import Counter
from wordcloud import WordCloud
import random
from io import BytesIO
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import asyncio
import json
from fastapi_poe.types import ProtocolMessage
from fastapi_poe.client import get_bot_response
import google.generativeai as genai
import urllib.parse
import openpyxl

# 配置 Matplotlib 与 Streamlit 兼容
plt.switch_backend('Agg')

# 加载 API keys 和配置模型
api_key = st.secrets["api_keys"]["my_api_key"]
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
genai.configure(api_key=gemini_api_key)

# 模型列表
gemini_bots = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
text_bots = gemini_bots + ["GPT-3.5-Turbo", "GPT-4", "Claude-3-Opus"]

# 加载 Excel 文件
xlsx_path = "aisetting.xlsx"
aisettings_df = pd.read_excel(xlsx_path)

# Helper: 异步获取生成的文本
async def fetch_text_response(prompt, model):
    message = ProtocolMessage(role="user", content=prompt)
    reply = ""
    async for partial in get_bot_response(messages=[message], bot_name=model, api_key=api_key):
        response = json.loads(partial.raw_response["text"])
        reply += response["text"]
    return reply

# 生成关键词并生成链接
def generate_keywords_and_links(input_text, language, model, fixed_prompt_append):
    prompt = f"{input_text}\n{fixed_prompt_append}\nLanguage: {language}" if language else input_text
    response = asyncio.run(fetch_text_response(prompt, model))
    return [line.strip()[2:] for line in response.splitlines() if line.startswith("-")]

# Word Cloud 页面：获取 Google Trends 数据并生成词云
def get_google_trends():
    pytrends = TrendReq(hl='en-US', tz=360)
    countries = [
        'united_states', 'japan', 'hong_kong', 'united_kingdom',
        'taiwan', 'india', 'singapore', 'australia'
    ]
    trends_list = []
    for country in countries:
        trends = pytrends.trending_searches(pn=country)
        trends_list.extend(trends.values.tolist())
    return [item[0].strip() for item in trends_list]

def generate_wordcloud(keywords):
    font_path = 'NotoSansCJK-Regular.ttc'
    wordcloud = WordCloud(
        font_path=font_path, width=1600, height=1600,
        background_color="white", color_func=muted_color_func
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
                google_link = f"https://www.google.com/search?q={keyword}"
                youtube_link = f"https://www.youtube.com/results?search_query={keyword}"
                st.markdown(f"- {keyword}: [Google]({google_link}) | [YouTube]({youtube_link})")

# Analysis Generation 页面
def display_analysis_keywords(keywords, selected_language, selected_text_model, round_idx, generate_links):
    for idx, keyword in enumerate(keywords):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{keyword}**")
            if generate_links:
                google_link = f"https://www.google.com/search?q={urllib.parse.quote(keyword)}"
                youtube_link = f"https://www.youtube.com/results?search_query={urllib.parse.quote(keyword)}"
                st.markdown(f"[Google]({google_link}) | [YouTube]({youtube_link})")

def analysis_generation_page():
    st.header("主题分析生成")

    if 'input_text_prompt_analysis' not in st.session_state:
        st.session_state.input_text_prompt_analysis = ''
    if 'analysis_rounds' not in st.session_state:
        st.session_state.analysis_rounds = []

    input_text_prompt_analysis = st.text_input("请输入文本生成提示词", value=st.session_state.input_text_prompt_analysis)
    language_options = aisettings_df['a1'].dropna().tolist()
    selected_language = st.selectbox("选择语言", language_options)
    selected_text_model = st.selectbox("选择文本生成模型", text_bots)
    fixed_prompt_options = aisettings_df['a6'].dropna().tolist()
    selected_fixed_prompt = st.selectbox("选择关键词生成模板", fixed_prompt_options)
    generate_links = st.checkbox("是否生成关键词相关的搜索链接", value=True)

    if st.button("生成关键词"):
        if input_text_prompt_analysis.strip():
            with st.spinner("正在生成关键词..."):
                new_keywords = generate_keywords_and_links(
                    input_text_prompt_analysis, selected_language, selected_text_model, selected_fixed_prompt
                )
                if new_keywords:
                    st.session_state.analysis_rounds.append({
                        'type': 'keywords', 'content': new_keywords, 'generate_links': generate_links
                    })
        else:
            st.warning("请输入文本生成提示词！")

    if st.button("清除结果"):
        st.session_state.analysis_rounds = []
        st.session_state.input_text_prompt_analysis = ''
        st.success("所有结果已清除！")

    for round_idx, round_data in enumerate(st.session_state.analysis_rounds):
        st.subheader(f"第 {round_idx + 1} 轮生成的主题关键词")
        display_analysis_keywords(
            round_data['content'], selected_language, selected_text_model, round_idx, round_data['generate_links']
        )

# Excel 页面功能
def excel_page():
    st.header("Excel 文件读取与编辑")
    try:
        df = pd.read_excel(xlsx_path, sheet_name=None)
        sheet_names = list(df.keys())
        selected_sheet = st.selectbox("选择工作表", sheet_names)

        data = df[selected_sheet]
        st.write(f"**当前显示的表：{selected_sheet}**")

        edited_data = st.data_editor(data, use_container_width=True)

        if st.button("保存编辑后的文件"):
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                for sheet_name, sheet_data in df.items():
                    if sheet_name == selected_sheet:
                        edited_data.to_excel(writer, index=False, sheet_name=sheet_name)
                    else:
                        sheet_data.to_excel(writer, index=False, sheet_name=sheet_name)
            st.success(f"已成功保存编辑后的内容到 {xlsx_path}")

    except Exception as e:
        st.error(f"读取或保存 Excel 文件时出错: {e}")

# 主函数
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
