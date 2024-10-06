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

# Load the API key from Streamlit secrets
api_key = st.secrets["api_keys"]["my_api_key"]

# Configure Gemini API
gemini_api_key = st.secrets["api_keys"]["gemini_api_key"]
genai.configure(api_key=gemini_api_key)

# Gemini bot names
gemini_bots = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]

# Select bot models from available options
image_bots = ["DALL-E-3", "FLUX-pro"]
text_bots = ["GPT-3.5-Turbo", "GPT-4", "Claude-3-Opus"] + gemini_bots
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

# Sidebar for navigation
st.sidebar.title("导航")
page = st.sidebar.selectbox("选择页面", ["图像生成", "文本生成"])

# Main header and description
st.title("AI 图像和文本生成器")

if page == "图像生成":
    # Image generation section
    st.header("图像生成")

    # Text input for user prompt for image generation
    input_image_prompt = st.text_input("请输入图像生成提示词")

    # Dropdown for selecting painting style
    selected_style = st.selectbox("选择绘画风格", painting_styles, index=0)

    # Dropdown for selecting image bot model
    selected_image_model = st.selectbox("选择图像生成模型", image_bots)

    # Button for generating the image
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
    # Text generation section
    st.header("文本生成")

    # Text input for user prompt for text generation
    input_text_prompt = st.text_input("请输入文本生成提示词")

    # Dropdown for selecting 'Language' and 'Thinking' options, with default empty option
    language_options = [''] + aisettings_df['a1'].dropna().tolist()  # Add '' as the first option
    thinking_options = [''] + aisettings_df['a2'].dropna().tolist()  # Add '' as the first option

    selected_language = st.selectbox("选择语言", language_options)
    selected_thinking = st.selectbox("选择思维方式", thinking_options)

    # Multi-select for 'a3' items (with default as unselected)
    a3_options = aisettings_df['a3'].dropna().tolist()  # Assuming 'a3' exists in the CSV
    selected_a3_items = st.multiselect("选择附加项 (a3 列)", a3_options, default=[])

    # Dropdown for selecting text bot model
    selected_text_model = st.selectbox("选择文本生成模型", text_bots)

    # Construct the message content by appending selected options only if not empty
    message_content = input_text_prompt
    if selected_language:  # Append 'Language' only if it's not empty
        message_content += f"\nLanguage: {selected_language}"
    if selected_thinking:  # Append 'Thinking' only if it's not empty
        message_content += f"\nThinking: {selected_thinking}"
    if selected_a3_items:  # Append 'a3' items if any are selected
        a3_string = ', '.join(selected_a3_items)
        message_content += f"\nAdditional: {a3_string}"

    # Display the final prompt input for user review
    st.subheader("AI 生成的最终提示词：")
    st.write(f"**Prompt Input:**\n{message_content}")

    # Button for generating the text
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



