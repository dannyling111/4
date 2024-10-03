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
painting_styles = ["", "油画", "水彩画", "水墨画", "素描", "丙烯画", "写实主义"]

# Streamlit title and description in Chinese
st.title("AI 图像和文本生成器")
st.write("输入提示词，并选择一个模型生成文本或图像。")

# First section: Image generation
st.header("图像生成")

# Text input for user prompt for image generation (no default value)
input_image_prompt = st.text_input("请输入图像生成提示词")

# Dropdown for selecting painting style (default is no selection)
selected_style = st.selectbox("选择绘画风格", painting_styles, index=0)  # index=0 means the first item is selected (empty)

# Dropdown for selecting image bot model
selected_image_model = st.selectbox("选择图像生成模型", image_bots)

# Button for generating the image
if st.button("生成图像"):
    # Show a loading spinner while the image is being fetched
    with st.spinner("正在生成图像..."):

        # Adjust the prompt to include the selected painting style (if applicable)
        if selected_style:
            image_prompt = f"{input_image_prompt}，风格为{selected_style}"
        else:
            image_prompt = input_image_prompt

        async def fetch_image_response():
            message = ProtocolMessage(role="user", content=image_prompt)
            reply = ""

            # Handling image generation using the image bot
            async for partial in get_bot_response(messages=[message], bot_name=selected_image_model, api_key=api_key):
                response = json.loads(partial.raw_response["text"])
                reply += response["text"]
            return reply

        # Running the async function to get the image response
        image_response = asyncio.run(fetch_image_response())

        # If the response is an image, attempt to display the image
        image_url = image_response.split("(")[-1].split(")")[0]  # Assuming image URL extraction
        try:
            image_data = requests.get(image_url).content
            # Save and display the image
            image_path = f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            with open(image_path, "wb") as f:
                f.write(image_data)

            # Display the image using Streamlit
            st.image(image_data, caption="生成的图像", use_column_width=True)
        except MissingSchema:
            st.error(f"无效的图像链接: {image_url}")
        except Exception as e:
            st.error(f"无法加载图像: {str(e)}")

# Second section: Text generation
st.header("文本生成")

# Text input for user prompt for text generation (no default value)
input_text_prompt = st.text_input("请输入文本生成提示词")

# Dropdown for selecting text bot model
selected_text_model = st.selectbox("选择文本生成模型", text_bots)

# Button for generating the text
if st.button("生成文本"):
    # Show a loading spinner while the text is being fetched
    with st.spinner("正在生成文本..."):

        async def fetch_text_response():
            message = ProtocolMessage(role="user", content=input_text_prompt)
            reply = ""

            # If Gemini bot is selected, use Gemini-specific API
            if selected_text_model in gemini_bots:
                try:
                    model = genai.GenerativeModel(selected_text_model)
                    response = model.generate_content(message.content)
                    reply = response.text
                except Exception as e:
                    st.error(f"Error getting response from Gemini: {str(e)}")
                    return None
            else:
                # Handling text generation for regular models like GPT
                async for partial in get_bot_response(messages=[message], bot_name=selected_text_model, api_key=api_key):
                    response = json.loads(partial.raw_response["text"])
                    reply += response["text"]

            return reply

        # Running the async function to get the text response
        text_response = asyncio.run(fetch_text_response())

        # Display the text response
        if text_response:
            st.success("生成的文本：")
            st.write(text_response)
