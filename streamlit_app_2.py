import pandas as pd
from pytrends.request import TrendReq
from collections import Counter
from wordcloud import WordCloud
import random
import streamlit as st
import requests
from io import BytesIO
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to 'TkAgg' or any other suitable backend
import matplotlib.pyplot as plt

font_path = 'NotoSansCJK-Regular.ttc'  # Update to your font path

# Function to fetch Google Trends data
def get_google_trends():
    pytrends = TrendReq(hl='en-US', tz=360)
    countries = ['united_states', 'japan', 'hong_kong', 'united_kingdom', 'taiwan', 'india', 'singapore', 'australia']
    trends_list = []

    for country in countries:
        trends = pytrends.trending_searches(pn=country)
        trends_list.extend(trends.values.tolist())

    return [item[0].strip() for item in trends_list]

# Custom color function for muted colors
def muted_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    h = random.randint(180, 360)  # Muted colors in blue/purple range
    s = random.randint(50, 80)  # Moderate saturation
    l = random.randint(40, 50)  # Medium lightness
    return f"hsl({h}, {s}%, {l}%)"

# Function to generate and display word cloud
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

    # Save image to in-memory buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Streamlit page for word cloud and keyword links
st.sidebar.title("导航")
page = st.sidebar.selectbox("选择页面", ["图像生成", "文本生成", "词云生成"])

if page == "词云生成":
    st.header("生成 Google Trends 词云")

    if st.button("生成词云"):
        with st.spinner("正在生成词云..."):
            # Fetch trends and generate word cloud
            trend_keywords = get_google_trends()
            wordcloud_image = generate_wordcloud(trend_keywords)

            # Display word cloud image
            st.image(wordcloud_image, caption="Google Trends 词云", use_column_width=True)

            # Display keyword links to Google News and YouTube
            st.subheader("关键词搜索链接")
            for keyword in trend_keywords[:10]:  # Limit to top 10 for display
                google_news_link = f"https://news.google.com/search?q={keyword}"
                youtube_link = f"https://www.youtube.com/results?search_query={keyword}"
                st.markdown(f"- **{keyword}**: [Google News]({google_news_link}) | [YouTube]({youtube_link})")

