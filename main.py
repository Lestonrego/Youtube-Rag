import streamlit as st
import helper as lch
import textwrap

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="What is the YouTube video URL?",
            max_chars=100
        )
        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=200,
            key="query"
        )
        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
    with st.spinner("Fetching transcript and generating answer..."):
        db = lch.create_db_from_youtube_video_url(youtube_url)
        response, docs = lch.get_response_from_query(db, query)
        st.subheader("Answer:")
        st.text(textwrap.fill(response, width=85))
        with st.expander("Relevant transcript chunks"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc)