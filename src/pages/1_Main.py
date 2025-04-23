import streamlit as st
import pandas as pd
import itertools
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json

def build_wordcloud(data):
    data_ = data.reset_index()
    word_freq = dict(
        zip(
            data_[0],
            data_["count"]
        )
    )
    return WordCloud(width=1200, height=800, background_color="white").generate_from_frequencies(word_freq)






df_text = pd.read_json(r"D:\nltk-dost\data\parsed_text.json", encoding="utf-8")

with open(r"D:\nltk-dost\data\description.json", 'r', encoding='utf-8') as file:
    description  = json.load(file)


for col in df_text.columns[2:]:
    df_text[col] = df_text[col].apply(list)


st.set_page_config(
    page_title="Welcome page",
)

st.sidebar.title("Params selector")
st.sidebar.divider()
st.sidebar.header("Depth")

depth_type_radio = st.sidebar.radio("Depth analysis selection",
                                    ["All", "Parts", "Chapters"],
                                    captions=["parts and chapters agg", "parts agg", "chapters agg"]
                                    )

st.sidebar.divider()
st.sidebar.header("Preprocessing type")
stem_checkbox = st.sidebar.checkbox("Stemming")
lemm_checkbox = st.sidebar.checkbox("Lemmatization")
st.sidebar.divider()
run_checkbox = st.sidebar.checkbox("Run")
comparable_checkbox = st.sidebar.checkbox("Comparable")
st.sidebar.divider()

if "wordclouds_dict" not in st.session_state:
    st.session_state.wordclouds_dict = {}

is_common_preprocessing = not(stem_checkbox) and not(lemm_checkbox)

if is_common_preprocessing:
    col_prep = "text_clean_tokenized"
else:
    col_prep = "text"

if lemm_checkbox:
    col_prep += "_lemm"

if stem_checkbox:
    col_prep += "_stem"

if depth_type_radio=="All":
    frequencies_dict = {}
    frequencies_dict[0] = pd.DataFrame(list(itertools.chain(*df_text[col_prep].apply(list)))).value_counts()
    frequencies_dict[0] = frequencies_dict[0] / frequencies_dict[0].sum()

elif depth_type_radio=="Parts":
    frequencies_dict = {}
    for part in df_text["part"].unique():
        frequencies_dict[part] = pd.DataFrame(list(itertools.chain(*df_text[df_text["part"] == part][col_prep].apply(list)))).value_counts()
        frequencies_dict[part] = frequencies_dict[part] / frequencies_dict[part].sum()
    
else:
    frequencies_dict = {}
    for chapter, part in list(df_text.groupby(["chapter_id", "part"]).size().reset_index()[["chapter_id", "part"]].values):
        frequencies_dict[(chapter, part)] = pd.DataFrame(
            list(
                itertools.chain(*df_text[
                    (df_text["part"] == part) & (df_text["chapter_id"] == chapter)
                    ][col_prep].apply(list)
                    )
                )
            ).value_counts()
        frequencies_dict[(chapter, part)] = frequencies_dict[(chapter, part)] / frequencies_dict[(chapter, part)].sum()

if depth_type_radio == "All":
    wordcloud_id = [0]
    df_freq = frequencies_dict[wordcloud_id[0]]
    if run_checkbox:
        st.title("Word cloud: all")

else:
    st.header("Depth selector")

if depth_type_radio == "Chapters" or depth_type_radio == "Parts":
    parts_selector = st.selectbox("Part selector:", df_text["part"].unique())
    st.divider()
    if depth_type_radio == "Parts":
        wordcloud_id = [parts_selector]
        df_freq = frequencies_dict[wordcloud_id[0]]
        if run_checkbox:
            st.header(f"Word cloud: Part #{parts_selector}")
            st.write("Описание: " + description[str((parts_selector, 1))]["описание_части"])
        
if depth_type_radio == "Chapters":
    chapter_selector = st.selectbox("Chapter selector:", df_text[df_text["part"] == parts_selector]["chapter_id"].unique())
    wordcloud_id = [chapter_selector, parts_selector]
    df_freq = frequencies_dict[tuple(wordcloud_id)]
    if run_checkbox:
        st.divider()
        st.header(f"Word cloud: Part #{parts_selector} Chapter #{chapter_selector}")
        st.write("Описание: " + description[str((parts_selector, 1))][f"глава_{chapter_selector}"])
if (*wordcloud_id, col_prep) in st.session_state.wordclouds_dict:
    wordcloud = st.session_state.wordclouds_dict[(*wordcloud_id, col_prep)]
else:
    wordcloud = build_wordcloud(df_freq)
    st.session_state.wordclouds_dict[(*wordcloud_id, col_prep)] = wordcloud
    print(st.session_state.wordclouds_dict)

if run_checkbox:
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot(plt)
    st.divider()

if run_checkbox and comparable_checkbox:
    st.header("Comparing 2 wordclouds")
    if depth_type_radio == "Parts":
        parts_col_selector = st.columns(2)
        with parts_col_selector[0]:
            left_part = st.selectbox("Left part selector:", df_text["part"].unique())
        
        with parts_col_selector[1]:
            right_part = st.selectbox("Right part selector:", df_text["part"].unique())
        
        cols_parts_wc = st.columns(2)
        for i, part in enumerate([left_part, right_part]):
            with cols_parts_wc[i]:
                st.subheader(f"Part #{part}")
                st.write("Описание: " + description[str((part, 1))]["описание_части"])
                wordcloud_id = [part]
                if (*wordcloud_id, col_prep) in st.session_state.wordclouds_dict:
                    col_wordcloud = st.session_state.wordclouds_dict[(*wordcloud_id, col_prep)]
                else:
                    df_freq = frequencies_dict[wordcloud_id[0]]
                    col_wordcloud = build_wordcloud(df_freq)
                    st.session_state.wordclouds_dict[(*wordcloud_id, col_prep)] = col_wordcloud

                plt.imshow(col_wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.show()
                st.pyplot(plt)
    
    if depth_type_radio == "Chapters":
        st.header(f"Part #{parts_selector}")
        parts_col_selector = st.columns(2)
        with parts_col_selector[0]:
            left_part = st.selectbox("Left part selector:", df_text[df_text["part"] == parts_selector]["chapter_id"].unique())
        
        with parts_col_selector[1]:
            right_part = st.selectbox("Right part selector:", df_text[df_text["part"] == parts_selector]["chapter_id"].unique())

        cols_parts_wc = st.columns(2)
        for i, chapter in enumerate([left_part, right_part]):
            with cols_parts_wc[i]:
                st.subheader(f"Chapter #{chapter}")
                st.write("Описание: " + description[str((parts_selector, 1))][f"глава_{chapter}"])
                wordcloud_id = [chapter, parts_selector]
                if (*wordcloud_id, col_prep) in st.session_state.wordclouds_dict:
                    col_wordcloud = st.session_state.wordclouds_dict[(*wordcloud_id, col_prep)]
                else:
                    df_freq = frequencies_dict[tuple(wordcloud_id)]
                    col_wordcloud = build_wordcloud(df_freq)
                    st.session_state.wordclouds_dict[(*wordcloud_id, col_prep)] = col_wordcloud

                plt.imshow(col_wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.show()
                st.pyplot(plt)