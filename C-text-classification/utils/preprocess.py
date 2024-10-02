import re

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize
from pandarallel import pandarallel


def sliding_window_chunks(text, max_words=275, min_words=50):
    sentences = sent_tokenize(text, "russian")
    chunks = []
    current_chunk = []
    start_idx = 0

    while start_idx < len(sentences):
        current_chunk = []
        current_word_count = 0
        for idx in range(start_idx, len(sentences)):
            sentence_word_count = len(sentences[idx].split())

            if current_word_count + sentence_word_count > max_words:
                break
            current_chunk.append(sentences[idx])
            current_word_count += sentence_word_count

        if current_word_count >= min_words:
            chunks.append(" ".join(current_chunk))

        start_idx += 1
    return chunks


def get_chunked_dataframe(df):
    df = df.copy()
    chunked_rows = []

    for _, row in df.iterrows():
        if pd.notna(row["text"]):
            text_chunks = sliding_window_chunks(row["text"])
            for chunk in text_chunks:
                chunked_rows.append([row["movie"], chunk, row["genres"]])
    return pd.DataFrame(chunked_rows, columns=["movie", "text", "genres"])


def clear_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\xa0", " ")
    return " ".join(text.split())


def replace_entities(text, nlp):
    doc = nlp(text)
    modified_text = text
    for ent in doc.ents:
        if ent.label_ == "PER":
            modified_text = modified_text.replace(ent.text, "герой")
        elif ent.label_ == "LOC":
            modified_text = modified_text.replace(ent.text, "локация")
    return modified_text


def remove_film_name_from_text(text, film_title):

    film_title_clean = re.sub(r"\(.*?\)", "", film_title).strip()
    text = text.replace(film_title, "фильм")
    text = text.replace(film_title_clean, "фильм")
    return text


def preprocess_train(df: pd.DataFrame):
    df = df.copy()
    pandarallel.initialize(progress_bar=True, nb_workers=4)
    nlp = spacy.load("ru_core_news_lg")
    df = df.rename(
        {
            "Фильм": "movie",
            "Сюжет": "plot",
            "Жанры": "genres",
            "Описание": "description",
        },
        axis=1,
    )
    print(df.isnull().any())
    df_description = (
        df[["movie", "description", "genres"]]
        .rename(columns={"description": "text"})
        .dropna(subset=["text"])
    )
    df_plot = (
        df[["movie", "plot", "genres"]]
        .rename(columns={"plot": "text"})
        .dropna(subset=["text"])
    )
    df_doubled = pd.concat([df_description, df_plot], ignore_index=True)
    df_doubled["text"] = df_doubled["text"].apply(clear_text)
    df_doubled["text"] = df_doubled["text"].parallel_apply(
        lambda x: replace_entities(x, nlp)
    )
    df_doubled["text"] = df_doubled["text"].apply(clear_text)
    df_doubled["text"] = df_doubled.apply(
        lambda x: remove_film_name_from_text(x["text"], x["movie"]), axis=1
    )
    df_doubled["text"] = df_doubled["text"].apply(clear_text)
    df_doubled["text"].apply(lambda x: len(x.split())).hist(bins=100)
    plt.show()
    return df
