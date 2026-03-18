import spacy
pln = spacy.load("es_core_news_sm")
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    texto = texto.strip()
    return texto


def lematizar_por_lotes(series_textos):
    """
    Recibe una pandas Series de textos ya limpios y devuelve otra Series
    con los lemas (sin stopwords) unidos por espacios.
    """
    # Aseguramos strings y mantenemos el índice original
    textos = series_textos.fillna("").astype(str).tolist()
    idx = series_textos.index

    salida = []
    for doc in pln.pipe(textos, batch_size=300, n_process=4):
        lemas = [
            t.lemma_
            for t in doc
            if not t.is_space and not t.is_punct and not t.is_stop
        ]
        salida.append(" ".join(lemas))

    return pd.Series(salida, index=idx, name=getattr(series_textos, "name", None))


def get_tokens_lemas(texto):
    doc = pln(texto)
    output = [token.lemma_ for token in doc if not token.is_stop]
    output = ' '.join(output)
    return output

def pln_pipeline(texto):
    output = limpiar_texto(texto)
    output = get_tokens_lemas(output)
    return output

def vectorizar_bow(textos, **vectorizer_kwargs):
    cv = CountVectorizer(**vectorizer_kwargs)
    dtm = cv.fit_transform(textos)

    dtm_df = pd.DataFrame(
        dtm.toarray(),
        columns=cv.get_feature_names_out(),
        index=textos.index if isinstance(textos, pd.Series) else None
    )
    return dtm_df, cv


from sklearn.feature_extraction.text import TfidfVectorizer


def vectorizar_tfidf(textos, **vectorizer_kwargs):
    tf = TfidfVectorizer(**vectorizer_kwargs)
    tfidf = tf.fit_transform(textos)

    tfidf_df = pd.DataFrame(
        tfidf.toarray(),
        columns=tf.get_feature_names_out(),
        index=textos.index if isinstance(textos, pd.Series) else None
    )
    return tfidf_df, tf
