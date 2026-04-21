"""
Preprocesamiento NLP — Turismo México
======================================
Lee   : data/final/maestro_limpio.csv
Escribe: data/final/maestro_nlp.csv

Por qué este script es DISTINTO al text_cleaner.py
----------------------------------------------------
text_cleaner.py hizo limpieza de "ruido": URLs, emojis, encoding roto,
spam. El texto resultante (comentario_limpio) es bueno para BERTopic porque
los transformers necesitan texto natural y completo.

Este script hace preprocesamiento "lingüístico" para las técnicas clásicas
de NLP que trabajan con frecuencias de palabras:

  Técnica                | Necesita           | Columna a usar
  -----------------------|--------------------|------------------
  BERTopic               | texto natural      | comentario_limpio
  Word cloud             | sin stopwords      | comentario_nlp
  Bag of Words           | sin stopwords      | comentario_nlp
  TF-IDF                 | sin stopwords      | comentario_nlp
  N-gramas               | sin stopwords      | comentario_nlp
  LDA                    | sin stopwords      | comentario_nlp
  Clustering (K-Means)   | puede usar ambas   | comentario_limpio

Columnas que agrega este script
--------------------------------
  idioma_det   : idioma detectado con langdetect ('es','en','otro')
                 Reemplaza la columna 'idioma' que usaba heurística débil
  comentario_nlp : texto listo para BoW/TF-IDF/n-gramas/wordcloud
                   = comentario_limpio
                   - stopwords en español e inglés
                   + stemming con Snowball
  n_palabras_nlp : longitud de comentario_nlp (post-stopwords)

Instalaciones necesarias (una sola vez en tu entorno)
-------------------------------------------------------
  pip install langdetect ftfy
  # Para lemmatización real (mejor que stemming, requiere descarga):
  pip install spacy
  python -m spacy download es_core_news_md
  python -m spacy download en_core_web_sm

Stemming vs Lematización — diferencia clave
--------------------------------------------
  Stemming   : corta el final de la palabra por regla.
               Rápido, no necesita modelo.
               "visitando" → "visit", "playas" → "play"
               Puede producir no-palabras.

  Lematización: mapea a la forma canónica del diccionario.
               Necesita modelo lingüístico (spaCy).
               "visitando" → "visitar", "playas" → "playa"
               Más preciso, mucho más lento.

  Para este dataset (48k docs, mezcla ES/EN, textos cortos),
  stemming es suficiente para BoW/TF-IDF/n-gramas.
  Si quieres lematización, descomenta el bloque de spaCy al final.

Uso
---
  python src/nlp_preprocessor.py
  python src/nlp_preprocessor.py --dry-run     # sin guardar
  python src/nlp_preprocessor.py --sample 500  # probar con muestra
"""

import re
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
from langdetect import detect, LangDetectException
from nltk.stem import SnowballStemmer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
FINAL_DIR  = BASE_DIR / "data" / "final"
INPUT_CSV  = FINAL_DIR / "maestro_limpio.csv"
OUTPUT_CSV = FINAL_DIR / "maestro_nlp.csv"

# ── Stemmers ──────────────────────────────────────────────────────────────────
STEMMER_ES = SnowballStemmer("spanish")
STEMMER_EN = SnowballStemmer("english")

# ── Palabras mínimas para considerar el texto (post-stopwords) ────────────────
MIN_PALABRAS_NLP = 3


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STOPWORDS — ESPAÑOL + INGLÉS
# ═══════════════════════════════════════════════════════════════════════════════
# Lista manual extensa porque nltk no está disponible en este entorno.
# En tu máquina puedes reemplazar con:
#   from nltk.corpus import stopwords
#   STOPWORDS_ES = set(stopwords.words("spanish"))
#   STOPWORDS_EN = set(stopwords.words("english"))

STOPWORDS_ES = {
    # Artículos
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # Preposiciones
    "a", "ante", "bajo", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por", "según",
    "sin", "sobre", "tras", "versus",
    # Conjunciones
    "y", "e", "ni", "o", "u", "pero", "sino", "aunque", "porque",
    "pues", "que", "si", "como", "cuando", "donde", "mientras",
    # Pronombres
    "yo", "tu", "tú", "el", "ella", "nosotros", "vosotros", "ellos",
    "ellas", "me", "te", "se", "nos", "os", "le", "les", "lo", "la",
    "mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra",
    "nuestros", "nuestras", "este", "esta", "estos", "estas", "ese",
    "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas",
    "esto", "eso", "aquello",
    # Verbos auxiliares y copulativos muy frecuentes
    "es", "son", "era", "fue", "ser", "estar", "tiene", "tener",
    "hay", "había", "haber", "hace", "hacer", "han", "ha", "he",
    "sido", "estado", "tenido", "hecho", "va", "ir",
    # Adverbios comunes
    "no", "sí", "si", "también", "tampoco", "solo", "sólo", "muy",
    "más", "menos", "bien", "mal", "aquí", "ahí", "allí", "allá",
    "ahora", "antes", "después", "ya", "aún", "todavía", "siempre",
    "nunca", "jamás", "casi", "bastante", "demasiado", "tan", "tanto",
    "así", "además",
    # Determinantes y cuantificadores
    "todo", "toda", "todos", "todas", "mucho", "mucha", "muchos",
    "muchas", "poco", "poca", "pocos", "pocas", "otro", "otra",
    "otros", "otras", "mismo", "misma", "mismos", "mismas",
    "cada", "cual", "cuales", "quien", "quienes", "que",
    # Números como texto
    "uno", "dos", "tres", "cuatro", "cinco", "primer", "primera",
    "segundo", "segunda",
    # Palabras muy genéricas en reseñas turísticas (no aportan a topics)
    "lugar", "lugares", "vez", "veces", "día", "días", "año", "años",
    "cosa", "cosas", "parte", "partes", "tipo", "tipos", "forma",
    "manera", "momento", "momentos",
}

STOPWORDS_EN = {
    # Articles
    "a", "an", "the",
    # Prepositions
    "in", "on", "at", "by", "for", "with", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "from",
    "to", "of", "up", "down", "out", "off", "over", "under",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either",
    "neither", "not", "only", "whether", "because", "if", "when",
    "while", "although", "though", "since", "unless",
    # Pronouns
    "i", "you", "he", "she", "we", "they", "it", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "this", "that", "these", "those",
    # Auxiliaries
    "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    # Common adverbs
    "very", "really", "also", "just", "more", "most", "much",
    "well", "even", "still", "already", "now", "here", "there",
    "always", "never", "too", "quite",
    # Generic review words (no topic value)
    "place", "time", "day", "year", "thing", "way", "lot",
    "little", "big", "good", "great", "nice", "one", "two",
}

# Unión para textos mixtos
STOPWORDS_TODAS = STOPWORDS_ES | STOPWORDS_EN


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DETECCIÓN DE IDIOMA
# ═══════════════════════════════════════════════════════════════════════════════

def detectar_idioma(texto: str) -> str:
    """
    Detecta el idioma de un texto con langdetect.

    langdetect es un port de la librería de Google, usa modelos estadísticos
    de n-gramas de caracteres. Funciona bien incluso con texto corto y
    sin puntuación (como el texto pre-tokenizado de Trip).

    Retorna: 'es', 'en', u 'otro'

    Manejo de casos especiales:
    - Textos muy cortos (< 10 caracteres): langdetect lanza excepción o
      da resultados poco confiables → retornamos 'otro'
    - Lenguas como 'ca' (catalán), 'pt' (portugués), 'it' (italiano)
      se colapsan a 'otro' porque son muy pocos en este dataset
    """
    if not isinstance(texto, str) or len(texto.strip()) < 10:
        return "otro"
    try:
        lang = detect(texto)
        if lang == "es":
            return "es"
        elif lang == "en":
            return "en"
        else:
            return "otro"
    except LangDetectException:
        return "otro"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESAMIENTO NLP (stopwords + stemming)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocesar_nlp(texto: str, idioma: str) -> str:
    """
    Convierte comentario_limpio en comentario_nlp:
      1. Tokeniza por espacios
      2. Elimina stopwords según idioma detectado
      3. Aplica stemming con Snowball según idioma
      4. Devuelve tokens reunidos en string

    Por qué tokenizamos por espacio (no nltk word_tokenize):
    El texto ya viene limpio de puntuación. Tokenizar por espacio es
    equivalente y no requiere modelos descargados.

    Por qué stemming y no lematización:
    Lematización (spaCy) necesita el modelo descargado y es ~20x más lenta.
    Para BoW/TF-IDF/n-gramas, stemming da resultados comparables.
    Si quieres lematización: ver bloque comentado al final del archivo.

    Ejemplo:
      entrada : "hermosas playas visitando restaurantes locales comida"
      stopwords: set()  ← ninguna de estas es stopword
      stemming : "herm play visit restaur local comid"
      ← el stemmer produce raíces, no palabras completas
    """
    if not isinstance(texto, str) or texto.strip() == "":
        return ""

    tokens = texto.lower().split()

    # Elegir stopwords según idioma
    if idioma == "es":
        stops = STOPWORDS_ES
        stemmer = STEMMER_ES
    elif idioma == "en":
        stops = STOPWORDS_EN
        stemmer = STEMMER_EN
    else:
        # Para "otro" usamos el conjunto completo y el stemmer español
        # (mayoría del corpus es español aunque langdetect no lo detecte)
        stops = STOPWORDS_TODAS
        stemmer = STEMMER_ES

    # Eliminar stopwords + aplicar stemming en un solo pass
    tokens_procesados = [
        stemmer.stem(token)
        for token in tokens
        if token not in stops          # quitar stopword
        and len(token) > 2             # quitar tokens de 1-2 chars ("al", "ha")
        and not token.isdigit()        # quitar números solos ("150", "2024")
    ]

    return " ".join(tokens_procesados)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def procesar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica detección de idioma y preprocesamiento NLP al DataFrame.

    Trabaja sobre los registros NO spam (es_spam == False).
    Los spam conservan sus columnas pero con valores vacíos.
    """
    total = len(df)
    utiles = (~df["es_spam"]).sum()
    log.info(f"  Procesando {utiles:,} registros útiles de {total:,} totales")
    log.info("  (los spam se omiten del procesamiento NLP)")

    # ── Detección de idioma (reemplaza heurística anterior) ───────────────────
    log.info("\n  [1/3] Detectando idioma con langdetect...")
    # Solo detectamos en no-spam para ahorrar tiempo
    mask_util = ~df["es_spam"]
    df["idioma_det"] = "otro"
    df.loc[mask_util, "idioma_det"] = (
        df.loc[mask_util, "comentario_limpio"].apply(detectar_idioma)
    )

    dist = df.loc[mask_util, "idioma_det"].value_counts()
    for lang, cnt in dist.items():
        log.info(f"    {lang:<6} {cnt:>6,}  ({cnt/utiles*100:.1f}%)")

    # ── Stopwords + stemming → comentario_nlp ─────────────────────────────────
    log.info("\n  [2/3] Aplicando stopwords + stemming → comentario_nlp...")
    df["comentario_nlp"] = ""
    df.loc[mask_util, "comentario_nlp"] = df.loc[mask_util].apply(
        lambda row: preprocesar_nlp(row["comentario_limpio"], row["idioma_det"]),
        axis=1,
    )

    # ── Conteo de palabras post-stopwords ─────────────────────────────────────
    log.info("\n  [3/3] Calculando longitudes post-stopwords...")
    df["n_palabras_nlp"] = df["comentario_nlp"].str.split().str.len().fillna(0).astype(int)

    # Marcar como spam los que quedaron demasiado cortos DESPUÉS de stopwords
    # (comentarios que eran básicamente puros artículos/preposiciones)
    quedaron_vacios = (mask_util) & (df["n_palabras_nlp"] < MIN_PALABRAS_NLP)
    if quedaron_vacios.sum() > 0:
        log.info(f"\n  ⚠ {quedaron_vacios.sum()} registros quedaron con < {MIN_PALABRAS_NLP} "
                 f"palabras tras stopwords → marcados como spam")
        df.loc[quedaron_vacios, "es_spam"] = True
        df.loc[quedaron_vacios, "comentario_nlp"] = ""

    return df


def imprimir_estadisticas(df: pd.DataFrame) -> None:
    utiles = df[~df["es_spam"]]
    log.info("\n" + "═" * 55)
    log.info("  RESUMEN FINAL")
    log.info("═" * 55)
    log.info(f"  Total registros      : {len(df):>7,}")
    log.info(f"  Útiles para NLP      : {len(utiles):>7,}  ({len(utiles)/len(df)*100:.1f}%)")
    log.info(f"  Descartados (spam)   : {df['es_spam'].sum():>7,}")

    log.info("\n  Idioma (registros útiles):")
    for lang, cnt in utiles["idioma_det"].value_counts().items():
        log.info(f"    {lang:<6}  {cnt:>6,}  ({cnt/len(utiles)*100:.1f}%)")

    log.info("\n  Palabras por fuente (comentario_nlp, sin spam):")
    log.info(
        utiles.groupby("fuente")["n_palabras_nlp"]
              .agg(["mean", "median"])
              .round(1)
              .to_string()
    )

    log.info("\n  Ejemplo ANTES → DESPUÉS del preprocesamiento NLP:")
    ejemplos = utiles[utiles["idioma_det"] == "es"].sample(3, random_state=7)
    for _, row in ejemplos.iterrows():
        log.info(f"\n  [{row['fuente'].upper()} / {row['ubicacion']}]")
        log.info(f"  LIMPIO : {row['comentario_limpio'][:100]}")
        log.info(f"  NLP    : {row['comentario_nlp'][:100]}")

    log.info("\n  Columnas en el archivo de salida:")
    log.info("    comentario      → texto raw original (nunca se toca)")
    log.info("    comentario_limpio → para BERTopic (transformers)")
    log.info("    comentario_nlp    → para BoW, TF-IDF, n-gramas, word cloud")
    log.info("    idioma_det        → detección real con langdetect")
    log.info("    n_palabras_nlp    → longitud post-stopwords")
    log.info("    es_spam           → filtro principal (usa df[~df.es_spam])")


def main(dry_run: bool = False, sample: int = None) -> None:
    log.info("═" * 55)
    log.info("  PREPROCESAMIENTO NLP — TURISMO MÉXICO")
    log.info("═" * 55)

    if not INPUT_CSV.exists():
        log.error(f"No encontrado: {INPUT_CSV}")
        sys.exit(1)

    log.info(f"Leyendo {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV, dtype=str)
    df["rating"]    = pd.to_numeric(df["rating"],    errors="coerce")
    df["n_palabras"] = pd.to_numeric(df["n_palabras"], errors="coerce")
    df["es_spam"]   = df["es_spam"].map({"True": True, "False": False}).fillna(False)
    log.info(f"  {len(df):,} registros cargados")

    if sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
        log.info(f"  [--sample] Usando muestra de {sample} registros")

    df = procesar(df)
    imprimir_estadisticas(df)

    if dry_run:
        log.info("\n  [--dry-run] No se guardó ningún archivo.")
        return

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info(f"\n  → Guardado: {OUTPUT_CSV.relative_to(BASE_DIR)}")
    log.info("═" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", type=int, default=None,
                        help="Procesar solo N filas (para pruebas rápidas)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, sample=args.sample)


# ═══════════════════════════════════════════════════════════════════════════════
# APÉNDICE: LEMATIZACIÓN CON SPACY (descomenta cuando tengas el modelo)
# ═══════════════════════════════════════════════════════════════════════════════
# Instalar una sola vez:
#   pip install spacy
#   python -m spacy download es_core_news_md
#   python -m spacy download en_core_web_sm
#
# Reemplaza preprocesar_nlp() con esta versión:
#
# import spacy
# NLP_ES = spacy.load("es_core_news_md", disable=["parser", "ner"])
# NLP_EN = spacy.load("en_core_web_sm",  disable=["parser", "ner"])
#
# def preprocesar_nlp_spacy(texto: str, idioma: str) -> str:
#     nlp = NLP_ES if idioma in ("es", "otro") else NLP_EN
#     doc = nlp(texto)
#     tokens = [
#         token.lemma_.lower()
#         for token in doc
#         if not token.is_stop          # stopwords del modelo
#         and not token.is_punct
#         and not token.is_space
#         and not token.like_num
#         and len(token.lemma_) > 2
#     ]
#     return " ".join(tokens)
#
# Ventajas sobre stemming:
#   "visitando" → "visitar"  (no "visit")
#   "playas"    → "playa"    (no "play")
#   "mejores"   → "bueno"    (reconoce forma comparativa)
# ═══════════════════════════════════════════════════════════════════════════════