"""
Análisis Exploratorio NLP — Turismo México
==========================================
Lee   : data/final/maestro_nlp.csv
Escribe: data/reports/  (PNGs de cada figura)

Qué genera este script
-----------------------
1. Nube de palabras global
2. Nubes de palabras por ubicación (5 destinos)
3. Frecuencia de unigramas, bigramas y trigramas (global)
4. N-gramas por ubicación (top 10 bigramas cada uno)
5. Ranking TF-IDF por ubicación (qué palabras son DISTINTIVAS
   de cada destino, no solo frecuentes)

Diferencia entre frecuencia y TF-IDF
--------------------------------------
  Frecuencia: "playa" aparece mucho → es importante globalmente
  TF-IDF    : "balandra" aparece mucho EN La Paz pero poco en otros
               destinos → es CARACTERÍSTICA de La Paz

  Para el dashboard, TF-IDF te da las palabras que realmente
  distinguen a cada destino turístico.

Uso
---
  python src/exploratory_analysis.py
  python src/exploratory_analysis.py --location lapaz  # solo un destino
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # sin GUI, genera archivos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
INPUT_CSV   = BASE_DIR / "data" / "final" / "maestro_nlp.csv"
REPORTS_DIR = BASE_DIR / "plots" / "reports"

# ── Stopwords extra ───────────────────────────────────────────────────────────
# El stemmer dejó pasar algunas palabras funcionales o stems sin sentido.
# Las agregamos aquí para limpiar wordcloud y n-gramas.
EXTRA_STOPS = {
    # Artículos / preposiciones que escaparon el stemmer
    "del", "al", "los", "las", "una", "uno",
    # Stems de palabras muy genéricas que no aportan topics turísticos
    "hac",   # hacer
    "quier", # querer
    "pued",  # poder
    "dej",   # dejar
    "vol",   # volver
    "mej",   # mejor (ya tenemos "mejor" sin stemming)
    "nad",   # nadar — relevante en playas, lo dejamos
    # Tokens numéricos que escaparon
    "10", "20", "30", "100", "150", "200",
}

# ── Paleta de colores por destino ─────────────────────────────────────────────
LOCATION_COLORS = {
    "La Paz":           "#1a6faf",
    "Puerto Vallarta":  "#e05c2f",
    "Rivera Maya":      "#2e8b57",
    "Riviera Nayarit":  "#7b4fa6",
    "Huatulco":         "#c8960c",
}

# ── Estilo global de matplotlib ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "x",
    "grid.alpha":         0.3,
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.dpi":        150,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save(fig: plt.Figure, name: str) -> None:
    path = REPORTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  → {path.name}")


def get_ngrams(corpus: list[str], n: int, top_k: int = 20,
               extra_stops: set = None) -> list[tuple]:
    """
    Devuelve los top_k n-gramas más frecuentes del corpus.

    Parámetros
    ----------
    corpus     : lista de strings (comentario_nlp)
    n          : 1=unigramas, 2=bigramas, 3=trigramas
    top_k      : cuántos retornar
    extra_stops: tokens adicionales a ignorar (para limpiar stems sin sentido)
    """
    vec = CountVectorizer(
        ngram_range=(n, n),
        max_features=top_k * 3,    # pedimos más y filtramos después
    )
    X = vec.fit_transform(corpus)
    freq = zip(vec.get_feature_names_out(), X.toarray().sum(axis=0))
    freq = sorted(freq, key=lambda x: -x[1])

    # Filtrar extra_stops si las hay (aplica solo a unigramas)
    if extra_stops and n == 1:
        freq = [(w, c) for w, c in freq if w not in extra_stops]

    return freq[:top_k]


def get_tfidf(df_sub: pd.DataFrame, df_all: pd.DataFrame,
              top_k: int = 15) -> list[tuple]:
    """
    Calcula TF-IDF usando el grupo (df_sub) como documentos a rankear
    pero el vocabulario de todo el corpus (df_all) como referencia.

    Cada 'documento' en TF-IDF aquí es el conjunto de comentarios de
    una ubicación. Usamos un CountVectorizer por ubicación:
      - TF = frecuencia del término en la ubicación
      - IDF = inverso de en cuántas ubicaciones aparece el término

    Retorna los top_k términos más representativos de la ubicación.
    """
    # Construir un 'documento' por ubicación (concatenar todos los textos)
    ubicaciones = df_all["ubicacion"].unique()
    docs_by_loc = {
        loc: " ".join(df_all[df_all["ubicacion"] == loc]["comentario_nlp"].fillna(""))
        for loc in ubicaciones
    }

    # TF-IDF con un documento por ubicación
    vec = TfidfVectorizer(
        max_features=500,
        min_df=1,          # aparece en al menos 1 documento (ubicación)
    )
    tfidf_matrix = vec.fit_transform(list(docs_by_loc.values()))
    feature_names = vec.get_feature_names_out()

    # Encontrar el índice de la ubicación que nos interesa
    target_loc = df_sub["ubicacion"].iloc[0]
    loc_idx = list(docs_by_loc.keys()).index(target_loc)

    # Scores TF-IDF de esa ubicación
    scores = zip(feature_names, tfidf_matrix[loc_idx].toarray()[0])
    scores = sorted(scores, key=lambda x: -x[1])
    return scores[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NUBE DE PALABRAS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wordcloud_global(corpus: list[str]) -> None:
    """
    Nube de palabras con todos los destinos combinados.
    Tamaño de la palabra ∝ frecuencia en el corpus.
    """
    log.info("[1/5] Generando nube de palabras global...")
    text = " ".join(corpus)

    wc = WordCloud(
        width=1400, height=700,
        background_color="white",
        max_words=120,
        colormap="viridis",
        stopwords=EXTRA_STOPS,
        collocations=False,         # evita bigramas automáticos, queremos unigramas
        prefer_horizontal=0.85,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nube de palabras — todos los destinos", fontsize=16, pad=16)
    save(fig, "01_wordcloud_global.png")


def plot_wordclouds_por_ubicacion(df: pd.DataFrame) -> None:
    """
    Una nube por cada destino, en un grid 2x3.
    Cada nube usa el color representativo del destino.
    """
    log.info("[2/5] Generando nubes de palabras por ubicación...")

    ubicaciones = df["ubicacion"].unique()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, loc in enumerate(ubicaciones):
        corpus_loc = df[df["ubicacion"] == loc]["comentario_nlp"].fillna("").tolist()
        text = " ".join(corpus_loc)
        color = LOCATION_COLORS.get(loc, "#555555")

        # Colormap personalizado para que la nube use el color del destino
        def make_color_func(base_color):
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(base_color)
            def color_func(word, font_size, position, orientation,
                           random_state=None, **kwargs):
                # Mezclamos el color base con blanco según tamaño de fuente
                # factor=1 → color base puro, factor=0 → blanco
                factor = max(0.3, min(1.0, font_size / 80))
                r = int(max(0, min(255, rgb[0] * 255 * factor + 255 * (1 - factor))))
                g = int(max(0, min(255, rgb[1] * 255 * factor + 255 * (1 - factor))))
                b = int(max(0, min(255, rgb[2] * 255 * factor + 255 * (1 - factor))))
                return f"rgb({r},{g},{b})"
            return color_func

        wc = WordCloud(
            width=700, height=380,
            background_color="white",
            max_words=80,
            stopwords=EXTRA_STOPS,
            collocations=False,
            color_func=make_color_func(color),
            prefer_horizontal=0.8,
        ).generate(text)

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(loc, fontsize=13, color=color, fontweight="bold")

    # Ocultar subplots vacíos (si los hay)
    for j in range(len(ubicaciones), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Nube de palabras por destino turístico", fontsize=16, y=1.01)
    fig.tight_layout()
    save(fig, "02_wordclouds_por_ubicacion.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FRECUENCIA DE N-GRAMAS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ngrams_global(corpus: list[str]) -> None:
    """
    3 paneles en una figura:
      - Top 20 unigramas
      - Top 20 bigramas
      - Top 15 trigramas
    Barras horizontales para que los textos sean legibles.
    """
    log.info("[3/5] Generando frecuencias de n-gramas globales...")

    uni  = get_ngrams(corpus, 1, 20, EXTRA_STOPS)
    bi   = get_ngrams(corpus, 2, 20)
    tri  = get_ngrams(corpus, 3, 15)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for ax, data, title, color in [
        (axes[0], uni,  "Unigramas", "#1a6faf"),
        (axes[1], bi,   "Bigramas",  "#e05c2f"),
        (axes[2], tri,  "Trigramas", "#2e8b57"),
    ]:
        words  = [w for w, _ in reversed(data)]
        counts = [c for _, c in reversed(data)]
        bars = ax.barh(words, counts, color=color, alpha=0.75, height=0.65)
        ax.set_title(f"Top {len(data)} {title}", fontsize=13, pad=10)
        ax.set_xlabel("Frecuencia")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        # Etiqueta de valor al final de cada barra
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{cnt:,}", va="center", fontsize=8.5)
        ax.set_xlim(0, max(counts) * 1.14)

    fig.suptitle("Frecuencia de n-gramas — corpus completo (38 374 comentarios)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "03_ngrams_global.png")


def plot_bigramas_por_ubicacion(df: pd.DataFrame) -> None:
    """
    Top 10 bigramas por cada destino, en un grid.
    Permite comparar qué temas dominan en cada lugar.
    """
    log.info("[4/5] Generando bigramas por ubicación...")

    ubicaciones = df["ubicacion"].unique()
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, loc in enumerate(ubicaciones):
        corpus_loc = df[df["ubicacion"] == loc]["comentario_nlp"].fillna("").tolist()
        bigrams    = get_ngrams(corpus_loc, 2, 10)
        color      = LOCATION_COLORS.get(loc, "#555555")

        words  = [w for w, _ in reversed(bigrams)]
        counts = [c for _, c in reversed(bigrams)]

        bars = axes[i].barh(words, counts, color=color, alpha=0.75, height=0.6)
        axes[i].set_title(loc, fontsize=12, color=color, fontweight="bold")
        axes[i].set_xlabel("Frecuencia")
        for bar, cnt in zip(bars, counts):
            axes[i].text(bar.get_width() + max(counts) * 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f"{cnt:,}", va="center", fontsize=8)
        axes[i].set_xlim(0, max(counts) * 1.16)

    for j in range(len(ubicaciones), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Top 10 bigramas por destino turístico", fontsize=15, y=1.01)
    fig.tight_layout()
    save(fig, "04_bigramas_por_ubicacion.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TF-IDF
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tfidf_por_ubicacion(df: pd.DataFrame) -> None:
    """
    Palabras MÁS CARACTERÍSTICAS de cada destino según TF-IDF.

    Diferencia con frecuencia pura:
      - Frecuencia: "playa" sale top en todos → no distingue nada
      - TF-IDF: "balandra" sale top en La Paz → SÍ distingue

    Esto es lo más útil para el dashboard: muestra la "identidad"
    léxica de cada destino.
    """
    log.info("[5/5] Generando ranking TF-IDF por ubicación...")

    ubicaciones = df["ubicacion"].unique()
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, loc in enumerate(ubicaciones):
        df_loc = df[df["ubicacion"] == loc]
        scores = get_tfidf(df_loc, df, top_k=12)
        color  = LOCATION_COLORS.get(loc, "#555555")

        words  = [w for w, _ in reversed(scores)]
        vals   = [v for _, v in reversed(scores)]

        bars = axes[i].barh(words, vals, color=color, alpha=0.8, height=0.6)
        axes[i].set_title(loc, fontsize=12, color=color, fontweight="bold")
        axes[i].set_xlabel("Score TF-IDF")
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_width() + max(vals) * 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f"{val:.4f}", va="center", fontsize=8)
        axes[i].set_xlim(0, max(vals) * 1.18)

    for j in range(len(ubicaciones), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Palabras más características por destino — TF-IDF\n"
                 "(qué términos distinguen a cada lugar del resto)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "05_tfidf_por_ubicacion.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main(location_filter: str = None) -> None:
    log.info("═" * 55)
    log.info("  ANÁLISIS EXPLORATORIO NLP")
    log.info("═" * 55)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cargar ───────────────────────────────────────────────────────────────
    log.info(f"Leyendo {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV, dtype=str)
    df["es_spam"]      = df["es_spam"].map({"True": True, "False": False}).fillna(False)
    df["n_palabras_nlp"] = pd.to_numeric(df["n_palabras_nlp"], errors="coerce").fillna(0)

    # Filtrar spam y textos vacíos
    df = df[~df["es_spam"]].copy()
    df["comentario_nlp"] = df["comentario_nlp"].fillna("")
    df = df[df["comentario_nlp"].str.strip() != ""]

    if location_filter:
        df = df[df["ubicacion"].str.lower().str.replace(" ", "_") == location_filter]
        log.info(f"Filtrado a: {df['ubicacion'].unique()}")

    log.info(f"  {len(df):,} comentarios útiles")
    log.info(f"  Destinos: {df['ubicacion'].nunique()}")
    log.info(f"  Fuentes:  {df['fuente'].value_counts().to_dict()}")

    corpus = df["comentario_nlp"].tolist()

    # ── Ejecutar análisis ────────────────────────────────────────────────────
    plot_wordcloud_global(corpus)
    plot_wordclouds_por_ubicacion(df)
    plot_ngrams_global(corpus)
    plot_bigramas_por_ubicacion(df)
    plot_tfidf_por_ubicacion(df)

    log.info("\n" + "═" * 55)
    log.info(f"  ✓ {len(list(REPORTS_DIR.glob('*.png')))} figuras guardadas en data/reports/")
    log.info("═" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default=None,
                        help="Filtrar a un solo destino, ej: lapaz")
    args = parser.parse_args()
    main(location_filter=args.location)