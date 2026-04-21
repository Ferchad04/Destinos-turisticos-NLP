"""
Pipeline de Limpieza de Texto — Turismo México
===============================================
Lee:   data/final/maestro_completo.csv
Escribe: data/final/maestro_limpio.csv

Qué hace este script
--------------------
1. Corrige artefactos de encoding (Ã© → é) que quedaron en los archivos Maps
   que estaban en latin-1 pero se leyeron como si fueran UTF-8.

2. Limpia el texto para NLP:
   - Elimina URLs, menciones (@usuario) y hashtags (#palabra)
   - Elimina emojis y caracteres no alfanuméricos que no aporten significado
   - Normaliza espacios múltiples y saltos de línea
   - Convierte a minúsculas

3. Detecta y marca spam (especialmente en Instagram):
   - Publicaciones comerciales (precios, "DM", "envío", etc.)
   - Comentarios que son casi puro hashtag
   - Comentarios demasiado cortos para dar información útil

4. Agrega columnas nuevas SIN borrar el texto original:
   - comentario_limpio  : texto después de limpieza
   - n_palabras         : conteo de palabras del texto limpio
   - es_spam            : True/False para filtrar en pasos siguientes
   - idioma             : 'es', 'en', u 'otro' (detección simple por heurística)

Por qué NO lematizamos / stemming aquí
---------------------------------------
BERTopic usa embeddings de transformers (sentence-transformers).
Esos modelos fueron entrenados con texto natural completo, no con raíces.
Lematizar "hermosas playas cristalinas" → "hermoso playa cristalino"
destruye el contexto y reduce la calidad de los embeddings.
La limpieza aquí es solo para quitar ruido, no para reducir vocabulario.

Uso
---
  python src/text_cleaner.py

  Opcional — ver estadísticas sin guardar:
  python src/text_cleaner.py --dry-run
"""

import re
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import ftfy                  # pip install ftfy  — corrige encoding corrupto automáticamente

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
INPUT_CSV  = FINAL_DIR / "maestro_completo.csv"
OUTPUT_CSV = FINAL_DIR / "maestro_limpio.csv"

# ── Parámetros de filtrado ────────────────────────────────────────────────────
MIN_PALABRAS = 5    # Comentarios con menos palabras se marcan como spam.
                    # BERTopic necesita suficiente texto para crear embeddings
                    # significativos. 5 palabras es el mínimo razonable.

MAX_PALABRAS = 500  # Comentarios extremadamente largos suelen ser copiar-pegar
                    # o texto automático. Podría filtrarse o truncarse.

# Ratio máximo de hashtags vs palabras totales en un comentario.
# Ej: "🏖️ #playa #mexico #beach #travel #vacation" → 5/6 = 0.83 → spam
MAX_RATIO_HASHTAGS = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CORRECCIÓN DE ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def corregir_encoding(texto: str) -> str:
    """
    Corrige artefactos comunes cuando un archivo latin-1 se leyó como UTF-8.

    Ejemplos reales del dataset:
      'autÃ©ntica'     → 'auténtica'
      'energÃ\\xada'   → 'energía'
      'distingue'      → 'distingue'  (sin cambio si ya está bien)

    Usa la librería `ftfy` (Fix Text For You) que maneja estos casos
    automáticamente, incluyendo variantes raras. No modifica texto que
    ya está bien codificado.
    """
    if not isinstance(texto, str):
        return ""
    return ftfy.fix_text(texto)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LIMPIEZA DE TEXTO
# ═══════════════════════════════════════════════════════════════════════════════

# Patrón para detectar emojis (rangos Unicode de bloques de emojis comunes)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticonos
    "\U0001F300-\U0001F5FF"   # símbolos y pictogramas
    "\U0001F680-\U0001F6FF"   # transporte y mapas
    "\U0001F1E0-\U0001F1FF"   # banderas
    "\U00002500-\U00002BEF"   # símbolos varios
    "\U00010000-\U0010FFFF"   # caracteres suplementarios
    "]+",
    flags=re.UNICODE,
)


def limpiar_texto(texto: str) -> str:
    """
    Limpia un comentario individual para NLP.

    Pasos en orden:
    1. Corregir encoding corrupto
    2. Eliminar URLs  (http://... o https://...)
    3. Eliminar menciones de usuario  (@nombre)
    4. Eliminar hashtags  (#palabra) — el # y la palabra
    5. Eliminar emojis
    6. Eliminar caracteres no alfanuméricos excepto letras con tilde y ñ
    7. Normalizar espacios
    8. Convertir a minúsculas

    No eliminamos stopwords aquí — BERTopic tiene su propio manejo de
    stopwords a través del parámetro `stop_words` en CountVectorizer.
    """
    if not isinstance(texto, str) or texto.strip() == "":
        return ""

    # Paso 1: encoding
    texto = corregir_encoding(texto)

    # Paso 2: URLs (con o sin www, con parámetros)
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)

    # Paso 3: menciones @usuario
    texto = re.sub(r"@\w+", " ", texto)

    # Paso 4: hashtags — eliminamos # y la palabra que le sigue
    texto = re.sub(r"#\w+", " ", texto)

    # Paso 5: emojis
    texto = _EMOJI_PATTERN.sub(" ", texto)

    # Paso 6: caracteres especiales
    # Conservamos: letras (a-z, A-Z), números, espacios,
    #              vocales con acento (á é í ó ú ü), ñ/Ñ
    # Eliminamos: puntuación, comillas, asteriscos, pipes, etc.
    texto = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", texto)

    # Paso 7: múltiples espacios / saltos de línea → un solo espacio
    texto = re.sub(r"\s+", " ", texto).strip()

    # Paso 8: minúsculas
    texto = texto.lower()

    return texto


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DETECCIÓN DE SPAM
# ═══════════════════════════════════════════════════════════════════════════════

# Palabras clave que indican publicidad / contenido comercial.
# Presencia de cualquiera de estas en el texto raw es señal de spam.
_PALABRAS_SPAM = [
    # Comercial directo
    "dm", "direct message", "pedido", "envío", "envios",
    "disponible", "precio", "descuento", "oferta", "promoción",
    "contactanos", "contáctanos", "whatsapp", "link en bio",
    "tienda", "mayoreo", "menudeo",
    # Maquillaje / moda en contexto de reseñas turísticas
    "marykaydemexico", "marykay",
    # Sorteos / concursos
    "giveaway", "sorteo", "ganador", "participa",
]

# Compilamos como regex para una sola pasada (más eficiente que varios `in`)
_SPAM_REGEX = re.compile(
    "|".join(re.escape(p) for p in _PALABRAS_SPAM),
    flags=re.IGNORECASE,
)


def es_spam(texto_raw: str, texto_limpio: str, fuente: str) -> bool:
    """
    Retorna True si el comentario es spam o no aporta información turística.

    Criterios:
    A) Demasiado corto (< MIN_PALABRAS) — no hay contexto suficiente
    B) Contiene palabras clave comerciales
    C) Más del MAX_RATIO_HASHTAGS son hashtags (publicación de Instagram
       que es básicamente una lista de etiquetas sin contenido)

    Los criterios B y C solo aplican al texto raw (antes de limpiar)
    porque después de limpiar los hashtags ya no están presentes.
    """
    if not isinstance(texto_raw, str):
        return True

    # A) Muy corto — aplica a todas las fuentes
    n_palabras = len(texto_limpio.split())
    if n_palabras < MIN_PALABRAS:
        return True

    # B) Palabras clave comerciales — más común en Instagram
    if _SPAM_REGEX.search(texto_raw):
        return True

    # C) Ratio de hashtags — aplica sobre todo a Instagram
    palabras_raw = texto_raw.split()
    if len(palabras_raw) > 0:
        n_hashtags = sum(1 for p in palabras_raw if p.startswith("#"))
        ratio = n_hashtags / len(palabras_raw)
        if ratio > MAX_RATIO_HASHTAGS:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DETECCIÓN DE IDIOMA (heurística ligera)
# ═══════════════════════════════════════════════════════════════════════════════

# Palabras altamente frecuentes y exclusivas de cada idioma.
# No usamos langdetect para evitar una dependencia pesada y lenta en 48k filas.
# Esta heurística es suficiente para el análisis: separar ES / EN / otro.
_PALABRAS_ES = {"el", "la", "los", "las", "de", "en", "que", "es", "un",
                "una", "con", "muy", "para", "por", "del", "fue",
                "pero", "más", "están", "todo", "también"}

_PALABRAS_EN = {"the", "and", "was", "is", "are", "with", "this", "for",
                "that", "have", "great", "very", "good", "place", "beach",
                "hotel", "food", "water", "really", "beautiful"}


def detectar_idioma(texto_limpio: str) -> str:
    """
    Clasifica el comentario en 'es' (español), 'en' (inglés) u 'otro'.

    Método: contar cuántas palabras del texto coinciden con vocabulario
    frecuente de cada idioma. El idioma con más coincidencias gana.
    Si hay empate o ninguno supera 2 coincidencias, retorna 'otro'.

    Limitación conocida: Textos muy cortos o mezcla de idiomas
    (muy común en Instagram) pueden quedar como 'otro'. Para el
    análisis del dashboard esto es aceptable.
    """
    palabras = set(texto_limpio.lower().split())
    hits_es = len(palabras & _PALABRAS_ES)
    hits_en = len(palabras & _PALABRAS_EN)

    if hits_es == 0 and hits_en == 0:
        return "otro"
    if hits_es > hits_en:
        return "es"
    if hits_en > hits_es:
        return "en"
    return "otro"  # empate


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica toda la pipeline de limpieza al DataFrame completo.

    Retorna el mismo DataFrame con columnas adicionales:
      - comentario_limpio : texto después de limpieza
      - n_palabras        : número de palabras en comentario_limpio
      - es_spam           : bool — True = debe excluirse del análisis NLP
      - idioma            : 'es' | 'en' | 'otro'

    El comentario original (columna 'comentario') NO se modifica.
    Esto es importante: siempre podemos volver al texto crudo para debug.
    """
    log.info("Aplicando limpieza de texto...")

    # Asegurarnos de que comentario sea string (puede haber NaN)
    df["comentario"] = df["comentario"].fillna("").astype(str)

    # Aplicar limpieza — usamos .apply() en lugar de vectorizado porque
    # las operaciones regex encadenadas son más claras así, y 48k filas
    # corre en ~3-5 segundos, tiempo aceptable.
    df["comentario_limpio"] = df["comentario"].apply(limpiar_texto)

    log.info("Contando palabras...")
    df["n_palabras"] = df["comentario_limpio"].str.split().str.len()

    log.info("Detectando spam...")
    df["es_spam"] = df.apply(
        lambda row: es_spam(row["comentario"], row["comentario_limpio"], row["fuente"]),
        axis=1,
    )

    log.info("Detectando idioma...")
    df["idioma"] = df["comentario_limpio"].apply(detectar_idioma)

    return df


def imprimir_estadisticas(df: pd.DataFrame) -> None:
    """Muestra un resumen del resultado de la limpieza."""
    total    = len(df)
    limpios  = (~df["es_spam"]).sum()
    spam     = df["es_spam"].sum()

    log.info("═" * 55)
    log.info("  ESTADÍSTICAS DE LIMPIEZA")
    log.info("═" * 55)
    log.info(f"  Total registros     : {total:>7,}")
    log.info(f"  Registros limpios   : {limpios:>7,}  ({limpios/total*100:.1f}%)")
    log.info(f"  Marcados como spam  : {spam:>7,}  ({spam/total*100:.1f}%)")

    log.info("\n  Spam por fuente:")
    for fuente in df["fuente"].unique():
        sub = df[df["fuente"] == fuente]
        n_spam = sub["es_spam"].sum()
        log.info(f"    {fuente:<10} {n_spam:>5} spam  /  {len(sub):>6} total  "
                 f"({n_spam/len(sub)*100:.1f}%)")

    log.info("\n  Idioma (solo registros limpios):")
    clean = df[~df["es_spam"]]
    for idioma, cnt in clean["idioma"].value_counts().items():
        log.info(f"    {idioma:<6}  {cnt:>6,}  ({cnt/len(clean)*100:.1f}%)")

    log.info("\n  Longitud promedio por fuente (palabras, sin spam):")
    log.info(
        clean.groupby("fuente")["n_palabras"]
             .agg(["mean", "median", "min", "max"])
             .round(1)
             .to_string()
    )

    log.info("\n  Muestra de texto antes → después de limpiar:")
    ejemplos = df[~df["es_spam"]].sample(4, random_state=42)
    for _, row in ejemplos.iterrows():
        log.info(f"\n  [{row['fuente'].upper()} / {row['ubicacion']}]")
        log.info(f"  ANTES : {row['comentario'][:120]}")
        log.info(f"  DESPUÉS: {row['comentario_limpio'][:120]}")


def main(dry_run: bool = False) -> None:
    log.info("═" * 55)
    log.info("  LIMPIEZA DE TEXTO — TURISMO MÉXICO")
    log.info("═" * 55)

    # ── Leer ──────────────────────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        log.error(f"No se encontró el archivo de entrada: {INPUT_CSV}")
        sys.exit(1)

    log.info(f"Leyendo {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV, dtype=str)   # dtype=str para no perder nada
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    log.info(f"  {len(df):,} registros cargados")

    # ── Limpiar ───────────────────────────────────────────────────────────────
    df = limpiar_dataset(df)

    # ── Estadísticas ──────────────────────────────────────────────────────────
    imprimir_estadisticas(df)

    # ── Guardar ───────────────────────────────────────────────────────────────
    if dry_run:
        log.info("\n  [DRY-RUN] No se guardó ningún archivo.")
        return

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info(f"\n  → Guardado: {OUTPUT_CSV.relative_to(BASE_DIR)}")
    log.info(f"     Total filas  : {len(df):,}")
    log.info(f"     Para BERTopic: {(~df['es_spam']).sum():,}  "
             f"(filtra con df[~df['es_spam']])")
    log.info("═" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limpieza de texto para NLP")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra estadísticas sin guardar el archivo de salida",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)