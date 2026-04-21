"""
Data Consolidation Script — Turismo México
==========================================
Consolida datos de 3 fuentes (TripAdvisor, Instagram, Google Maps) para
5 destinos turísticos en el esquema final:

  ubicacion | lugar_especifico | fuente | rating | comentario | fecha_resena

Estructura de archivos raw esperada:
  data/raw/trip/   → trip_{location}.csv          cols: [índice, comment]
  data/raw/insta/  → insta_{location}.csv         cols: [índice, comment]
  data/raw/maps/   → maps_{location}.csv          cols: varía por destino

Particularidades por archivo Maps
----------------------------------
  maps_huatulco.csv      : [cantidad_estrellas, Comentarios]
  maps_lapaz.csv         : [lugar, estrellas, texto]
  maps_puerto_vallarta.csv: [place_name, place_category, rating,
                              review_date, review_text, city_target]
  maps_rivera_maya.csv   : [Lugar, Estrellas, Comentario, Red_Social]
  maps_riviera_nayarit.csv: [Lugar, Estrellas, Comentario, Tiempo, Red_Social]

Salida
------
  data/processed/{location}_consolidated.csv   (5 archivos)
  data/final/maestro_completo.csv              (dataset unificado)
"""

import pandas as pd
from pathlib import Path
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
RAW_DIR         = BASE_DIR / "data" / "raw"
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
FINAL_DIR       = BASE_DIR / "data" / "final"

# Columnas canónicas del esquema final
SCHEMA = ["ubicacion", "lugar_especifico", "fuente", "rating",
          "comentario", "fecha_resena"]

# Nombres legibles para el dashboard (snake_case → display)
DISPLAY_NAME = {
    "lapaz":            "La Paz",
    "puerto_vallarta":  "Puerto Vallarta",
    "rivera_maya":      "Rivera Maya",
    "riviera_nayarit":  "Riviera Nayarit",
    "huatulco":         "Huatulco",
}

LOCATIONS = list(DISPLAY_NAME.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza columnas canónicas, limpia nulos y duplicados."""
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA].copy()
    df = df.dropna(subset=["comentario"])
    df["comentario"] = df["comentario"].astype(str).str.strip()
    df = df[df["comentario"] != ""]
    df = df.drop_duplicates(subset=["comentario"])
    return df.reset_index(drop=True)


def _read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        log.warning(f"  [ENCODING] {path.name} no es UTF-8, reintentando con latin-1...")
        return pd.read_csv(path, encoding="latin-1", **kwargs)
    except FileNotFoundError:
        log.warning(f"  [SKIP] No encontrado: {path.name}")
        return None
    except Exception as e:
        log.error(f"  [ERROR] {path.name}: {e}")
        return None


def _clean_lugar(series: pd.Series) -> pd.Series:
    """Normaliza nombres de lugar: quita guiones bajos, title-case."""
    return series.astype(str).str.replace("_", " ").str.strip().str.title()


# ═══════════════════════════════════════════════════════════════════════════════
# Fuentes genéricas: Trip y Instagram
# ═══════════════════════════════════════════════════════════════════════════════

def load_trip(location: str) -> pd.DataFrame:
    """
    Trip: índice + comment → sin lugar_especifico, sin rating, sin fecha.
    Nombre especial para huatulco: trip_huatulco-T.csv
    """
    filename = "trip_huatulco-T.csv" if location == "huatulco" else f"trip_{location}.csv"
    df = _read_csv(RAW_DIR / "trip" / filename)
    if df is None:
        return _empty()

    if "comment" not in df.columns:
        log.error(f"  [Trip/{location}] columna 'comment' no encontrada — cols: {df.columns.tolist()}")
        return _empty()

    out = pd.DataFrame({
        "ubicacion":        DISPLAY_NAME[location],
        "lugar_especifico": None,
        "fuente":           "trip",
        "rating":           None,
        "comentario":       df["comment"],
        "fecha_resena":     None,
    })
    result = _finalize(out)
    log.info(f"  [Trip/{location}] {len(result):>5} registros")
    return result


def load_insta(location: str) -> pd.DataFrame:
    """
    Instagram: índice + comment → sin lugar_especifico, sin rating, sin fecha.
    """
    df = _read_csv(RAW_DIR / "insta" / f"insta_{location}.csv")
    if df is None:
        return _empty()

    if "comment" not in df.columns:
        log.error(f"  [Insta/{location}] columna 'comment' no encontrada — cols: {df.columns.tolist()}")
        return _empty()

    out = pd.DataFrame({
        "ubicacion":        DISPLAY_NAME[location],
        "lugar_especifico": None,
        "fuente":           "insta",
        "rating":           None,
        "comentario":       df["comment"],
        "fecha_resena":     None,
    })
    result = _finalize(out)
    log.info(f"  [Insta/{location}] {len(result):>5} registros")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Google Maps — un loader por destino (columnas distintas en cada archivo)
# ═══════════════════════════════════════════════════════════════════════════════

def _maps_huatulco() -> pd.DataFrame:
    """
    Columnas: cantidad_estrellas, Comentarios
    Sin lugar_especifico, sin fecha.
    """
    df = _read_csv(RAW_DIR / "maps" / "maps_huatulco.csv")
    if df is None:
        return _empty()

    # Buscar columnas flexiblemente por nombre
    rating_col  = next((c for c in df.columns if "estrell" in c.lower()), None)
    comment_col = next((c for c in df.columns if "comentario" in c.lower()
                        or "comment" in c.lower()), None)

    if comment_col is None:
        log.error(f"  [Maps/huatulco] columna de comentario no encontrada — cols: {df.columns.tolist()}")
        return _empty()

    out = pd.DataFrame({
        "ubicacion":        DISPLAY_NAME["huatulco"],
        "lugar_especifico": None,
        "fuente":           "maps",
        "rating":           df[rating_col] if rating_col else None,
        "comentario":       df[comment_col],
        "fecha_resena":     None,
    })
    result = _finalize(out)
    log.info(f"  [Maps/huatulco] {len(result):>5} registros")
    return result


def _maps_lapaz() -> pd.DataFrame:
    """
    Columnas: lugar, estrellas, texto
    'lugar' → lugar_especifico (nombre del sitio específico en La Paz).
    Sin fecha.
    """
    df = _read_csv(RAW_DIR / "maps" / "maps_lapaz.csv")
    if df is None:
        return _empty()

    lugar_col   = next((c for c in df.columns if c.lower() == "lugar"), None)
    rating_col  = next((c for c in df.columns if "estrell" in c.lower()
                        or c.lower() == "rating"), None)
    comment_col = next((c for c in df.columns if "texto" in c.lower()
                        or "comentario" in c.lower()
                        or "comment" in c.lower()), None)

    if comment_col is None:
        log.error(f"  [Maps/lapaz] columna de comentario no encontrada — cols: {df.columns.tolist()}")
        return _empty()

    out = pd.DataFrame({
        "ubicacion":        DISPLAY_NAME["lapaz"],
        "lugar_especifico": _clean_lugar(df[lugar_col]) if lugar_col else None,
        "fuente":           "maps",
        "rating":           df[rating_col] if rating_col else None,
        "comentario":       df[comment_col],
        "fecha_resena":     None,
    })
    result = _finalize(out)
    log.info(f"  [Maps/lapaz] {len(result):>5} registros  "
             f"({result['lugar_especifico'].nunique()} lugares únicos)")
    return result


def _maps_puerto_vallarta() -> pd.DataFrame:
    """
    Columnas: place_name, place_category, rating, review_date, review_text, city_target
    Usamos: place_name → lugar_especifico, rating, review_date → fecha_resena,
            review_text → comentario. Descartamos place_category y city_target.
    """
    df = _read_csv(RAW_DIR / "maps" / "maps_puerto_vallarta.csv")
    if df is None:
        return _empty()

    # Mapeo explícito basado en columnas conocidas
    col_map = {
        "place_name":   "lugar_especifico",
        "rating":       "rating",
        "review_date":  "fecha_resena",
        "review_text":  "comentario",
    }
    # Verificar qué columnas existen realmente
    missing = [c for c in col_map if c not in df.columns]
    if missing:
        log.warning(f"  [Maps/puerto_vallarta] columnas faltantes: {missing} — cols disponibles: {df.columns.tolist()}")

    out = pd.DataFrame({
        "ubicacion":        DISPLAY_NAME["puerto_vallarta"],
        "lugar_especifico": _clean_lugar(df["place_name"])  if "place_name"  in df.columns else None,
        "fuente":           "maps",
        "rating":           df["rating"]      if "rating"      in df.columns else None,
        "comentario":       df["review_text"] if "review_text" in df.columns else None,
        "fecha_resena":     pd.to_datetime(df["review_date"], errors="coerce")
                            if "review_date" in df.columns else None,
    })
    result = _finalize(out)
    log.info(f"  [Maps/puerto_vallarta] {len(result):>5} registros  "
             f"({result['lugar_especifico'].nunique()} lugares únicos)")
    return result


def _maps_rivera_maya() -> pd.DataFrame:
    """
    Columnas: Lugar, Estrellas, Comentario, Red_Social
    Usamos: Lugar → lugar_especifico, Estrellas → rating, Comentario → comentario.
    Descartamos Red_Social.  Sin fecha.
    """
    df = _read_csv(RAW_DIR / "maps" / "maps_rivera_maya.csv")
    if df is None:
        return _empty()

    lugar_col   = next((c for c in df.columns if c.lower() == "lugar"), None)
    rating_col  = next((c for c in df.columns if "estrell" in c.lower()
                        or c.lower() == "rating"), None)
    comment_col = next((c for c in df.columns if "comentario" in c.lower()
                        or "comment" in c.lower()), None)

    if comment_col is None:
        log.error(f"  [Maps/rivera_maya] columna de comentario no encontrada — cols: {df.columns.tolist()}")
        return _empty()

    out = pd.DataFrame({
        "ubicacion":        DISPLAY_NAME["rivera_maya"],
        "lugar_especifico": _clean_lugar(df[lugar_col]) if lugar_col else None,
        "fuente":           "maps",
        "rating":           df[rating_col] if rating_col else None,
        "comentario":       df[comment_col],
        "fecha_resena":     None,
    })
    result = _finalize(out)
    log.info(f"  [Maps/rivera_maya] {len(result):>5} registros  "
             f"({result['lugar_especifico'].nunique()} lugares únicos)")
    return result


def _maps_riviera_nayarit() -> pd.DataFrame:
    """
    Columnas: Lugar, Estrellas, Comentario, Tiempo, Red_Social
    Usamos: Lugar → lugar_especifico, Estrellas → rating,
            Comentario → comentario, Tiempo → fecha_resena.
    Descartamos Red_Social.
    """
    df = _read_csv(RAW_DIR / "maps" / "maps_riviera_nayarit.csv")
    if df is None:
        return _empty()

    lugar_col   = next((c for c in df.columns if c.lower() == "lugar"), None)
    rating_col  = next((c for c in df.columns if "estrell" in c.lower()
                        or c.lower() == "rating"), None)
    comment_col = next((c for c in df.columns if "comentario" in c.lower()
                        or "comment" in c.lower()), None)
    fecha_col   = next((c for c in df.columns if "tiempo" in c.lower()
                        or "fecha" in c.lower() or "date" in c.lower()), None)

    if comment_col is None:
        log.error(f"  [Maps/riviera_nayarit] columna de comentario no encontrada — cols: {df.columns.tolist()}")
        return _empty()

    out = pd.DataFrame({
        "ubicacion":        DISPLAY_NAME["riviera_nayarit"],
        "lugar_especifico": _clean_lugar(df[lugar_col]) if lugar_col else None,
        "fuente":           "maps",
        "rating":           df[rating_col] if rating_col else None,
        "comentario":       df[comment_col],
        "fecha_resena":     pd.to_datetime(df[fecha_col], errors="coerce")
                            if fecha_col else None,
    })
    result = _finalize(out)
    log.info(f"  [Maps/riviera_nayarit] {len(result):>5} registros  "
             f"({result['lugar_especifico'].nunique()} lugares únicos)")
    return result


# Dispatcher: location → función maps específica
MAPS_LOADERS = {
    "huatulco":         _maps_huatulco,
    "lapaz":            _maps_lapaz,
    "puerto_vallarta":  _maps_puerto_vallarta,
    "rivera_maya":      _maps_rivera_maya,
    "riviera_nayarit":  _maps_riviera_nayarit,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Orquestación
# ═══════════════════════════════════════════════════════════════════════════════

def consolidate_location(location: str) -> pd.DataFrame:
    log.info(f"\n{'─'*55}")
    log.info(f"  {DISPLAY_NAME[location].upper()}")
    log.info(f"{'─'*55}")

    trip_df  = load_trip(location)
    insta_df = load_insta(location)
    maps_df  = MAPS_LOADERS[location]()

    parts = [df for df in [trip_df, insta_df, maps_df] if not df.empty]

    if not parts:
        log.error(f"  [FAIL] {location}: ninguna fuente cargó datos")
        return _empty()

    combined = pd.concat(parts, ignore_index=True)

    # Resumen
    log.info(f"\n  TOTAL: {len(combined)} registros")
    for fuente, cnt in combined["fuente"].value_counts().items():
        pct = cnt / len(combined) * 100
        log.info(f"    {fuente:<6} {cnt:>5}  ({pct:.1f}%)")

    return combined


def save(df: pd.DataFrame, path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"  → Guardado: {path.relative_to(BASE_DIR)}  ({len(df)} filas)")
        return True
    except Exception as e:
        log.error(f"  → Error guardando {path}: {e}")
        return False


def process_all() -> dict[str, pd.DataFrame]:
    """Procesa los 5 destinos y guarda en /processed. Retorna dict de DataFrames."""
    results = {}
    for loc in LOCATIONS:
        df = consolidate_location(loc)
        if df.empty:
            continue
        out_path = PROCESSED_DIR / f"{loc}_consolidated.csv"
        if save(df, out_path):
            results[loc] = df
    return results


def create_master(dfs: dict[str, pd.DataFrame]) -> bool:
    """Une todos los processed en un maestro y guarda en /final."""
    if not dfs:
        # Intentar leer desde disco si no se pasaron DataFrames
        for loc in LOCATIONS:
            p = PROCESSED_DIR / f"{loc}_consolidated.csv"
            if p.exists():
                dfs[loc] = pd.read_csv(p)

    if not dfs:
        log.error("No hay datos consolidados para crear el maestro")
        return False

    master = pd.concat(dfs.values(), ignore_index=True)

    log.info(f"\n{'═'*55}")
    log.info("  DATASET MAESTRO")
    log.info(f"{'═'*55}")
    log.info(f"  Total registros : {len(master)}")
    log.info(f"  Por ubicación:")
    for loc, cnt in master["ubicacion"].value_counts().items():
        log.info(f"    {loc:<25} {cnt:>5}")
    log.info(f"  Por fuente:")
    for fuente, cnt in master["fuente"].value_counts().items():
        log.info(f"    {fuente:<10} {cnt:>5}")
    log.info(f"  Con rating      : {master['rating'].notna().sum()}")
    log.info(f"  Con fecha       : {master['fecha_resena'].notna().sum()}")
    log.info(f"  Con lugar esp.  : {master['lugar_especifico'].notna().sum()}")

    return save(master, FINAL_DIR / "maestro_completo.csv")


def main():
    log.info("═" * 55)
    log.info("  CONSOLIDACIÓN TURISMO MÉXICO")
    log.info("═" * 55)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    dfs = process_all()

    ok = create_master(dfs)

    log.info(f"\n{'═'*55}")
    status = "✓ COMPLETADO" if ok else "✗ COMPLETADO CON ERRORES"
    log.info(f"  {status}")
    log.info(f"  {len(dfs)}/{len(LOCATIONS)} destinos procesados")
    log.info(f"{'═'*55}\n")
    return ok


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)