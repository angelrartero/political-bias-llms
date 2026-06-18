from pathlib import Path
import re
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "parliament"
OUTPUT_DIR = ROOT / "data" / "processed" / "parliament_multiclass"

OUTPUT_DATASET_PATH = OUTPUT_DIR / "all_samples.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "preprocessing_summary.json"

PARQUET_PATTERN = "legislature_*.parquet.gzip"

REQUIRED_COLUMNS = [
    "legislatura",
    "fecha",
    "numero_expediente",
    "orador",
    "political_group",
    "text",
]

EXCLUDED_GROUPS = {"rtve", "mixto", "plural"}

MIN_TEXT_CHARS = 300
TARGET_WORDS_PER_CHUNK = 220
MIN_WORDS_PER_CHUNK = 80
MAX_WORDS_PER_CHUNK = 320


def normalize_string(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_label(value) -> str:
    return normalize_string(value).lower()


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_speaker_id(speaker: str) -> str:
    speaker = normalize_label(speaker)
    speaker = re.sub(r"\s+", " ", speaker)
    return speaker.strip()


def split_into_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[\.\!\?\:\;])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_text(
    text: str,
    target_words: int = TARGET_WORDS_PER_CHUNK,
    min_words: int = MIN_WORDS_PER_CHUNK,
    max_words: int = MAX_WORDS_PER_CHUNK,
) -> list[str]:
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # Si una frase sola ya es enorme, cerramos primero el chunk acumulado
        # y luego partimos esa frase por palabras.
        if sentence_words > max_words:
            if current_chunk:
                chunk = " ".join(current_chunk).strip()
                if min_words <= len(chunk.split()) <= max_words:
                    chunks.append(chunk)
                current_chunk = []
                current_words = 0

            words = sentence.split()
            for i in range(0, len(words), target_words):
                piece = " ".join(words[i:i + target_words]).strip()
                if min_words <= len(piece.split()) <= max_words:
                    chunks.append(piece)
            continue

        if current_chunk and current_words + sentence_words > target_words:
            chunk = " ".join(current_chunk).strip()
            if min_words <= len(chunk.split()) <= max_words:
                chunks.append(chunk)
            current_chunk = [sentence]
            current_words = sentence_words
        else:
            current_chunk.append(sentence)
            current_words += sentence_words

    if current_chunk:
        chunk = " ".join(current_chunk).strip()
        if min_words <= len(chunk.split()) <= max_words:
            chunks.append(chunk)

    return chunks


def main() -> None:
    parquet_files = sorted(RAW_DIR.glob(PARQUET_PATTERN))
    if not parquet_files:
        raise FileNotFoundError(
            f"No se encontraron ficheros {PARQUET_PATTERN} en {RAW_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    file_summaries = []

    total_original_rows = 0
    total_after_basic_clean = 0
    total_excluded_groups = 0
    total_short_text_removed = 0
    total_generated_chunks = 0

    print("=" * 70)
    print("PREPROCESAMIENTO PARLAMENTARIO MULTICLASE")
    print("=" * 70)
    print(f"Directorio origen: {RAW_DIR}")
    print(f"Ficheros detectados: {len(parquet_files)}")

    for file_path in parquet_files:
        print(f"\nLeyendo: {file_path.name}")

        df = pd.read_parquet(
            file_path,
            engine="pyarrow",
            columns=REQUIRED_COLUMNS,
        )

        original_rows = len(df)
        total_original_rows += original_rows

        df = df.copy()
        df["legislatura"] = df["legislatura"].apply(normalize_string)
        df["fecha"] = df["fecha"].apply(normalize_string)
        df["numero_expediente"] = df["numero_expediente"].apply(normalize_string)
        df["orador"] = df["orador"].apply(normalize_string)
        df["speaker_id"] = df["orador"].apply(normalize_speaker_id)
        df["political_group"] = df["political_group"].apply(normalize_label)
        df["text"] = df["text"].apply(normalize_string)
        df["text"] = df["text"].apply(normalize_whitespace)

        # Limpieza básica
        df = df[df["orador"] != ""].copy()
        df = df[df["speaker_id"] != ""].copy()
        df = df[df["political_group"] != ""].copy()
        df = df[df["text"] != ""].copy()

        after_basic_clean = len(df)
        total_after_basic_clean += after_basic_clean

        # Excluir grupos no válidos
        excluded_mask = df["political_group"].isin(EXCLUDED_GROUPS)
        excluded_count = int(excluded_mask.sum())
        total_excluded_groups += excluded_count
        df = df.loc[~excluded_mask].copy()

        # Eliminar textos demasiado cortos
        short_mask = df["text"].str.len() < MIN_TEXT_CHARS
        short_removed = int(short_mask.sum())
        total_short_text_removed += short_removed
        df = df.loc[~short_mask].copy()

        chunk_count_this_file = 0

        for source_row_id, row in enumerate(df.itertuples(index=False)):
            chunks = chunk_text(row.text)

            for idx, chunk in enumerate(chunks):
                sample_id = (
                    f"{row.legislatura}_{row.numero_expediente}_"
                    f"{row.speaker_id}_{source_row_id}_{idx}"
                )

                all_rows.append(
                    {
                        "sample_id": sample_id,
                        "speaker_id": row.speaker_id,
                        "legislatura": row.legislatura,
                        "fecha": row.fecha,
                        "numero_expediente": row.numero_expediente,
                        "orador": row.orador,
                        "political_group": row.political_group,
                        "text_segment": chunk,
                        "segment_word_count": len(chunk.split()),
                        "segment_char_count": len(chunk),
                    }
                )
                chunk_count_this_file += 1

        total_generated_chunks += chunk_count_this_file

        file_summary = {
            "file": file_path.name,
            "original_rows": original_rows,
            "after_basic_clean": after_basic_clean,
            "excluded_groups_removed": excluded_count,
            "short_text_removed": short_removed,
            "remaining_rows_for_chunking": len(df),
            "generated_chunks": chunk_count_this_file,
        }
        file_summaries.append(file_summary)

        print(f"  Filas originales: {original_rows}")
        print(f"  Tras limpieza básica: {after_basic_clean}")
        print(f"  Excluidas por grupo: {excluded_count}")
        print(f"  Eliminadas por texto corto: {short_removed}")
        print(f"  Filas válidas para segmentar: {len(df)}")
        print(f"  Fragmentos generados: {chunk_count_this_file}")

    processed_df = pd.DataFrame(all_rows)

    if processed_df.empty:
        raise ValueError("El dataset procesado quedó vacío tras el preprocesado.")

    before_dedup = len(processed_df)

    processed_df = processed_df.drop_duplicates(
        subset=["speaker_id", "political_group", "text_segment"]
    ).reset_index(drop=True)

    removed_duplicates = before_dedup - len(processed_df)

    processed_df.to_csv(OUTPUT_DATASET_PATH, index=False, encoding="utf-8")

    summary = {
        "raw_dir": str(RAW_DIR),
        "num_files": len(parquet_files),
        "excluded_groups": sorted(EXCLUDED_GROUPS),
        "min_text_chars": MIN_TEXT_CHARS,
        "target_words_per_chunk": TARGET_WORDS_PER_CHUNK,
        "min_words_per_chunk": MIN_WORDS_PER_CHUNK,
        "max_words_per_chunk": MAX_WORDS_PER_CHUNK,
        "total_original_rows": total_original_rows,
        "total_after_basic_clean": total_after_basic_clean,
        "total_excluded_groups_removed": total_excluded_groups,
        "total_short_text_removed": total_short_text_removed,
        "total_generated_chunks_before_dedup": total_generated_chunks,
        "duplicates_removed_after_chunking": removed_duplicates,
        "final_num_samples": int(len(processed_df)),
        "final_num_speakers": int(processed_df["speaker_id"].nunique()),
        "final_num_classes": int(processed_df["political_group"].nunique()),
        "class_distribution": processed_df["political_group"].value_counts().to_dict(),
        "file_summaries": file_summaries,
    }

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"Muestras finales: {processed_df.shape[0]}")
    print(f"Oradores únicos: {processed_df['speaker_id'].nunique()}")
    print(f"Clases finales: {processed_df['political_group'].nunique()}")
    print(f"Duplicados eliminados tras segmentación: {removed_duplicates}")
    print("\nDistribución final por political_group:")
    print(processed_df["political_group"].value_counts())
    print(f"\nDataset guardado en: {OUTPUT_DATASET_PATH}")
    print(f"Resumen guardado en: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()