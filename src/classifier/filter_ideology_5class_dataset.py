from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

BASE_INPUT_DIR = BASE_DIR / "data" / "raw" / "parliament"
BASE_OUTPUT_DIR = BASE_DIR / "data" / "processed" / "parliament_ideology_5class"

PARTY_COLUMN = "political_group"
SEED = 42

# Legislatura 13: desde 2019 aprox.
MIN_LEGISLATURE = 13

LABEL_COLUMN = "ideology_5_label"


IDEOLOGY_5CLASS_MAP = {
    # Izquierda extrema
    "iu": "far_left",
    "up": "far_left",

    # Izquierda
    "psoe": "left",

    # Centro
    "cs": "center",
    "upd": "center",

    # Derecha
    "pp": "right",

    # Derecha extrema
    "vox": "far_right",
}


NATIONALIST_OR_PERIPHERAL_PARTIES = {
    "cc",
    "ciu",
    "ehb",
    "erc",
    "pnv",
}


VERSIONS = {
    "1": {
        "name": "v1_full_5class",
        "description": "Todo el dataset + etiquetas ideológicas de 5 clases",
        "filter_legislature": False,
        "remove_nationalists": False,
        "nationalist_class": False,
        "balance": False,
    },
    "2": {
        "name": "v2_since_legislature_13_5class",
        "description": "Desde legislatura 13 + etiquetas ideológicas de 5 clases",
        "filter_legislature": True,
        "remove_nationalists": False,
        "nationalist_class": False,
        "balance": False,
    },
    "3": {
        "name": "v3_since_legislature_13_5class_balanced",
        "description": "Desde legislatura 13 + etiquetas ideológicas de 5 clases + balanceo",
        "filter_legislature": True,
        "remove_nationalists": False,
        "nationalist_class": False,
        "balance": True,
    },
    "4": {
        "name": "v4_since_legislature_13_5class_no_nationalist_balanced",
        "description": "Desde legislatura 13 + sin nacionalistas/periféricos + balanceo",
        "filter_legislature": True,
        "remove_nationalists": True,
        "nationalist_class": False,
        "balance": True,
    },
    "5": {
        "name": "v5_since_legislature_13_6class_nationalist_balanced",
        "description": "Desde legislatura 13 + nacionalistas como clase propia + balanceo",
        "filter_legislature": True,
        "remove_nationalists": False,
        "nationalist_class": True,
        "balance": True,
    },
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalize_party(value: str) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def extract_legislature_number(file_path: Path) -> int:
    """
    Ejemplo:
    legislature_07.parquet.gzip -> 7
    legislature_14.parquet.gzip -> 14
    """
    name = file_path.name
    number = name.replace("legislature_", "").replace(".parquet.gzip", "")
    return int(number)


def load_full_dataset() -> pd.DataFrame:
    files = sorted(BASE_INPUT_DIR.glob("legislature_*.parquet.gzip"))

    if not files:
        raise FileNotFoundError(
            f"No se han encontrado archivos parquet en: {BASE_INPUT_DIR}"
        )

    dfs = []

    print("\nCargando archivos parquet...")

    for file in files:
        print(f"  - {file.name}")

        df = pd.read_parquet(file)
        df["legislature"] = extract_legislature_number(file)

        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    print(f"\nDataset completo cargado: {len(full_df)} muestras")
    print("\nColumnas disponibles:")
    print(list(full_df.columns))

    if PARTY_COLUMN not in full_df.columns:
        raise ValueError(
            f"No existe la columna '{PARTY_COLUMN}'. "
            f"Columnas disponibles: {list(full_df.columns)}"
        )

    return full_df


def filter_by_legislature(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df[df["legislature"] >= MIN_LEGISLATURE].copy()

    after = len(df)

    print(
        f"Filtrado desde legislatura {MIN_LEGISLATURE}: "
        f"{before} -> {after} muestras"
    )

    return df


def remove_nationalists(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["party_normalized"] = df[PARTY_COLUMN].apply(normalize_party)

    before = len(df)
    df = df[~df["party_normalized"].isin(NATIONALIST_OR_PERIPHERAL_PARTIES)]
    after = len(df)

    print(
        "Eliminación de partidos nacionalistas/periféricos: "
        f"{before} -> {after} muestras"
    )

    return df


def add_ideology_5class_labels(
    df: pd.DataFrame,
    include_nationalists_as_class: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    df["party_normalized"] = df[PARTY_COLUMN].apply(normalize_party)

    ideology_map = IDEOLOGY_5CLASS_MAP.copy()

    if include_nationalists_as_class:
        for party in NATIONALIST_OR_PERIPHERAL_PARTIES:
            ideology_map[party] = "nationalist"

    df[LABEL_COLUMN] = df["party_normalized"].map(ideology_map)

    before = len(df)
    df = df.dropna(subset=[LABEL_COLUMN])
    after = len(df)

    print(
        "Asignación de etiquetas ideológicas: "
        f"{before} -> {after} muestras"
    )

    return df


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    class_counts = df[LABEL_COLUMN].value_counts()
    min_count = class_counts.min()

    print("\nDistribución antes del balanceo:")
    print(class_counts)

    balanced_df = (
        df.groupby(LABEL_COLUMN, group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=SEED))
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    print("\nDistribución después del balanceo:")
    print(balanced_df[LABEL_COLUMN].value_counts())

    return balanced_df


def split_and_save(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df[LABEL_COLUMN],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df[LABEL_COLUMN],
    )

    train_path = output_dir / "train_split.csv"
    val_path = output_dir / "val_split.csv"
    test_path = output_dir / "test_split.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"\nArchivos guardados en: {output_dir}")
    print(f"  Train: {len(train_df)} -> {train_path.name}")
    print(f"  Val:   {len(val_df)} -> {val_path.name}")
    print(f"  Test:  {len(test_df)} -> {test_path.name}")


def print_final_distribution(df: pd.DataFrame) -> None:
    print("\nDistribución final por ideología:")
    print(df[LABEL_COLUMN].value_counts())

    print("\nDistribución final por partido:")
    print(df["party_normalized"].value_counts())


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("GENERANDO TODAS LAS VERSIONES DEL DATASET DE 5/6 CLASES")
    print("=" * 70)

    original_df = load_full_dataset()

    for _, config in VERSIONS.items():
        print("\n" + "=" * 70)
        print(f"Generando {config['name']}")
        print(config["description"])
        print("=" * 70)

        df = original_df.copy()

        if config["filter_legislature"]:
            df = filter_by_legislature(df)

        if config["remove_nationalists"]:
            df = remove_nationalists(df)

        df = add_ideology_5class_labels(
            df,
            include_nationalists_as_class=config["nationalist_class"],
        )

        print_final_distribution(df)

        if config["balance"]:
            df = balance_dataset(df)

        output_dir = BASE_OUTPUT_DIR / config["name"]
        split_and_save(df, output_dir)

    print("\n" + "=" * 70)
    print("TODAS LAS VERSIONES SE HAN GENERADO CORRECTAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()