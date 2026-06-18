from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "parliament_multiclass" / "all_samples.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "parliament_multiclass"

TRAIN_PATH = OUTPUT_DIR / "train_split.csv"
VAL_PATH = OUTPUT_DIR / "val_split.csv"
TEST_PATH = OUTPUT_DIR / "test_split.csv"
SUMMARY_PATH = OUTPUT_DIR / "split_summary.json"

TARGET_COL = "political_group"
GROUP_COL = "speaker_id"
RANDOM_STATE = 42

# Para usar StratifiedGroupKFold con 5 folds en el primer split,
# cada clase debería tener al menos 5 grupos distintos.
MIN_GROUPS_PER_CLASS = 5


def print_distribution(name: str, df: pd.DataFrame) -> None:
    print(f"\nDistribución de {TARGET_COL} en {name}:")
    print(df[TARGET_COL].value_counts().to_dict())
    print(f"Oradores únicos ({GROUP_COL}) en {name}: {df[GROUP_COL].nunique()}")
    print(f"Shape {name}: {df.shape}")


def get_first_fold_indices(df: pd.DataFrame, n_splits: int):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_idx, holdout_idx = next(
        splitter.split(df, y=df[TARGET_COL], groups=df[GROUP_COL])
    )
    return train_idx, holdout_idx


def filter_classes_with_few_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    group_counts = (
        df.groupby(TARGET_COL)[GROUP_COL]
        .nunique()
        .sort_values(ascending=True)
    )

    invalid_classes = group_counts[group_counts < MIN_GROUPS_PER_CLASS].index.tolist()
    removed_info = {}

    if invalid_classes:
        for cls in invalid_classes:
            removed_info[cls] = {
                "num_groups": int(group_counts[cls]),
                "num_rows": int((df[TARGET_COL] == cls).sum()),
            }

        df = df[~df[TARGET_COL].isin(invalid_classes)].copy()

    return df, removed_info


def build_summary(train_df, val_df, test_df, removed_classes):
    train_groups = set(train_df[GROUP_COL])
    val_groups = set(val_df[GROUP_COL])
    test_groups = set(test_df[GROUP_COL])

    return {
        "target_col": TARGET_COL,
        "group_col": GROUP_COL,
        "random_state": RANDOM_STATE,
        "min_groups_per_class": MIN_GROUPS_PER_CLASS,
        "removed_classes_due_to_few_groups": removed_classes,
        "train_shape": list(train_df.shape),
        "val_shape": list(val_df.shape),
        "test_shape": list(test_df.shape),
        "train_num_groups": int(train_df[GROUP_COL].nunique()),
        "val_num_groups": int(val_df[GROUP_COL].nunique()),
        "test_num_groups": int(test_df[GROUP_COL].nunique()),
        "train_distribution": train_df[TARGET_COL].value_counts().to_dict(),
        "val_distribution": val_df[TARGET_COL].value_counts().to_dict(),
        "test_distribution": test_df[TARGET_COL].value_counts().to_dict(),
        "group_overlap": {
            "train_val": len(train_groups.intersection(val_groups)),
            "train_test": len(train_groups.intersection(test_groups)),
            "val_test": len(val_groups.intersection(test_groups)),
        },
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No existe el dataset procesado esperado en {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_cols = {TARGET_COL, GROUP_COL, "text_segment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {sorted(missing)}")

    print("=" * 60)
    print("SPLIT PARLAMENTARIO MULTICLASE")
    print("=" * 60)
    print(f"Ruta origen: {DATA_PATH}")
    print(f"Shape inicial: {df.shape}")
    print(f"Clases iniciales: {df[TARGET_COL].nunique()}")
    print(f"Oradores iniciales: {df[GROUP_COL].nunique()}")

    df, removed_classes = filter_classes_with_few_groups(df)

    if df.empty:
        raise ValueError("El dataset quedó vacío tras filtrar clases con pocos grupos.")

    if df[TARGET_COL].nunique() < 2:
        raise ValueError("Se necesitan al menos 2 clases para hacer el split.")

    if removed_classes:
        print("\nClases eliminadas por tener pocos grupos distintos:")
        for cls, info in removed_classes.items():
            print(
                f"  - {cls}: {info['num_groups']} oradores, {info['num_rows']} muestras"
            )

    print(f"\nShape tras posible filtrado de clases raras: {df.shape}")
    print(f"Clases finales para el split: {df[TARGET_COL].nunique()}")
    print(f"Oradores finales para el split: {df[GROUP_COL].nunique()}")

    # 80% train_val, 20% test
    train_val_idx, test_idx = get_first_fold_indices(df, n_splits=5)
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Del 80% restante, 25% a validación -> 60/20/20 final
    train_idx, val_idx = get_first_fold_indices(train_val_df, n_splits=4)
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    print_distribution("train", train_df)
    print_distribution("val", val_df)
    print_distribution("test", test_df)

    train_groups = set(train_df[GROUP_COL])
    val_groups = set(val_df[GROUP_COL])
    test_groups = set(test_df[GROUP_COL])

    print("\nComprobación de fuga de grupos:")
    print(f"Train-Val overlap: {len(train_groups.intersection(val_groups))}")
    print(f"Train-Test overlap: {len(train_groups.intersection(test_groups))}")
    print(f"Val-Test overlap: {len(val_groups.intersection(test_groups))}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")

    summary = build_summary(train_df, val_df, test_df, removed_classes)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\nArchivos guardados:")
    print(f"Train: {TRAIN_PATH}")
    print(f"Val: {VAL_PATH}")
    print(f"Test: {TEST_PATH}")
    print(f"Resumen: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()