# EDA.py
# -----------------------------------------
# Recolector de métricas de desbalance para MedMNIST (2D y opcional 3D)
# Devuelve un DataFrame resumen y un objeto detallado para graficar después.

from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional, Any

import numpy as np
import pandas as pd
from importlib import import_module
from medmnist import INFO
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Utilidades de selección
# -----------------------------
def listar_datasets_medmnist(include_3d: bool = False) -> List[str]:
    """
    Lista los nombres de datasets registrados en medmnist.INFO.
    Por defecto excluye los 3D (p.ej. organmnist3d).
    """
    nombres = []
    for name, meta in INFO.items():
        pyclass = str(meta.get("python_class", "")).lower()
        # Heurística para 3D
        es_3d = ("3d" in name.lower()) or ("3d" in pyclass)
        if es_3d and not include_3d:
            continue
        nombres.append(name)
    # Orden alfabético estable para reproducibilidad
    return sorted(nombres)


# -----------------------------
# Métricas de distribución
# -----------------------------
def _entropia(proportions: np.ndarray) -> float:
    p = proportions[proportions > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0

def _gini(proportions: np.ndarray) -> float:
    # Índice de Gini de distribución (1 - sum p_i^2)
    return float(1.0 - np.sum(proportions ** 2)) if proportions.size else 0.0

def _imbalance_ratio(counts: np.ndarray) -> Optional[float]:
    if counts.size == 0:
        return None
    mn = counts.min()
    mx = counts.max()
    if mn == 0:
        return None  # o math.inf si prefieres
    return float(mx / mn)


# -----------------------------
# Núcleo de cuantificación
# -----------------------------
def cuantificar_desbalance_dataset(
    dataset_name: str,
    split: str = "train",
    download: bool = False
) -> Dict[str, Any]:
    """
    Cuantifica el desbalance para un dataset/split de MedMNIST.
    Soporta multiclase y multilabel. No grafica.
    Retorna un dict con conteos, proporciones y métricas.
    """
    assert dataset_name in INFO, f"{dataset_name} no está en medmnist.INFO"
    info = INFO[dataset_name]

    # Cargar clase de dataset
    class_name = info["python_class"]
    DatasetClass = getattr(import_module("medmnist"), class_name)
    ds = DatasetClass(split=split, download=download)

    y = ds.labels
    labels_map = info.get("label", None)

    # Detectar multilabel vs multiclase
    multilabel = (y.ndim > 1 and y.shape[1] > 1)

    if multilabel:
        # En multilabel, el "conteo por clase" es el nº de positivos por clase
        counts = (y > 0).sum(axis=0).astype(int)
        n_classes = y.shape[1]
    else:
        y1 = y.squeeze()
        # Inferir nº de clases
        n_classes = info.get("n_classes", len(labels_map) if labels_map else int(y1.max()) + 1)
        counts = np.bincount(y1, minlength=int(n_classes))

    total = counts.sum()
    proportions = counts / total if total > 0 else np.zeros_like(counts, dtype=float)

    resumen = {
        "dataset": dataset_name,
        "split": split,
        "task": info.get("task", "unknown"),
        "multilabel": bool(multilabel),
        "n_samples": int(y.shape[0]),
        "n_classes": int(n_classes),
        "class_counts": counts.astype(int).tolist(),
        "class_proportions": proportions.astype(float).tolist(),
        "imbalance_ratio": _imbalance_ratio(counts),
        "entropy": _entropia(proportions),
        "gini_index": _gini(proportions),
        "labels": labels_map
    }
    return resumen


# -----------------------------
# Orquestador: múltiples datasets/splits
# -----------------------------
def ejecutar_coleccion_medmnist(
    datasets: Optional[Iterable[str]] = None,
    splits: Iterable[str] = ("train",),
    download: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Ejecuta la cuantificación para varios datasets y splits.
    Devuelve:
      - df_resumen: DataFrame con una fila por (dataset, split)
      - obj_detallado: dict[dataset][split] = resumen_completo (incluye conteos y proporciones)
    """
    if datasets is None:
        datasets = listar_datasets_medmnist(include_3d=False)

    registros = []
    obj_detallado: Dict[str, Dict[str, Any]] = {}

    for ds_name in datasets:
        obj_detallado[ds_name] = {}
        for sp in splits:
            r = cuantificar_desbalance_dataset(ds_name, split=sp, download=download)
            obj_detallado[ds_name][sp] = r

            counts = np.array(r["class_counts"], dtype=float)
            fila = {
                "dataset": r["dataset"],
                "split": r["split"],
                "task": r["task"],
                "multilabel": r["multilabel"],
                "n_samples": r["n_samples"],
                "n_classes": r["n_classes"],
                "min_count": int(counts.min()) if counts.size else None,
                "max_count": int(counts.max()) if counts.size else None,
                "mean_count": float(counts.mean()) if counts.size else None,
                "std_count": float(counts.std(ddof=0)) if counts.size else None,
                "imbalance_ratio": r["imbalance_ratio"] if r["imbalance_ratio"] is None else float(r["imbalance_ratio"]),
                "entropy": float(r["entropy"]),
                "gini_index": float(r["gini_index"]),
            }
            registros.append(fila)

    df_resumen = pd.DataFrame(registros).sort_values(["dataset", "split"]).reset_index(drop=True)
    return df_resumen, obj_detallado

# -----------------------------
# Funciones de graficado
# -----------------------------
def plot_imbalance_bar(df, split="train"):
    """
    Barplot de imbalance_ratio por dataset
    """
    df_split = df[df["split"] == split].copy()
    df_plot = df_split.sort_values("imbalance_ratio", ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(
        data=df_plot,
        x="imbalance_ratio",
        y="dataset",
        palette="viridis"
    )
    plt.title(f"Imbalance Ratio por Dataset ({split})")
    plt.xlabel("Imbalance Ratio (max/min)")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.show()


def plot_entropy_gini_scatter(df, split="train"):
    """
    Scatter entropía vs gini, tamaño = n_samples
    """
    df_split = df[df["split"] == split].copy()

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        x=df_split["entropy"],
        y=df_split["gini_index"],
        s=df_split["n_samples"]/500,   # escala para tamaños
        color="steelblue",
        alpha=0.7,
        edgecolor="k"
    )

    for _, row in df_split.iterrows():
        plt.text(
            row["entropy"]+0.02, row["gini_index"]+0.01,
            row["dataset"], fontsize=8
        )

    plt.title(f"Entropía vs Gini ({split})")
    plt.xlabel("Entropía (bits)")
    plt.ylabel("Índice de Gini")
    plt.tight_layout()
    plt.show()

