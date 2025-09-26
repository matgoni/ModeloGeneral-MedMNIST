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


# Histogramas
def plot_class_histograms(
    obj_detallado: Dict[str, Dict[str, Any]],
    split: str = "train",
    max_cols: int = 3,
    figsize_per_plot: Tuple[int, int] = (5, 3)
):
    
    datasets = [ds for ds in obj_detallado if split in obj_detallado[ds]]

    n = len(datasets)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows),
        squeeze=False
    )

    for idx, ds_name in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]

        info = obj_detallado[ds_name][split]
        counts = np.array(info["class_counts"])
        labels = info["labels"]
        if labels is not None:
            label_names = [labels[k] for k in sorted(labels.keys())]
        else:
            label_names = [str(i) for i in range(len(counts))]

        sns.barplot(x=np.arange(len(counts)), y=counts, ax=ax, palette="viridis")
        ax.set_title(f"{ds_name} ({split})")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Nº muestras")
        ax.set_xlabel("Clase")

    
    for j in range(idx+1, nrows*ncols):
        fig.delaxes(axes[j // ncols][j % ncols])

    fig.tight_layout()
    plt.show()


#Correlaciones
def plot_rgb_channel_correlations(
    datasets: Optional[Iterable[str]] = None,
    split: str = "train",
    sample_size: int = 1000,
    download: bool = False,
    random_state: int = 42,
    max_cols: int = 3,
    figsize_per_plot: Tuple[int,int] = (4,3)
):
   
    if datasets is None:
        datasets = listar_datasets_medmnist(include_3d=False)

    n = len(datasets)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows),
        squeeze=False
    )

    for idx, ds_name in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]

        info = INFO[ds_name]
        class_name = info["python_class"]
        DatasetClass = getattr(import_module("medmnist"), class_name)
        ds = DatasetClass(split=split, download=download)

        imgs = ds.imgs  # (n, h, w, c)
        if imgs.ndim != 4 or imgs.shape[-1] < 3:
            ax.axis("off")
            ax.set_title(f"{ds_name}\n(no RGB)")
            continue

        n_total = imgs.shape[0]
        rng = np.random.default_rng(random_state+idx)
        idxs = rng.choice(n_total, size=min(sample_size, n_total), replace=False)
        sample_imgs = imgs[idxs]

        # separar canales
        R = sample_imgs[:,:,:,0].ravel().astype(np.float32)
        G = sample_imgs[:,:,:,1].ravel().astype(np.float32)
        B = sample_imgs[:,:,:,2].ravel().astype(np.float32)

        data = np.vstack([R,G,B])
        corr = np.corrcoef(data)

        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=["R","G","B"], yticklabels=["R","G","B"],
            vmin=-1, vmax=1, ax=ax, cbar=False
        )
        ax.set_title(f"{ds_name} ({split})")

    
    for j in range(idx+1, nrows*ncols):
        fig.delaxes(axes[j // ncols][j % ncols])

    fig.tight_layout()
    plt.show()
