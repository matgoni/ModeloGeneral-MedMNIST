# ----------------------------------------------------
# Correlación media entre datasets MedMNIST
# ----------------------------------------------------

from __future__ import annotations
from typing import Optional, Iterable
import numpy as np
import pandas as pd
from importlib import import_module
from medmnist import INFO
import matplotlib.pyplot as plt
import seaborn as sns

# si tienes la función listar_datasets_medmnist en otro archivo,
# puedes copiarla aquí o importarla:
def listar_datasets_medmnist(include_3d: bool = False) -> list[str]:
    """
    Lista los nombres de datasets registrados en medmnist.INFO.
    Por defecto excluye los 3D (p.ej. organmnist3d).
    """
    nombres = []
    for name, meta in INFO.items():
        pyclass = str(meta.get("python_class", "")).lower()
        es_3d = ("3d" in name.lower()) or ("3d" in pyclass)
        if es_3d and not include_3d:
            continue
        nombres.append(name)
    return sorted(nombres)


def correlation_between_datasets(
    datasets: Optional[Iterable[str]] = None,
    split: str = "train",
    sample_size: int = 1000,
    download: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calcula una matriz de correlación entre datasets de MedMNIST.
    Para cada dataset:
        - muestrea sample_size imágenes,
        - las aplana a vectores 1D,
        - calcula un vector medio de intensidades.
    Luego calcula la correlación de Pearson entre los vectores medios
    de cada par de datasets.
    Devuelve un DataFrame (datasets × datasets).
    """
    if datasets is None:
        datasets = listar_datasets_medmnist(include_3d=False)

    features = {}
    for idx, ds_name in enumerate(datasets):
        info = INFO[ds_name]
        class_name = info["python_class"]
        DatasetClass = getattr(import_module("medmnist"), class_name)
        ds = DatasetClass(split=split, download=download)
        imgs = ds.imgs

        # normalizamos a forma (n,h,w,c)
        if imgs.ndim == 3:  # (n,h,w)
            imgs = imgs[:,:,:,None]  # añade canal
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs.repeat(3,axis=-1)  # pseudo-RGB para correlación uniforme

        n_total = imgs.shape[0]
        rng = np.random.default_rng(random_state+idx)
        idxs = rng.choice(n_total, size=min(sample_size, n_total), replace=False)
        sample_imgs = imgs[idxs]

        # aplanar
        X = sample_imgs.reshape(sample_imgs.shape[0], -1).astype(np.float32)

        # vector medio
        features[ds_name] = X.mean(axis=0)

    # construir matriz de correlación
    names = list(features.keys())
    n = len(names)
    corr_matrix = np.zeros((n,n),dtype=float)

    for i in range(n):
        for j in range(n):
            if i<=j:
                r = np.corrcoef(features[names[i]], features[names[j]])[0,1]
                corr_matrix[i,j] = corr_matrix[j,i] = r

    df_corr = pd.DataFrame(corr_matrix, index=names, columns=names)
    return df_corr


def plot_dataset_correlation_heatmap(df_corr: pd.DataFrame, figsize=(10,8)):
    """
    Dibuja un heatmap de la matriz de correlación entre datasets.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlación media entre datasets MedMNIST")
    plt.tight_layout()
    plt.show()

