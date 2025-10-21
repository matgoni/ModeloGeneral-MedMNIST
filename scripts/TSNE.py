import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from importlib import import_module
from medmnist import INFO

def plot_tsne_clusters(
    dataset_name: str,
    split: str = "train",
    n_samples: int = 2000,
    random_state: int = 42,
    perplexity: int = 30,
    download: bool = False
):
    
    assert dataset_name in INFO, f"Dataset '{dataset_name}' no encontrado en medmnist.INFO"

    info = INFO[dataset_name]
    class_name = info["python_class"]
    DatasetClass = getattr(import_module("medmnist"), class_name)
    ds = DatasetClass(split=split, download=download)
    
    total_samples = len(ds)
    if total_samples > n_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(total_samples, size=n_samples, replace=False)
        imgs = ds.imgs[indices]
        labels = ds.labels[indices].squeeze()
    else:
        imgs = ds.imgs
        labels = ds.labels.squeeze()
        n_samples = total_samples
    

    X = imgs.reshape(imgs.shape[0], -1).astype('float32') / 255.0

    print(f"Aplicando t-SNE a {n_samples} muestras de {dataset_name}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=2000,
        random_state=random_state,
        verbose=1
    )
    X_embedded = tsne.fit_transform(X)
    
    # Mapeo de etiquetas
    labels_map = info.get("label", None)
    num_classes = info.get("n_classes", len(np.unique(labels)))
    
    plt.figure(figsize=(10, 8))
    
    if labels_map is None:
        unique_labels = sorted(np.unique(labels))
        class_names = [f"Clase {l}" for l in unique_labels]
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
    else:
        sorted_keys = sorted(labels_map.keys())
        class_names = [labels_map[k] for k in sorted_keys]
        colors = plt.cm.get_cmap('tab10', len(sorted_keys))

    for i in range(num_classes):
        if i in labels:
            indices_i = labels == i
            plt.scatter(
                X_embedded[indices_i, 0],
                X_embedded[indices_i, 1],
                color=colors(i),
                label=class_names[i],
                alpha=0.7,
                s=25
            )

    plt.title(f"Visualización t-SNE del dataset '{dataset_name}' (Split: {split})")
    plt.xlabel("t-SNE Componente 1")
    plt.ylabel("t-SNE Componente 2")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.show()

