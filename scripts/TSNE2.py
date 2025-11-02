# ---- TSNE combinado para múltiples datasets de MedMNIST

import inspect
from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from importlib import import_module
from sklearn.manifold import TSNE

from medmnist import INFO


class TSNEVisualizer:
    """t-SNE helper that can combine multiple MedMNIST datasets in a single projection."""

    def __init__(
        self,
        dataset_names: Union[str, Sequence[str]],
        split: str = "train",
        samples_per_dataset: int = 1000,
        perplexity: float = 30.0,
        color_mode: str = "grayscale",
        random_state: int = 42,
        download: bool = False,
        verbose: int = 1,
    ) -> None:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        self.dataset_names: List[str] = list(dataset_names)
        self.split = split
        self.samples_per_dataset = samples_per_dataset
        self.perplexity = perplexity
        allowed_color_modes = {"grayscale", "original"}
        if color_mode not in allowed_color_modes:
            raise ValueError(f"color_mode debe ser uno de {allowed_color_modes}")
        self.color_mode = color_mode
        self.random_state = random_state
        self.download = download
        self.verbose = verbose

        self._check_datasets_exist()

        self._embeddings: Optional[np.ndarray] = None
        self._features: Optional[np.ndarray] = None
        self._dataset_labels: Optional[np.ndarray] = None
        self._class_labels: Optional[np.ndarray] = None
        self._label_maps = {}

    def _check_datasets_exist(self) -> None:
        missing = [name for name in self.dataset_names if name not in INFO]
        if missing:
            raise ValueError(f"Datasets no encontrados en medmnist.INFO: {missing}")

    def _load_dataset(self, dataset_name: str):
        info = INFO[dataset_name]
        class_name = info["python_class"]
        dataset_class = getattr(import_module("medmnist"), class_name)
        ds = dataset_class(split=self.split, download=self.download)
        labels_map = info.get("label", None)
        return ds, labels_map

    def _prepare_features(self) -> None:
        rng = np.random.default_rng(self.random_state)
        image_batches = []
        dataset_chunks = []
        class_chunks = []

        for dataset_name in self.dataset_names:
            ds, labels_map = self._load_dataset(dataset_name)
            total_samples = len(ds)
            sample_size = min(self.samples_per_dataset, total_samples)

            if sample_size < total_samples:
                indices = rng.choice(total_samples, size=sample_size, replace=False)
            else:
                indices = np.arange(total_samples)

            imgs = self._normalize_batch(ds.imgs[indices])
            labels = ds.labels[indices].reshape(-1)

            image_batches.append(imgs)
            dataset_chunks.append(np.full(imgs.shape[0], dataset_name))
            class_chunks.append(labels)
            self._label_maps[dataset_name] = labels_map

        processed_batches = self._apply_color_mode(image_batches)
        feature_chunks = [self._flatten_images(batch) for batch in processed_batches]
        feature_dims = {chunk.shape[1] for chunk in feature_chunks}
        if len(feature_dims) != 1:
            raise ValueError(
                "No se pudieron alinear las dimensiones de las características en modo "
                f"{self.color_mode}. Dimensiones detectadas: {sorted(feature_dims)}"
            )

        self._features = np.concatenate(feature_chunks, axis=0)
        self._dataset_labels = np.concatenate(dataset_chunks, axis=0)
        self._class_labels = np.concatenate(class_chunks, axis=0)

    def _ensure_features(self) -> None:
        if self._features is None:
            self._prepare_features()

    def fit_transform(self) -> np.ndarray:
        self._ensure_features()
        tsne_params = dict(
            n_components=2,
            perplexity=self.perplexity,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        # scikit-learn renamed n_iter to max_iter starting at 1.8.0
        if "max_iter" in inspect.signature(TSNE.__init__).parameters:
            tsne_params["max_iter"] = 2000
        else:
            tsne_params["n_iter"] = 2000

        tsne = TSNE(**tsne_params)
        self._embeddings = tsne.fit_transform(self._features)
        return self._embeddings

    def plot(self, color_by: str = "dataset") -> None:
        if color_by not in {"dataset", "class"}:
            raise ValueError("color_by debe ser 'dataset' o 'class'")

        if self._embeddings is None:
            self.fit_transform()

        plt.figure(figsize=(10, 8))

        if color_by == "dataset":
            self._plot_by_dataset()
            title = f"t-SNE combinado por dataset ({', '.join(self.dataset_names)})"
        else:
            self._plot_by_class()
            title = f"t-SNE combinado por clase ({', '.join(self.dataset_names)})"

        plt.title(title)
        plt.xlabel("t-SNE Componente 1")
        plt.ylabel("t-SNE Componente 2")
        plt.legend(loc="best", fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _plot_by_dataset(self) -> None:
        unique_datasets = list(dict.fromkeys(self._dataset_labels))
        cmap = plt.cm.get_cmap("tab10", len(unique_datasets))

        for idx, dataset_name in enumerate(unique_datasets):
            mask = self._dataset_labels == dataset_name
            plt.scatter(
                self._embeddings[mask, 0],
                self._embeddings[mask, 1],
                color=cmap(idx),
                label=dataset_name,
                alpha=0.7,
                s=25,
            )

    def _plot_by_class(self) -> None:
        color_palette = plt.cm.get_cmap("tab20")
        seen_labels = set()
        offset = 0

        for dataset_name in self.dataset_names:
            mask = self._dataset_labels == dataset_name
            dataset_embeddings = self._embeddings[mask]
            dataset_classes = self._class_labels[mask]
            labels_map = self._label_maps.get(dataset_name)

            unique_classes = sorted(np.unique(dataset_classes))
            for class_idx, class_value in enumerate(unique_classes):
                class_mask = dataset_classes == class_value
                readable = self._lookup_label(labels_map, class_value)
                label_name = f"{dataset_name} - {readable}"
                color = color_palette((offset + class_idx) % color_palette.N)
                legend_label = label_name if label_name not in seen_labels else None
                seen_labels.add(label_name)

                plt.scatter(
                    dataset_embeddings[class_mask, 0],
                    dataset_embeddings[class_mask, 1],
                    color=color,
                    label=legend_label,
                    alpha=0.7,
                    s=20,
                )
            offset += len(unique_classes)

    @staticmethod
    def _lookup_label(labels_map: Optional[dict], class_value: Union[int, np.integer]) -> str:
        if labels_map is None:
            return f"clase {int(class_value)}"
        key_variants = [str(int(class_value)), int(class_value)]
        for key in key_variants:
            if key in labels_map:
                return labels_map[key]
        return f"clase {int(class_value)}"

    def _normalize_batch(self, imgs: np.ndarray) -> np.ndarray:
        imgs = imgs.astype("float32")
        if imgs.ndim == 4:
            channel_axis = None
            for axis in range(1, imgs.ndim):
                if imgs.shape[axis] in {1, 3}:
                    channel_axis = axis
                    break
            if channel_axis is None:
                raise ValueError(f"No se encontró eje de canales en imágenes con forma {imgs.shape}")
            if channel_axis != imgs.ndim - 1:
                imgs = np.moveaxis(imgs, channel_axis, -1)
        elif imgs.ndim == 3:
            imgs = imgs[..., np.newaxis]
        else:
            raise ValueError(f"Formato de imagen no soportado: {imgs.shape}")
        return imgs

    def _apply_color_mode(self, batches: List[np.ndarray]) -> List[np.ndarray]:
        processed: List[np.ndarray] = []

        if self.color_mode == "grayscale":
            for batch in batches:
                if batch.ndim == 4 and batch.shape[-1] == 1:
                    processed.append(batch[..., 0])
                elif batch.ndim == 4:
                    processed.append(batch.mean(axis=-1))
                elif batch.ndim == 3:
                    processed.append(batch)
                else:
                    raise ValueError(f"No se puede convertir a escala de grises el lote con forma {batch.shape}")
            return processed

        # color_mode == "original"
        has_color = any(batch.shape[-1] > 1 for batch in batches)
        target_channels = 3 if has_color else 1

        for batch in batches:
            if batch.ndim == 3:
                batch = batch[..., np.newaxis]
            if batch.ndim != 4:
                raise ValueError(f"Formato de imagen no soportado: {batch.shape}")

            channels = batch.shape[-1]
            if channels == target_channels:
                processed.append(batch)
            elif channels == 1 and target_channels == 3:
                processed.append(np.repeat(batch, 3, axis=-1))
            elif channels == 3 and target_channels == 1:
                processed.append(batch.mean(axis=-1, keepdims=True))
            else:
                raise ValueError(
                    f"No se puede ajustar el número de canales de {channels} a {target_channels} para forma {batch.shape}"
                )

        if any(batch.shape[-1] != target_channels for batch in processed):
            shapes = [batch.shape for batch in processed]
            raise ValueError(f"Inconsistencia en los canales tras procesar: {shapes}")

        return processed

    @staticmethod
    def _flatten_images(imgs: np.ndarray) -> np.ndarray:
        if imgs.ndim not in {3, 4}:
            raise ValueError(f"Formato de imagen no soportado tras preprocesamiento: {imgs.shape}")

        num_samples = imgs.shape[0]
        return imgs.reshape(num_samples, -1) / 255.0
