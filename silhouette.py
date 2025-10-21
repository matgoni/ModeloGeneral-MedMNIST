from typing import List, Optional, Sequence, Union

import numpy as np
from importlib import import_module
from sklearn.metrics import silhouette_score

from medmnist import INFO


class SilhouetteAnalyzer:
    """Calculates mean silhouette coefficients for MedMNIST datasets or arbitrary embeddings."""

    def __init__(
        self,
        dataset_names: Union[str, Sequence[str]],
        split: str = "train",
        samples_per_dataset: int = 1000,
        random_state: int = 42,
        download: bool = False,
        metric: str = "euclidean",
    ) -> None:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names: List[str] = list(dataset_names)
        self.split = split
        self.samples_per_dataset = samples_per_dataset
        self.random_state = random_state
        self.download = download
        self.metric = metric

        self._check_datasets_exist()

        self._features: Optional[np.ndarray] = None
        self._dataset_labels: Optional[np.ndarray] = None
        self._class_labels: Optional[np.ndarray] = None

    def _check_datasets_exist(self) -> None:
        missing = [name for name in self.dataset_names if name not in INFO]
        if missing:
            raise ValueError(f"Datasets no encontrados en medmnist.INFO: {missing}")

    def _load_dataset(self, dataset_name: str):
        info = INFO[dataset_name]
        class_name = info["python_class"]
        dataset_class = getattr(import_module("medmnist"), class_name)
        ds = dataset_class(split=self.split, download=self.download)
        return ds

    def _prepare_features(self) -> None:
        rng = np.random.default_rng(self.random_state)
        feature_chunks = []
        dataset_chunks = []
        class_chunks = []

        for dataset_name in self.dataset_names:
            ds = self._load_dataset(dataset_name)
            total_samples = len(ds)
            sample_size = min(self.samples_per_dataset, total_samples)

            if sample_size < total_samples:
                indices = rng.choice(total_samples, size=sample_size, replace=False)
            else:
                indices = np.arange(total_samples)

            imgs = ds.imgs[indices]
            class_labels = self._extract_class_labels(ds.labels[indices])

            features = self._flatten_images(imgs)
            feature_chunks.append(features)
            dataset_chunks.append(np.full(features.shape[0], dataset_name))
            class_chunks.append(class_labels)

        self._features = np.concatenate(feature_chunks, axis=0)
        self._dataset_labels = np.concatenate(dataset_chunks, axis=0)
        self._class_labels = np.concatenate(class_chunks, axis=0)

    def _ensure_features(self) -> None:
        if self._features is None:
            self._prepare_features()

    def mean_score(self, label_level: str = "class") -> float:
        """Compute the mean silhouette coefficient using raw dataset pixels."""
        self._ensure_features()

        if label_level == "class":
            labels = self._class_labels
        elif label_level == "dataset":
            labels = self._dataset_labels
        else:
            raise ValueError("label_level debe ser 'class' o 'dataset'")

        self._validate_labels(labels)
        return float(silhouette_score(self._features, labels, metric=self.metric))

    def mean_score_from_embeddings(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute the mean silhouette coefficient using precomputed embeddings."""
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("El número de embeddings y etiquetas debe coincidir.")
        self._validate_labels(labels)
        return float(silhouette_score(embeddings, labels, metric=self.metric))

    @staticmethod
    def _validate_labels(labels: np.ndarray) -> None:
        unique = np.unique(labels)
        if unique.size < 2:
            raise ValueError("Se requieren al menos dos etiquetas diferentes para el coeficiente de silhouette.")

    @staticmethod
    def _extract_class_labels(raw_labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(raw_labels)
        if labels.ndim == 1:
            return labels
        if labels.ndim == 2:
            if labels.shape[1] == 1:
                return labels[:, 0]
            # Convert multi-label vectors into unique integers via binary encoding
            powers = 2 ** np.arange(labels.shape[1], dtype=np.int64)
            encoded = (labels.astype(np.int64) * powers).sum(axis=1)
            return encoded
        raise ValueError(f"Formato de etiquetas no soportado: {labels.shape}")

    @staticmethod
    def _flatten_images(imgs: np.ndarray) -> np.ndarray:
        imgs = imgs.astype("float32")
        if imgs.ndim == 4:
            channel_axis = None
            for axis in range(1, imgs.ndim):
                if imgs.shape[axis] in {1, 3}:
                    channel_axis = axis
                    break
            if channel_axis is not None and channel_axis != imgs.ndim - 1:
                imgs = np.moveaxis(imgs, channel_axis, -1)
            if imgs.shape[-1] == 1:
                imgs = imgs[..., 0]
            elif imgs.shape[-1] == 3:
                imgs = imgs.mean(axis=-1)
            else:
                imgs = imgs.reshape(imgs.shape[0], -1)
                return imgs / 255.0
        if imgs.ndim != 3:
            raise ValueError(f"Formato de imagen no soportado: {imgs.shape}")
        num_samples = imgs.shape[0]
        return imgs.reshape(num_samples, -1) / 255.0
