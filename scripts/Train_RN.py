# --- Train del modelos ResNet-18 -> clasificador general MedMNIST

"""Script para entrenar un modelo ResNet-18 como clasificador general de MedMNIST.

Este módulo permite combinar múltiples conjuntos de MedMNIST en un único modelo,
ajustar hiperparámetros desde la línea de comandos, generar reportes de métricas
y evaluar el modelo resultante en el conjunto de test.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from medmnist import INFO
from PIL import Image
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class RegistroDataset:
    """Estructura auxiliar para guardar metadatos de cada dataset."""

    nombre: str
    indice_inicial: int
    clases: int


class ConjuntoGeneralMedMNIST(Dataset):
    """Dataset que concatena múltiples conjuntos de MedMNIST en un solo objeto."""

    def __init__(
        self,
        nombres_dataset: Iterable[str],
        split: str,
        descarga: bool,
        tamano_imagen: int,
        usar_aumentos: bool,
    ) -> None:
        self.nombres_dataset = list(nombres_dataset)
        if not self.nombres_dataset:
            raise ValueError("Se requiere al menos un nombre de dataset.")

        self.split = split
        self.descarga = descarga
        self.tamano_imagen = tamano_imagen
        self.usar_aumentos = usar_aumentos

        self._registros: List[Dict[str, object]] = []
        self._informacion: Dict[str, RegistroDataset] = {}
        self.total_clases = 0

        self._construir_registros()
        self.transformacion = self._construir_transformacion()

    def _construir_registros(self) -> None:
        """Descarga los datasets y arma la lista de muestras con offsets de clase."""
        for nombre in self.nombres_dataset:
            if nombre not in INFO:
                raise ValueError(f"Dataset '{nombre}' no existe en medmnist.INFO.")

            clase_python = INFO[nombre]["python_class"]
            dataset_clase = getattr(import_module("medmnist"), clase_python)
            dataset = dataset_clase(split=self.split, download=self.descarga)
            clases = int(INFO[nombre]["n_classes"])
            indice_base = self.total_clases
            self._informacion[nombre] = RegistroDataset(
                nombre=nombre, indice_inicial=indice_base, clases=clases
            )

            for indice in range(len(dataset)):
                self._registros.append(
                    {
                        "dataset": dataset,
                        "indice": indice,
                        "nombre": nombre,
                        "offset": indice_base,
                    }
                )

            self.total_clases += clases

    def _construir_transformacion(self) -> transforms.Compose:
        """Crea las transformaciones de preprocesamiento y aumentos."""
        lista_transformaciones: List[transforms.Compose] = [
            transforms.Resize((self.tamano_imagen, self.tamano_imagen)),
        ]
        if self.usar_aumentos and self.split == "train":
            lista_transformaciones.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=15),
                ]
            )

        lista_transformaciones.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda tensor: tensor.repeat(3, 1, 1)
                    if tensor.shape[0] == 1
                    else tensor
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return transforms.Compose(lista_transformaciones)

    @staticmethod
    def _a_pil(imagen: np.ndarray) -> Image.Image:
        """Convierte un arreglo numpy a imagen PIL manteniendo el modo correcto."""
        if imagen.ndim == 2:
            return Image.fromarray(imagen.astype(np.uint8), mode="L")
        if imagen.ndim == 3:
            return Image.fromarray(imagen.astype(np.uint8))
        raise ValueError(f"Dimensiones de imagen no soportadas: {imagen.shape}")

    def __len__(self) -> int:
        return len(self._registros)

    def __getitem__(self, indice: int) -> Tuple[torch.Tensor, int, str]:
        registro = self._registros[indice]
        dataset = registro["dataset"]
        indice_local = int(registro["indice"])
        nombre_dataset = str(registro["nombre"])
        offset = int(registro["offset"])

        imagen, etiqueta = dataset[indice_local]
        if isinstance(imagen, np.ndarray):
            imagen = self._a_pil(imagen)
        if isinstance(etiqueta, (np.ndarray, list)):
            etiqueta = int(np.array(etiqueta).squeeze())
        etiqueta_global = offset + int(etiqueta)

        tensor_imagen = self.transformacion(imagen)
        return tensor_imagen, etiqueta_global, nombre_dataset

    def obtener_informacion(self) -> Dict[str, RegistroDataset]:
        """Devuelve el diccionario con offsets y clases por dataset."""
        return self._informacion


def construir_modelo(total_clases: int, usar_pesos_imagenet: bool) -> nn.Module:
    """Inicializa la ResNet-18 ajustando la capa de salida al número total de clases."""
    if usar_pesos_imagenet:
        pesos = ResNet18_Weights.DEFAULT
        modelo = resnet18(weights=pesos)
    else:
        modelo = resnet18(weights=None)
    caracteristicas = modelo.fc.in_features
    modelo.fc = nn.Linear(caracteristicas, total_clases)
    return modelo


def crear_cargadores(
    nombres_dataset: Iterable[str],
    tamano_lote: int,
    descarga: bool,
    tamano_imagen: int,
    usar_aumentos: bool,
    trabajadores: int,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, Dict[str, RegistroDataset]]:
    """Crea y devuelve los DataLoaders de entrenamiento, validación y prueba."""
    conjunto_entrenamiento = ConjuntoGeneralMedMNIST(
        nombres_dataset=nombres_dataset,
        split="train",
        descarga=descarga,
        tamano_imagen=tamano_imagen,
        usar_aumentos=usar_aumentos,
    )

    try:
        conjunto_validacion = ConjuntoGeneralMedMNIST(
            nombres_dataset=nombres_dataset,
            split="val",
            descarga=descarga,
            tamano_imagen=tamano_imagen,
            usar_aumentos=False,
        )
    except Exception:
        conjunto_validacion = None

    conjunto_prueba = ConjuntoGeneralMedMNIST(
        nombres_dataset=nombres_dataset,
        split="test",
        descarga=descarga,
        tamano_imagen=tamano_imagen,
        usar_aumentos=False,
    )

    cargador_entrenamiento = DataLoader(
        conjunto_entrenamiento,
        batch_size=tamano_lote,
        shuffle=True,
        num_workers=trabajadores,
        pin_memory=True,
    )
    cargador_validacion = (
        DataLoader(
            conjunto_validacion,
            batch_size=tamano_lote,
            shuffle=False,
            num_workers=trabajadores,
            pin_memory=True,
        )
        if conjunto_validacion is not None
        else None
    )
    cargador_prueba = DataLoader(
        conjunto_prueba,
        batch_size=tamano_lote,
        shuffle=False,
        num_workers=trabajadores,
        pin_memory=True,
    )

    return (
        cargador_entrenamiento,
        cargador_validacion,
        cargador_prueba,
        conjunto_entrenamiento.obtener_informacion(),
    )


def mover_a_dispositivo(tesoros: Iterable[torch.Tensor], dispositivo: torch.device) -> Tuple[torch.Tensor, ...]:
    """Envía una colección de tensores al dispositivo elegido."""
    return tuple(tensor.to(dispositivo, non_blocking=True) for tensor in tesoros)


def entrenar_epoca(
    modelo: nn.Module,
    cargador: DataLoader,
    criterio: nn.Module,
    optimizador: Optimizer,
    dispositivo: torch.device,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """Realiza una época de entrenamiento y devuelve métricas agregadas."""
    modelo.train()
    perdida_acumulada = 0.0
    ejemplos_totales = 0
    aciertos_totales = 0
    metricas_por_dataset: Dict[str, Dict[str, float]] = defaultdict(lambda: {"aciertos": 0, "ejemplos": 0})

    for imagenes, etiquetas, nombres in cargador:
        imagenes, etiquetas = mover_a_dispositivo((imagenes, etiquetas), dispositivo)
        optimizador.zero_grad(set_to_none=True)

        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()

        cantidad = etiquetas.size(0)
        predicciones = salidas.argmax(dim=1)
        aciertos = (predicciones == etiquetas).sum().item()

        perdida_acumulada += perdida.item() * cantidad
        ejemplos_totales += cantidad
        aciertos_totales += aciertos

        for nombre, etiqueta, prediccion in zip(nombres, etiquetas, predicciones):
            nombre_ds = str(nombre)
            metricas_por_dataset[nombre_ds]["ejemplos"] += 1
            if etiqueta.item() == prediccion.item():
                metricas_por_dataset[nombre_ds]["aciertos"] += 1

    perdida_promedio = perdida_acumulada / max(ejemplos_totales, 1)
    exactitud = aciertos_totales / max(ejemplos_totales, 1)

    for nombre, valores in metricas_por_dataset.items():
        ejemplos = max(valores["ejemplos"], 1)
        valores["exactitud"] = valores["aciertos"] / ejemplos

    return perdida_promedio, exactitud, metricas_por_dataset


@torch.no_grad()
def evaluar_modelo(
    modelo: nn.Module,
    cargador: DataLoader,
    criterio: Optional[nn.Module],
    dispositivo: torch.device,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """Evalúa el modelo y devuelve pérdida, exactitud y métricas por dataset."""
    modelo.eval()
    perdida_acumulada = 0.0
    ejemplos_totales = 0
    aciertos_totales = 0
    metricas_por_dataset: Dict[str, Dict[str, float]] = defaultdict(lambda: {"aciertos": 0, "ejemplos": 0})

    for imagenes, etiquetas, nombres in cargador:
        imagenes, etiquetas = mover_a_dispositivo((imagenes, etiquetas), dispositivo)
        salidas = modelo(imagenes)

        if criterio is not None:
            perdida = criterio(salidas, etiquetas)
            perdida_acumulada += perdida.item() * etiquetas.size(0)

        predicciones = salidas.argmax(dim=1)
        aciertos_totales += (predicciones == etiquetas).sum().item()
        ejemplos_totales += etiquetas.size(0)

        for nombre, etiqueta, prediccion in zip(nombres, etiquetas, predicciones):
            nombre_ds = str(nombre)
            metricas_por_dataset[nombre_ds]["ejemplos"] += 1
            if etiqueta.item() == prediccion.item():
                metricas_por_dataset[nombre_ds]["aciertos"] += 1

    perdida_promedio = (
        perdida_acumulada / max(ejemplos_totales, 1) if criterio is not None else 0.0
    )
    exactitud = aciertos_totales / max(ejemplos_totales, 1)

    for nombre, valores in metricas_por_dataset.items():
        ejemplos = max(valores["ejemplos"], 1)
        valores["exactitud"] = valores["aciertos"] / ejemplos

    return perdida_promedio, exactitud, metricas_por_dataset


def construir_scheduler(
    optimizador: Optimizer,
    tipo_scheduler: Optional[str],
    factor: float,
    paciencia: int,
) -> Optional[ReduceLROnPlateau]:
    """Construye un scheduler sencillo basado en ReduceLROnPlateau."""
    if tipo_scheduler is None:
        return None
    if tipo_scheduler.lower() != "plateau":
        raise ValueError("Actualmente solo se admite el scheduler 'plateau'.")
    return ReduceLROnPlateau(optimizador, mode="max", factor=factor, patience=paciencia)


def recorrer_entrenamiento(
    modelo: nn.Module,
    cargador_entrenamiento: DataLoader,
    cargador_validacion: Optional[DataLoader],
    criterio: nn.Module,
    optimizador: Optimizer,
    scheduler: Optional[ReduceLROnPlateau],
    epocas: int,
    dispositivo: torch.device,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Ejecuta el ciclo de entrenamiento completo y devuelve historial y mejor estado."""
    historial: List[Dict[str, float]] = []
    mejor_exactitud = 0.0
    mejor_estado: Dict[str, float] = {}

    for epoca in range(1, epocas + 1):
        perdida_entrenamiento, exactitud_entrenamiento, _ = entrenar_epoca(
            modelo, cargador_entrenamiento, criterio, optimizador, dispositivo
        )

        if cargador_validacion is not None:
            perdida_val, exactitud_val, _ = evaluar_modelo(
                modelo, cargador_validacion, criterio, dispositivo
            )
        else:
            perdida_val, exactitud_val = 0.0, 0.0

        if scheduler is not None:
            scheduler.step(exactitud_val if cargador_validacion is not None else exactitud_entrenamiento)

        if exactitud_val > mejor_exactitud and cargador_validacion is not None:
            mejor_exactitud = exactitud_val
            mejor_estado = modelo.state_dict()
        elif cargador_validacion is None and exactitud_entrenamiento > mejor_exactitud:
            mejor_exactitud = exactitud_entrenamiento
            mejor_estado = modelo.state_dict()

        historial.append(
            {
                "epoca": epoca,
                "perdida_entrenamiento": perdida_entrenamiento,
                "exactitud_entrenamiento": exactitud_entrenamiento,
                "perdida_validacion": perdida_val,
                "exactitud_validacion": exactitud_val,
            }
        )

        print(
            f"[Epoca {epoca:03d}] "
            f"Perdida ent: {perdida_entrenamiento:.4f} | "
            f"Exactitud ent: {exactitud_entrenamiento:.4f} | "
            f"Perdida val: {perdida_val:.4f} | "
            f"Exactitud val: {exactitud_val:.4f}"
        )

    return historial, mejor_estado


def guardar_reporte(
    ruta_salida: Path,
    argumentos: argparse.Namespace,
    historial: List[Dict[str, float]],
    metricas_prueba: Dict[str, object],
    informacion_datasets: Dict[str, RegistroDataset],
) -> None:
    """Genera un archivo JSON con la configuración y resultados del entrenamiento."""
    reporte = {
        "fecha": datetime.now().isoformat(),
        "configuracion": vars(argumentos),
        "historial_epocas": historial,
        "metricas_prueba": metricas_prueba,
        "datasets": {
            nombre: {
                "offset": info.indice_inicial,
                "clases": info.clases,
            }
            for nombre, info in informacion_datasets.items()
        },
    }
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, indent=2, ensure_ascii=False)


def parsear_argumentos() -> argparse.Namespace:
    """Define y parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena una ResNet-18 como modelo general para MedMNIST."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Lista de datasets MedMNIST a utilizar (ejemplo: bloodmnist pathmnist).",
    )
    parser.add_argument("--epocas", type=int, default=20, help="Número de épocas de entrenamiento.")
    parser.add_argument(
        "--tamano-lote",
        dest="tamano_lote",
        type=int,
        default=128,
        help="Tamaño de lote.",
    )
    parser.add_argument(
        "--tasa-aprendizaje",
        dest="tasa_aprendizaje",
        type=float,
        default=1e-3,
        help="Tasa de aprendizaje inicial.",
    )
    parser.add_argument(
        "--decaimiento-peso",
        dest="decaimiento_peso",
        type=float,
        default=1e-4,
        help="Factor de regularización L2 (weight decay).",
    )
    parser.add_argument(
        "--tamano-imagen",
        type=int,
        default=128,
        help="Tamaño al que se redimensionarán las imágenes.",
    )
    parser.add_argument(
        "--usar-aumentos",
        action="store_true",
        help="Habilita aumentos de datos simples en entrenamiento.",
    )
    parser.add_argument(
        "--descargar",
        action="store_true",
        help="Descarga los datasets si no están disponibles localmente.",
    )
    parser.add_argument(
        "--sin-preentrenar",
        action="store_true",
        help="Si se indica, no se cargan pesos preentrenados en ImageNet.",
    )
    parser.add_argument(
        "--dispositivo",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo a utilizar (cuda o cpu).",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("resultados") / "reporte_resnet_general.json",
        help="Ruta del archivo JSON donde se guardarán los resultados.",
    )
    parser.add_argument(
        "--trabajadores",
        type=int,
        default=2,
        help="Número de procesos auxiliares para cargar datos.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Tipo de scheduler a aplicar (opciones: plateau).",
    )
    parser.add_argument(
        "--factor-scheduler",
        type=float,
        default=0.5,
        help="Factor de reducción para el scheduler ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--paciencia-scheduler",
        type=int,
        default=3,
        help="Paciencia (en épocas) para el scheduler ReduceLROnPlateau.",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del script."""
    argumentos = parsear_argumentos()
    dispositivo = torch.device(argumentos.dispositivo)

    (
        cargador_entrenamiento,
        cargador_validacion,
        cargador_prueba,
        informacion_datasets,
    ) = crear_cargadores(
        nombres_dataset=argumentos.datasets,
        tamano_lote=argumentos.tamano_lote,
        descarga=argumentos.descargar,
        tamano_imagen=argumentos.tamano_imagen,
        usar_aumentos=argumentos.usar_aumentos,
        trabajadores=argumentos.trabajadores,
    )

    modelo = construir_modelo(
        total_clases=sum(info.clases for info in informacion_datasets.values()),
        usar_pesos_imagenet=not argumentos.sin_preentrenar,
    )
    modelo.to(dispositivo)

    criterio = nn.CrossEntropyLoss()
    optimizador = Adam(
        modelo.parameters(),
        lr=argumentos.tasa_aprendizaje,
        weight_decay=argumentos.decaimiento_peso,
    )
    scheduler = construir_scheduler(
        optimizador=optimizador,
        tipo_scheduler=argumentos.scheduler,
        factor=argumentos.factor_scheduler,
        paciencia=argumentos.paciencia_scheduler,
    )

    historial, mejor_estado = recorrer_entrenamiento(
        modelo=modelo,
        cargador_entrenamiento=cargador_entrenamiento,
        cargador_validacion=cargador_validacion,
        criterio=criterio,
        optimizador=optimizador,
        scheduler=scheduler,
        epocas=argumentos.epocas,
        dispositivo=dispositivo,
    )

    if mejor_estado:
        modelo.load_state_dict(mejor_estado)

    perdida_prueba, exactitud_prueba, metricas_por_dataset = evaluar_modelo(
        modelo, cargador_prueba, criterio, dispositivo
    )
    metricas_prueba = {
        "perdida": perdida_prueba,
        "exactitud": exactitud_prueba,
        "detalle_por_dataset": {
            nombre: {
                "exactitud": valores.get("exactitud", 0.0),
                "ejemplos": int(valores.get("ejemplos", 0)),
            }
            for nombre, valores in metricas_por_dataset.items()
        },
    }

    print(
        f"Exactitud en prueba: {exactitud_prueba:.4f} | "
        f"Perdida en prueba: {perdida_prueba:.4f}"
    )

    guardar_reporte(
        ruta_salida=argumentos.salida,
        argumentos=argumentos,
        historial=historial,
        metricas_prueba=metricas_prueba,
        informacion_datasets=informacion_datasets,
    )


if __name__ == "__main__":
    main()
