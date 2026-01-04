# --- Train del modelos EfficientNet-B0 -> clasificador general MedMNIST

"""Script para entrenar un modelo EfficientNet-B0 desde cero (pesos aleatorios) como
clasificador general de MedMNIST.

Combina múltiples conjuntos de MedMNIST en un único modelo, calcula la métrica
inicial antes de entrenar para medir la ganancia, y genera gráficas de pérdidas
y exactitud para visualizar la mejora a lo largo de las épocas.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from medmnist import INFO
from PIL import Image
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0



@dataclass
class RegistroDataset:
    """Estructura auxiliar para guardar metadatos de cada dataset."""

    nombre: str
    indice_inicial: int
    clases: int


class ConjuntoGeneralMedMNIST(Dataset):
    """Dataset que concatena múltiples conjuntos de MedMNIST en un solo objeto."""

    class _RepeatChannels:
        """Convierte imágenes de 1 canal a 3 canales preservando tensores."""

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat(3, 1, 1) if tensor.shape[0] == 1 else tensor

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
            info_dataset = INFO[nombre]
            clases = int(info_dataset.get("n_classes") or len(info_dataset["label"]))
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
                self._RepeatChannels(),
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
            etiqueta_array = np.array(etiqueta).squeeze()
            if etiqueta_array.ndim == 0:
                etiqueta = int(etiqueta_array)
            else:
                # Para etiquetas multi-etiqueta (p. ej., ChestMNIST) tomamos el índice con mayor probabilidad.
                etiqueta = int(np.argmax(etiqueta_array))
        etiqueta_global = offset + int(etiqueta)

        tensor_imagen = self.transformacion(imagen)
        return tensor_imagen, etiqueta_global, nombre_dataset

    def obtener_informacion(self) -> Dict[str, RegistroDataset]:
        """Devuelve el diccionario con offsets y clases por dataset."""
        return self._informacion


def construir_modelo(total_clases: int) -> nn.Module:
    """Inicializa EfficientNet-B0 con pesos aleatorios y ajusta la última capa."""
    modelo = efficientnet_b0(weights=None)

    # EfficientNet-B0 tiene la cabeza en classifier[1]
    in_features = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(in_features, total_clases)
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


def evaluar_baseline(
    modelo: nn.Module,
    cargador_entrenamiento: DataLoader,
    cargador_validacion: Optional[DataLoader],
    cargador_prueba: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
) -> Dict[str, Dict[str, object]]:
    """Evalúa el modelo sin entrenar para medir la ganancia inicial."""
    perdida_ent, exactitud_ent, _ = evaluar_modelo(
        modelo, cargador_entrenamiento, criterio, dispositivo
    )

    cargador_referencia: DataLoader
    nombre_referencia: str
    if cargador_validacion is not None:
        cargador_referencia = cargador_validacion
        nombre_referencia = "validacion"
    else:
        cargador_referencia = cargador_prueba
        nombre_referencia = "prueba"

    perdida_ref, exactitud_ref, _ = evaluar_modelo(
        modelo, cargador_referencia, criterio, dispositivo
    )

    return {
        "entrenamiento": {"perdida": perdida_ent, "exactitud": exactitud_ent},
        "referencia": {
            "nombre": nombre_referencia,
            "perdida": perdida_ref,
            "exactitud": exactitud_ref,
        },
    }


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


def _a_flotantes(valores: List[Optional[float]]) -> List[float]:
    """Convierte None en NaN para mantener la alineación en las gráficas."""
    return [float(valor) if valor is not None else float("nan") for valor in valores]


def generar_graficas(
    historial: List[Dict[str, Optional[float]]], ruta_reporte: Path
) -> Dict[str, str]:
    """Genera y guarda las gráficas de exactitud y pérdida."""
    epocas = [registro["epoca"] for registro in historial]
    exactitud_ent = _a_flotantes([registro.get("exactitud_entrenamiento") for registro in historial])
    exactitud_val = _a_flotantes([registro.get("exactitud_validacion") for registro in historial])
    perdida_ent = _a_flotantes([registro.get("perdida_entrenamiento") for registro in historial])
    perdida_val = _a_flotantes([registro.get("perdida_validacion") for registro in historial])

    ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
    ruta_exactitud = ruta_reporte.parent / f"{ruta_reporte.stem}_curva_exactitud.png"
    ruta_perdida = ruta_reporte.parent / f"{ruta_reporte.stem}_curva_perdida.png"

    plt.figure(figsize=(7, 4))
    plt.plot(epocas, exactitud_ent, marker="o", label="Entrenamiento")
    plt.plot(epocas, exactitud_val, marker="o", label="Validación/Prueba")
    plt.title("Exactitud vs épocas")
    plt.xlabel("Época")
    plt.ylabel("Exactitud")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_exactitud, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epocas, perdida_ent, marker="o", label="Entrenamiento")
    plt.plot(epocas, perdida_val, marker="o", label="Validación/Prueba")
    plt.title("Pérdida vs épocas")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_perdida, dpi=150)
    plt.close()

    return {"exactitud": str(ruta_exactitud), "perdida": str(ruta_perdida)}


def guardar_reporte(
    ruta_salida: Path,
    argumentos: argparse.Namespace,
    historial: List[Dict[str, float]],
    metricas_prueba: Dict[str, object],
    informacion_datasets: Dict[str, RegistroDataset],
    metricas_iniciales: Dict[str, object],
    rutas_graficas: Dict[str, str],
) -> None:
    """Genera un archivo JSON con la configuración y resultados del entrenamiento."""
    reporte = {
        "fecha": datetime.now().isoformat(),
        "configuracion": vars(argumentos),
        "historial_epocas": historial,
        "metricas_iniciales": metricas_iniciales,
        "metricas_prueba": metricas_prueba,
        "graficas": rutas_graficas,
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
        description="Entrena una EfficientNet-B0 desde cero para un clasificador general de MedMNIST."
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
        "--dispositivo",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo a utilizar (cuda o cpu).",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("resultados") / "reporte_efficientnet_scratch.json",
        help="Ruta del archivo JSON donde se guardarán los resultados y gráficas.",
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
        total_clases=sum(info.clases for info in informacion_datasets.values())
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

    metricas_iniciales = evaluar_baseline(
        modelo=modelo,
        cargador_entrenamiento=cargador_entrenamiento,
        cargador_validacion=cargador_validacion,
        cargador_prueba=cargador_prueba,
        criterio=criterio,
        dispositivo=dispositivo,
    )

    print(
        f"Exactitud inicial ({metricas_iniciales['referencia']['nombre']}): "
        f"{metricas_iniciales['referencia']['exactitud']:.4f}"
    )

    historial_base = [
        {
            "epoca": 0,
            "perdida_entrenamiento": metricas_iniciales["entrenamiento"]["perdida"],
            "exactitud_entrenamiento": metricas_iniciales["entrenamiento"]["exactitud"],
            "perdida_validacion": metricas_iniciales["referencia"]["perdida"],
            "exactitud_validacion": metricas_iniciales["referencia"]["exactitud"],
        }
    ]

    historial_entrenamiento, mejor_estado = recorrer_entrenamiento(
        modelo=modelo,
        cargador_entrenamiento=cargador_entrenamiento,
        cargador_validacion=cargador_validacion,
        criterio=criterio,
        optimizador=optimizador,
        scheduler=scheduler,
        epocas=argumentos.epocas,
        dispositivo=dispositivo,
    )
    historial = historial_base + historial_entrenamiento

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

    rutas_graficas = generar_graficas(historial=historial, ruta_reporte=argumentos.salida)

    guardar_reporte(
        ruta_salida=argumentos.salida,
        argumentos=argumentos,
        historial=historial,
        metricas_prueba=metricas_prueba,
        informacion_datasets=informacion_datasets,
        metricas_iniciales=metricas_iniciales,
        rutas_graficas=rutas_graficas,
    )


if __name__ == "__main__":
    main()
