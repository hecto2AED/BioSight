from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

# =========================
# Configuración por defecto
# =========================

DEFAULT_WEIGHTS_PATH = r"biosight_app/weights/best.pt"
DEFAULT_OUTPUT_DIR = Path("outputs")

CONF_THRESHOLD = 0.43
IOU_THRESHOLD = 0.45
INFER_IMGSZ = 1024
INFER_MAX_DET = 3000

_MODEL_CACHE: Dict[str, Any] = {
    "path": None,
    "model": None,
}


# =========================
# Utilidades generales
# =========================

def get_best_device() -> str:
    """Selecciona automáticamente el mejor dispositivo disponible."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Crea una carpeta si no existe y devuelve la ruta como Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_model_names(names: Union[Dict[Any, Any], list, tuple]) -> Dict[int, str]:
    """
    Normaliza model.names a un diccionario {id_clase: nombre_clase}.
    Ultralytics puede devolver dict o lista según versión/modelo.
    """
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {int(i): str(v) for i, v in enumerate(names)}
    raise TypeError(f"Formato no soportado para model.names: {type(names)}")


def find_class_id(names: Union[Dict[Any, Any], list, tuple], aliases) -> Optional[int]:
    """Busca el id de clase a partir de una lista de alias."""
    aliases = {str(a).strip().lower() for a in aliases}
    names_dict = normalize_model_names(names)

    for class_id, class_name in names_dict.items():
        if str(class_name).strip().lower() in aliases:
            return int(class_id)
    return None


def resolve_class_ids(model: YOLO) -> tuple[Dict[int, str], int, int]:
    """Obtiene los IDs de clases para vivas y muertas."""
    names = normalize_model_names(model.names)

    alive_id = find_class_id(names, ["alive", "a"])
    dead_id = find_class_id(names, ["dead", "d"])

    if alive_id is None or dead_id is None:
        raise ValueError(
            f"No se pudieron identificar las clases de vivas/muertas en model.names = {names}"
        )

    return names, alive_id, dead_id


# =========================
# Carga del modelo
# =========================

def load_model(weights_path: Union[str, Path] = DEFAULT_WEIGHTS_PATH) -> YOLO:
    """Carga el modelo YOLO y lo deja en caché para reutilizarlo."""
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos: {weights_path.resolve()}"
        )

    cached_path = _MODEL_CACHE["path"]
    if cached_path is not None and Path(cached_path).resolve() == weights_path.resolve():
        return _MODEL_CACHE["model"]

    model = YOLO(str(weights_path))

    _MODEL_CACHE["path"] = str(weights_path.resolve())
    _MODEL_CACHE["model"] = model
    return model


# =========================
# Preparación de imagen
# =========================

def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Convierte un array a uint8 de forma robusta."""
    if array.dtype == np.uint8:
        return array

    arr = np.asarray(array)

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        max_value = float(arr.max()) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _read_image_as_rgb(
    image_input: Union[str, Path, Image.Image, np.ndarray]
) -> tuple[Image.Image, Optional[str]]:
    """
    Lee una imagen desde ruta, PIL o numpy y la devuelve en RGB.
    También devuelve el nombre original si estaba disponible.
    """
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"No se encontró la imagen: {image_path.resolve()}")

        with Image.open(image_path) as img:
            return img.convert("RGB"), image_path.name

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB"), None

    if isinstance(image_input, np.ndarray):
        arr = _to_uint8(image_input)

        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB"), None

        if arr.ndim == 3 and arr.shape[2] == 3:
            return Image.fromarray(arr).convert("RGB"), None

        if arr.ndim == 3 and arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA").convert("RGB"), None

        raise ValueError(
            f"Array de imagen no soportado. Forma recibida: {arr.shape}"
        )

    raise TypeError(
        "image_input debe ser una ruta, un objeto PIL.Image o un numpy.ndarray"
    )


def prepare_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    image_name: Optional[str] = None,
) -> tuple[Path, str]:
    """
    Convierte la imagen a RGB y la guarda temporalmente en PNG.
    Devuelve:
    - ruta temporal RGB
    - nombre original usado para reportes
    """
    output_dir = ensure_dir(output_dir)
    temp_rgb_dir = ensure_dir(output_dir / "_tmp_rgb")

    rgb_image, detected_name = _read_image_as_rgb(image_input)

    original_name = image_name or detected_name or f"uploaded_{uuid.uuid4().hex[:8]}.png"
    stem = Path(original_name).stem
    temp_rgb_path = temp_rgb_dir / f"{stem}_{uuid.uuid4().hex[:8]}.png"

    rgb_image.save(temp_rgb_path)
    return temp_rgb_path, original_name


# =========================
# Métricas de conteo
# =========================

def compute_counts(result, alive_id: int, dead_id: int) -> Dict[str, float]:
    """
    Calcula:
    - Cantidad Total
    - Cantidad Vivas
    - Cantidad Muertas
    - % Viabilidad
    - Confianza media
    """
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {
            "Cantidad Total": 0,
            "Cantidad Vivas": 0,
            "Cantidad Muertas": 0,
            "% Viabilidad": 0.0,
            "Confianza media": 0.0,
        }

    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()

    alive_count = int((class_ids == alive_id).sum())
    dead_count = int((class_ids == dead_id).sum())
    total_count = int(len(class_ids))
    viability_pct = (100.0 * alive_count / total_count) if total_count > 0 else 0.0
    mean_conf = float(confs.mean()) if len(confs) > 0 else 0.0

    return {
        "Cantidad Total": total_count,
        "Cantidad Vivas": alive_count,
        "Cantidad Muertas": dead_count,
        "% Viabilidad": round(viability_pct, 2),
        "Confianza media": round(mean_conf, 4),
    }


# =========================
# Guardado de salidas
# =========================

def _build_unique_path(directory: Path, filename: str) -> Path:
    """Genera una ruta única si el archivo ya existe."""
    directory = ensure_dir(directory)
    path = directory / filename

    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    unique_name = f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    return directory / unique_name


def save_annotated_image(result, output_dir: Union[str, Path], original_name: str) -> Path:
    """Guarda la imagen anotada como PNG."""
    annotated_dir = ensure_dir(Path(output_dir) / "annotated")
    annotated_name = f"{Path(original_name).stem}_annotated.png"
    annotated_path = _build_unique_path(annotated_dir, annotated_name)

    result.save(filename=str(annotated_path))
    return annotated_path


def save_results_csv(row: Dict[str, Any], output_dir: Union[str, Path], original_name: str) -> Path:
    """Guarda un CSV de una sola fila con el resumen de la imagen."""
    csv_dir = ensure_dir(Path(output_dir) / "csv")
    csv_name = f"{Path(original_name).stem}_results.csv"
    csv_path = _build_unique_path(csv_dir, csv_name)

    df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    return csv_path


# =========================
# Función principal
# =========================

def predict_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    weights_path: Union[str, Path] = DEFAULT_WEIGHTS_PATH,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    conf: float = CONF_THRESHOLD,
    iou: float = IOU_THRESHOLD,
    imgsz: int = INFER_IMGSZ,
    max_det: int = INFER_MAX_DET,
    device: Optional[str] = None,
    image_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ejecuta inferencia sobre una imagen y devuelve:
    - métricas de conteo
    - ruta de imagen anotada
    - ruta de CSV
    - metadatos del modelo y dispositivo
    """
    output_dir = ensure_dir(output_dir)
    device = device or get_best_device()

    temp_rgb_path, original_name = prepare_image(
        image_input=image_input,
        output_dir=output_dir,
        image_name=image_name,
    )

    try:
        model = load_model(weights_path)
        class_names, alive_id, dead_id = resolve_class_ids(model)

        results = model.predict(
            source=str(temp_rgb_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            save=False,
            verbose=False,
        )

        if not results:
            raise RuntimeError("El modelo no devolvió resultados de inferencia.")

        result = results[0]
        counts = compute_counts(result, alive_id, dead_id)

        annotated_path = save_annotated_image(
            result=result,
            output_dir=output_dir,
            original_name=original_name,
        )

        row = {
            "Imagen": original_name,
            "Cantidad Total": counts["Cantidad Total"],
            "Cantidad Vivas": counts["Cantidad Vivas"],
            "Cantidad Muertas": counts["Cantidad Muertas"],
            "% Viabilidad": counts["% Viabilidad"],
            "Confianza media": counts["Confianza media"],
            "Ruta imagen anotada": str(annotated_path),
        }

        csv_path = save_results_csv(
            row=row,
            output_dir=output_dir,
            original_name=original_name,
        )

        return {
            "image_name": original_name,
            "annotated_image_path": str(annotated_path),
            "csv_path": str(csv_path),
            "metrics": {
                "total": counts["Cantidad Total"],
                "alive": counts["Cantidad Vivas"],
                "dead": counts["Cantidad Muertas"],
                "viability_pct": counts["% Viabilidad"],
                "mean_confidence": counts["Confianza media"],
            },
            "row": row,
            "class_names": class_names,
            "device": device,
            "weights_path": str(Path(weights_path).resolve()),
        }

    finally:
        try:
            if temp_rgb_path.exists():
                temp_rgb_path.unlink()
        except Exception:
            pass


# =========================
# Test manual por terminal
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inferencia de conteo celular con YOLO.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen a procesar.")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Ruta al archivo de pesos (.pt).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta donde guardar PNG anotado y CSV.",
    )
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Umbral de confianza.")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="Umbral de IoU.")
    parser.add_argument("--imgsz", type=int, default=INFER_IMGSZ, help="Tamaño de inferencia.")
    parser.add_argument("--max-det", type=int, default=INFER_MAX_DET, help="Máximo de detecciones.")
    parser.add_argument(
        "--device",
        default=None,
        help='Dispositivo de inferencia (por ejemplo: "cpu", "cuda:0", "mps").',
    )

    args = parser.parse_args()

    output = predict_image(
        image_input=args.image,
        weights_path=args.weights,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))