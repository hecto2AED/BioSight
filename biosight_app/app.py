from __future__ import annotations

from pathlib import Path

import gradio as gr

from predictor import (
    CONF_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEIGHTS_PATH,
    INFER_IMGSZ,
    IOU_THRESHOLD,
    predict_image,
)

APP_TITLE = "BioSight - Cell Counter"
APP_DESCRIPTION = (
    "Sube una imagen de hemocitómetro, ejecuta la inferencia con YOLO11 "
    "y obtén la imagen anotada junto con el recuento celular y el CSV."
)


def format_metrics_markdown(result: dict) -> str:
    """Convierte las métricas del predictor en un bloque Markdown legible."""
    metrics = result["metrics"]
    image_name = result["image_name"]
    device = result["device"]

    return f"""
## Resultados

**Imagen procesada:** `{image_name}`  
**Dispositivo:** `{device}`

| Métrica | Valor |
|---|---:|
| Cantidad total | {metrics["total"]} |
| Cantidad vivas | {metrics["alive"]} |
| Cantidad muertas | {metrics["dead"]} |
| Viabilidad (%) | {metrics["viability_pct"]} |
| Confianza media | {metrics["mean_confidence"]} |

### Parámetros usados
- **Pesos:** `{Path(DEFAULT_WEIGHTS_PATH)}`
- **Confianza:** `{CONF_THRESHOLD}`
- **IoU:** `{IOU_THRESHOLD}`
- **Tamaño de inferencia:** `{INFER_IMGSZ}`
""".strip()


def run_inference(image):
    """
    Recibe una imagen desde Gradio, llama al predictor y devuelve:
    - imagen anotada
    - markdown con métricas
    - CSV descargable
    - PNG anotado descargable
    """
    if image is None:
        raise gr.Error("Debes subir una imagen antes de procesarla.")

    try:
        result = predict_image(
            image_input=image,
            weights_path=DEFAULT_WEIGHTS_PATH,
            output_dir=DEFAULT_OUTPUT_DIR,
        )
    except Exception as e:
        raise gr.Error(f"Error durante la inferencia: {e}")

    annotated_image_path = result["annotated_image_path"]
    csv_path = result["csv_path"]
    metrics_md = format_metrics_markdown(result)

    return annotated_image_path, metrics_md, csv_path, annotated_image_path


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(APP_DESCRIPTION)

    with gr.Row():
        input_image = gr.Image(
            label="Imagen de entrada",
            type="pil",
        )
        output_image = gr.Image(
            label="Imagen anotada",
            type="filepath",
        )

    metrics_output = gr.Markdown("## Resultados\nSube una imagen y pulsa **Procesar imagen**.")

    with gr.Row():
        csv_output = gr.File(label="Descargar CSV")
        png_output = gr.File(label="Descargar PNG anotado")

    with gr.Row():
        process_button = gr.Button("Procesar imagen", variant="primary")
        clear_button = gr.ClearButton(
            components=[input_image, output_image, metrics_output, csv_output, png_output],
            value="Limpiar",
        )

    process_button.click(
        fn=run_inference,
        inputs=input_image,
        outputs=[output_image, metrics_output, csv_output, png_output],
    )


if __name__ == "__main__":
    Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    demo.launch()