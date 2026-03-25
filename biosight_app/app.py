from __future__ import annotations

# =========================
# IMPORTS
# =========================
from pathlib import Path

import gradio as gr

# Importamos la lógica del modelo desde predictor.py
from predictor import (
    CONF_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEIGHTS_PATH,
    INFER_IMGSZ,
    IOU_THRESHOLD,
    predict_image,
)

# =========================
# TEXTOS DE LA APP
# =========================
APP_TITLE = "BioSight Cell Counter"
APP_DESCRIPTION = (
    "Sube una imagen de hemocitómetro "
    "y obtén la imagen anotada junto con el recuento celular, el porcentaje de viabilidad y la confianza media de la muestra."
)

# =========================
# RUTAS DE LA APP Y FUENTES
# =========================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

AKIRA_FONT = (FONTS_DIR / "akira.otf").resolve().as_posix()
ZENDOTS_FONT = (FONTS_DIR / "zendots.otf").resolve().as_posix()
JETBRAINS_FONT = (FONTS_DIR / "jetbrainsmono.otf").resolve().as_posix()
BITCOIN_FONT = (FONTS_DIR / "bitcoin.otf").resolve().as_posix()

# =========================
# CSS PERSONALIZADO
# =========================
CUSTOM_CSS = f"""
/* =========================
   FUENTES PERSONALIZADAS
   ========================= */
@font-face {{
    font-family: "AkiraCustom";
    src: url("/gradio_api/file={AKIRA_FONT}") format("opentype");
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}}

@font-face {{
    font-family: "ZendotsCustom";
    src: url("/gradio_api/file={ZENDOTS_FONT}") format("opentype");
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}}

@font-face {{
    font-family: "JetBrainsMonoCustom";
    src: url("/gradio_api/file={JETBRAINS_FONT}") format("opentype");
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}}

@font-face {{
    font-family: "BitcoinCustom";
    src: url("/gradio_api/file={BITCOIN_FONT}") format("opentype");
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}}

/* =========================
   FONDO GENERAL
   ========================= */
html, body, .gradio-container {{
    background: #131313 !important;
    color: #919191 !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
}}

.gradio-container,
.gradio-container * {{
    font-family: "JetBrainsMonoCustom", monospace !important;
    box-sizing: border-box;
}}

/* =========================
   TÍTULO PRINCIPAL
   ========================= */
#app-title {{
    margin-bottom: 0.6rem !important;
    text-align: left !important;
    line-height: 0.95 !important;
}}

#app-title,
#app-title * {{
    font-family: "AkiraCustom", sans-serif !important;
    color: #f5f5f5 !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em !important;
}}

#app-title .title-line {{
    display: block;
    font-size: 8rem !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 0.95 !important;
}}

/* =========================
   SUBTÍTULO / DESCRIPCIÓN
   ========================= */
#app-description {{
    margin-bottom: 1.25rem !important;
    text-align: left !important;
}}

#app-description,
#app-description * {{
    font-family: "JetBrainsMonoCustom", monospace !important;
    color: #07d839 !important;
    font-size: 2rem !important;
    line-height: 1.4 !important;
    margin: 0 !important;
}}

/* =========================
   PANELES
   ========================= */
#results-panel,
#downloads-panel,
#input-panel,
#output-panel {{
    background: #131313 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 14px !important;
    padding: 14px !important;
}}

/* =========================
   TÍTULOS DE SECCIÓN
   ========================= */
.section-title {{
    color: #f5f5f5 !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
    font-size: 1.2rem !important;
    margin: 0 0 0.7rem 0 !important;
    line-height: 1.2 !important;
}}

/* =========================
   ETIQUETAS VERDES
   ========================= */
.component-label {{
    color: #07d839 !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
    font-size: 1rem !important;
    margin: 0 0 0.45rem 0 !important;
    line-height: 1.2 !important;
}}

#results-body,
#results-body * {{
    color: #07d839 !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
}}


/* Labels reales de los gr.File en verde */
#csv-output label,
#csv-output .block-title,
#csv-output [data-testid="block-label"],
#csv-output .wrap > label,

#png-output label,
#png-output .block-title,
#png-output [data-testid="block-label"],
#png-output .wrap > label {{
    color: #07d839 !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
}}


/* =========================
   TEXTO GENERAL
   ========================= */
label,
span,
p,
div,
li,
td,
th,
small,
.gr-markdown,
.gr-file,
.gr-image,
.gr-box,
.gr-form,
.gr-group,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose em {{
    color: #919191 !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
}}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4 {{
    color: #919191 !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
}}

/* =========================
   TABLAS
   ========================= */
#results-panel table {{
    width: 100%;
    border-collapse: collapse;
}}

#results-panel th,
#results-panel td {{
    border: 1px solid #2a2a2a !important;
    padding: 8px 10px;
    color: #919191 !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
}}

#results-panel code {{
    color: #919191 !important;
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
}}

/* =========================
   BOTONES
   ========================= */
#process-btn button,
#process-btn button *,
#clear-btn button,
#clear-btn button * {{
    background: #176722 !important;
    color: #f5f5f5 !important;
    border: 1px solid #07d839 !important;
    font-family: "ZendotsCustom", sans-serif !important;
    font-weight: 400 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}}

#process-btn button:hover,
#process-btn button:hover *,
#clear-btn button:hover,
#clear-btn button:hover * {{
    background: #1d7c2a !important;
    color: #f5f5f5 !important;
    border: 1px solid #07d839 !important;
    font-family: "ZendotsCustom", sans-serif !important;
}}

button.primary,
button.primary *,
button.secondary,
button.secondary * {{
    background: #176722 !important;
    color: #f5f5f5 !important;
    border: 1px solid #07d839 !important;
    font-family: "ZendotsCustom", sans-serif !important;
}}

/* =========================
   TEXTO FINAL
   ========================= */
#hello-footer,
#hello-footer * {{
    margin-top: 14px;
    text-align: center;
    font-size: 0.8rem;
    color: #919191 !important;
    font-family: "BitcoinCustom", sans-serif !important;
}}

/* =========================
   RESPONSIVE
   ========================= */
@media (max-width: 900px) {{
    #app-title .title-line {{
        font-size: 4.5rem !important;
    }}

    #app-description,
    #app-description * {{
        font-size: 1.25rem !important;
    }}
}}
"""

# =========================
# FUNCIÓN PARA FORMATEAR LAS MÉTRICAS
# Convierte los resultados del predictor en texto Markdown bonito
# =========================
def format_metrics_markdown(result: dict) -> str:
    metrics = result["metrics"]
    image_name = result["image_name"]
    device = result["device"]

    return f"""
## Resultados del análisis

**Imagen procesada:** `{image_name}`  
**Dispositivo usado:** `{device}`

| Métrica | Valor |
|---|---:|
| Cantidad total | {metrics["total"]} |
| Cantidad vivas | {metrics["alive"]} |
| Cantidad muertas | {metrics["dead"]} |
| Viabilidad (%) | {metrics["viability_pct"]} |
| Confianza media | {metrics["mean_confidence"]} |

### Parámetros de inferencia
- **Pesos:** `{Path(DEFAULT_WEIGHTS_PATH)}`
- **Confianza:** `{CONF_THRESHOLD}`
- **IoU:** `{IOU_THRESHOLD}`
- **Tamaño de inferencia:** `{INFER_IMGSZ}`
""".strip()


# =========================
# FUNCIÓN PRINCIPAL DE PROCESADO
# Esta es la que ejecuta el botón "Procesar imagen"
# =========================
def run_inference(image):
    # Comprobamos que el usuario haya subido una imagen
    if image is None:
        raise gr.Error("Debes subir una imagen antes de procesarla.")

    try:
        # Llamamos a la función del predictor que hace toda la inferencia
        result = predict_image(
            image_input=image,
            weights_path=DEFAULT_WEIGHTS_PATH,
            output_dir=DEFAULT_OUTPUT_DIR,
        )

        # Recuperamos rutas de salida
        annotated_image_path = result["annotated_image_path"]
        csv_path = result["csv_path"]

        # Generamos el texto con métricas
        metrics_md = format_metrics_markdown(result)

        # Devolvemos:
        # 1) Imagen anotada
        # 2) Texto de métricas
        # 3) CSV descargable
        # 4) PNG anotado descargable
        return annotated_image_path, metrics_md, csv_path, annotated_image_path

    except Exception as e:
        # Mostramos el error de forma visible en la interfaz
        raise gr.Error(f"Error durante la inferencia: {e}")


# =========================
# CONSTRUCCIÓN DE LA INTERFAZ
# Aquí defines la disposición visual de la app
# =========================
with gr.Blocks(
    title=APP_TITLE,
    css=CUSTOM_CSS,
    theme=gr.themes.Base(),
) as demo:
    # ---------- Cabecera ----------
    gr.HTML(
        """
        <div id="app-title">
            <span class="title-line">BioSight</span>
            <span class="subtitle-line">Cell Counter</span>
        </div>
        """
    )
    gr.HTML(f'<div id="app-description">{APP_DESCRIPTION}</div>')

    # ---------- Fila principal con las dos imágenes ----------
    with gr.Row():
        # Columna izquierda: imagen de entrada
        with gr.Column(scale=1, elem_id="input-panel"):
            gr.HTML('<div class="component-label">Imagen de entrada</div>')

            input_image = gr.Image(
                type="pil",  # predictor.py espera PIL o equivalente
                show_label=False,
                elem_id="input-image",
            )

            # Botón de procesar
            process_button = gr.Button(
                "Procesar imagen",
                variant="primary",
                elem_id="process-btn",
            )

            # Botón de limpiar
            clear_button = gr.ClearButton(
                components=[],
                value="Limpiar",
                elem_id="clear-btn",
            )

        # Columna derecha: imagen anotada de salida
        with gr.Column(scale=2, elem_id="output-panel"):
            gr.HTML('<div class="component-label">Imagen anotada</div>')

            output_image = gr.Image(
                type="filepath",  # devolvemos la ruta del PNG guardado
                show_label=False,
                elem_id="output-image",
            )

    # ---------- Panel de resultados ----------
    with gr.Column(elem_id="results-panel"):
        gr.HTML('<div class="section-title">Resultados</div>')
        metrics_output = gr.Markdown(
            "Sube una imagen y pulsa **Procesar imagen**.",
            elem_id="results-body",
        )

    # ---------- Panel de descargas ----------
    with gr.Column(elem_id="downloads-panel"):
        gr.HTML('<div class="section-title">Descargas</div>')
        with gr.Row():
            with gr.Column():
                csv_output = gr.File(
                    label="Descargar CSV",
                    show_label=True,
                    elem_id="csv-output",
                )
            with gr.Column():
                png_output = gr.File(
                    label="Descargar PNG anotado",
                    show_label=True,
                    elem_id="png-output",
                )

    # ---------- Texto pequeño al final ----------
    gr.HTML('<div id="hello-footer">The only limit to my freedom is the inevitable closure of the universe, as inevitable as your own last breath. And yet, there remains time to create, to create, and escape.</div>')

    # Ahora que todos los componentes existen, conectamos el botón Limpiar
    clear_button.add(
        components=[input_image, output_image, metrics_output, csv_output, png_output]
    )

    # Conectamos el botón Procesar con la función de inferencia
    process_button.click(
        fn=run_inference,
        inputs=input_image,
        outputs=[output_image, metrics_output, csv_output, png_output],
    )


# =========================
# ARRANQUE DE LA APP
# =========================
if __name__ == "__main__":
    # Creamos la carpeta de outputs si no existe
    Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # queue() ayuda a que las acciones de botones se gestionen mejor
    demo.queue()

    # allowed_paths permite servir las fuentes locales al navegador
    demo.launch(
        inbrowser=True,
        show_error=True,
        allowed_paths=[str(ASSETS_DIR), str(FONTS_DIR)],
    )