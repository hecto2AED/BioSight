from __future__ import annotations

#########
from __future__ import annotations

# =========================
# BOOTSTRAP DE DEPENDENCIAS
# Comprueba si faltan librerías y, solo en ese caso,
# instala requirements.txt antes de importar Gradio/predictor
# =========================
import importlib.util
import subprocess
import sys
from pathlib import Path

# # Ruta al archivo requirements.txt
REQUIREMENTS_FILE = Path(__file__).resolve().parent / "requirements.txt"

# # Paquetes que queremos comprobar antes de arrancar
# # Ojo: aquí usamos el nombre del módulo de importación, no siempre el nombre de pip
REQUIRED_MODULES = [
    "gradio",
    "ultralytics",
    "torch",
    "pandas",
    "numpy",
    "PIL",   # Pillow se importa como PIL
    "cv2",   # opencv-python se importa como cv2
]

def missing_modules(modules: list[str]) -> list[str]:
    # Devuelve los módulos que no están instalados
    missing = []
    for module_name in modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing

def ensure_requirements_installed():
    # Comprueba si faltan módulos
    missing = missing_modules(REQUIRED_MODULES)

    # Si no falta nada, no hace nada
    if not missing:
        return

    # Si falta algo, intenta instalar requirements.txt
    print(f"Faltan dependencias: {missing}")
    print("Instalando requirements.txt...")

    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró requirements.txt en: {REQUIREMENTS_FILE}"
        )

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    )

# # Ejecutamos la comprobación/instalación antes de importar librerías externas
ensure_requirements_installed()

##################

# =========================
# IMPORTS
# =========================
from pathlib import Path

import gradio as gr

# Import the model logic from predictor.py
from predictor import (
    CONF_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WEIGHTS_PATH,
    INFER_IMGSZ,
    IOU_THRESHOLD,
    predict_image,
)

# =========================
# APP TEXTS
# =========================
APP_TITLE = "BioSight Cell Counter - Héctor Torres Muñoz"
APP_DESCRIPTION = (
    "Upload hemocytometer images "
    "and receive an annotated version, cell count, viability percentage, and mean sample confidence."
)

# =========================
# APP PATHS AND FONTS
# =========================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

AKIRA_FONT = (FONTS_DIR / "akira.otf").resolve().as_posix()
ZENDOTS_FONT = (FONTS_DIR / "zendots.otf").resolve().as_posix()
JETBRAINS_FONT = (FONTS_DIR / "jetbrainsmono.otf").resolve().as_posix()
BITCOIN_FONT = (FONTS_DIR / "bitcoin.otf").resolve().as_posix()
HEADER_IMAGE = (ASSETS_DIR / "image.png").resolve().as_posix()

# =========================
# CUSTOM CSS
# =========================
CUSTOM_CSS = f"""
/* =========================
   CUSTOM FONTS
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
   HEADER WITH IMAGE ON THE RIGHT
   ========================= */
#header-row {{
    align-items: flex-start !important;
    margin-bottom: 1.25rem !important;
}}

#header-left {{
    justify-content: flex-start !important;
}}

#header-right {{
    display: flex !important;
    justify-content: flex-end !important;
    align-items: flex-start !important;
}}

#header-image-wrap {{
    width: 100%;
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
}}

#header-image-wrap img {{
    max-width: 220px;
    width: 100%;
    height: auto;
    display: block;
    object-fit: contain;
}}

/* Optional for small screens */
@media (max-width: 900px) {{
    #header-image-wrap {{
        justify-content: flex-start;
        margin-top: 1rem;
    }}

    #header-image-wrap img {{
        max-width: 160px;
    }}
}}


/* =========================
   GENERAL BACKGROUND
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
   MAIN TITLE
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
   SUBTITLE / DESCRIPTION
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
   PANELS
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
   SECTION TITLES
   ========================= */
.section-title {{
    color: #f5f5f5 !important;
    font-family: "JetBrainsMonoCustom", monospace !important;
    font-size: 1.2rem !important;
    margin: 0 0 0.7rem 0 !important;
    line-height: 1.2 !important;
}}

/* =========================
   GREEN LABELS
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


/* Real gr.File labels in green */
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
   GENERAL TEXT
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
   TABLES
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
   BUTTONS
   ========================= */
#process-btn button,
#process-btn button *,
#clear-btn button,
#clear-btn button * {{
    background: #176722 !important;
    color: #f5f5f5 !important;
    border: 1px solid #07d839 !important;
    font-family: "JetBrainsMonoCustom", sans-serif !important;
    font-weight: 100 !important;
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
    font-family: "JetBrainsMonoCustom", sans-serif !important;
}}

/* =========================
   FINAL TEXT
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
# FUNCTION TO FORMAT METRICS
# Converts predictor results into clean Markdown text
# =========================
def format_metrics_markdown(result: dict) -> str:
    metrics = result["metrics"]
    image_name = result["image_name"]
    device = result["device"]

    return f"""
## Analysis Results

**Processed image:** `{image_name}`  
**Device used:** `{device}`

| Metric | Value |
|---|---:|
| Total count | {metrics["total"]} |
| Live count | {metrics["alive"]} |
| Dead count | {metrics["dead"]} |
| Viability (%) | {metrics["viability_pct"]} |
| Mean confidence | {metrics["mean_confidence"]} |

### Inference Parameters
- **Weights:** `{Path(DEFAULT_WEIGHTS_PATH)}`
- **Confidence:** `{CONF_THRESHOLD}`
- **IoU:** `{IOU_THRESHOLD}`
- **Inference size:** `{INFER_IMGSZ}`
""".strip()


# =========================
# MAIN PROCESSING FUNCTION
# This is triggered by the "Process Image" button
# =========================
def run_inference(image):
    # Check that the user has uploaded an image
    if image is None:
        raise gr.Error("You must upload an image before processing it.")

    try:
        # Call the predictor function that performs the full inference
        result = predict_image(
            image_input=image,
            weights_path=DEFAULT_WEIGHTS_PATH,
            output_dir=DEFAULT_OUTPUT_DIR,
        )

        # Retrieve output paths
        annotated_image_path = result["annotated_image_path"]
        csv_path = result["csv_path"]

        # Generate the metrics text
        metrics_md = format_metrics_markdown(result)

        # Return:
        # 1) Annotated image
        # 2) Metrics text
        # 3) Downloadable CSV
        # 4) Downloadable annotated PNG
        return annotated_image_path, metrics_md, csv_path, annotated_image_path

    except Exception as e:
        # Show the error visibly in the interface
        raise gr.Error(f"Error during inference: {e}")


# =========================
# INTERFACE CONSTRUCTION
# This is where the app layout is defined
# =========================
with gr.Blocks(
    title=APP_TITLE,
    css=CUSTOM_CSS,
    theme=gr.themes.Base(),
) as demo:
    # ---------- Header ----------
    with gr.Row(elem_id="header-row"):
        with gr.Column(scale=20, elem_id="header-left"):
            gr.HTML(
                """
                <div id="app-title">
                    <span class="title-line">BioSight</span>
                    <span class="subtitle-line">Cell Counter - Héctor Torres Muñoz - hector.tomu@gmail.com</span>
                </div>
                """
            )
            gr.HTML(f'<div id="app-description">{APP_DESCRIPTION}</div>')

        with gr.Column(scale=1, min_width=180, elem_id="header-right"):
            gr.HTML(
                f'''
                <div id="header-image-wrap">
                    <img src="/gradio_api/file={HEADER_IMAGE}" alt="BioSight image">
                </div>
                '''
            )

    # ---------- Main row with the two images ----------
    with gr.Row():
        # Left column: input image
        with gr.Column(scale=1, elem_id="input-panel"):
            gr.HTML('<div class="component-label">Input Image</div>')

            input_image = gr.Image(
                type="pil",  # predictor.py expects PIL or equivalent
                show_label=False,
                elem_id="input-image",
            )

            # Process button
            process_button = gr.Button(
                "Process Image",
                variant="primary",
                elem_id="process-btn",
            )

            # Clear button
            clear_button = gr.ClearButton(
                components=[],
                value="Clear",
                elem_id="clear-btn",
            )

        # Right column: annotated output image
        with gr.Column(scale=2, elem_id="output-panel"):
            gr.HTML('<div class="component-label">Annotated Image</div>')

            output_image = gr.Image(
                type="filepath",  # we return the saved PNG path
                show_label=False,
                elem_id="output-image",
            )

    # ---------- Results panel ----------
    with gr.Column(elem_id="results-panel"):
        gr.HTML('<div class="section-title">Results</div>')
        metrics_output = gr.Markdown(
            "Upload an image and click **Process Image**.",
            elem_id="results-body",
        )

    # ---------- Downloads panel ----------
    with gr.Column(elem_id="downloads-panel"):
        gr.HTML('<div class="section-title">Downloads</div>')
        with gr.Row():
            with gr.Column():
                csv_output = gr.File(
                    label="Download CSV",
                    show_label=True,
                    elem_id="csv-output",
                )
            with gr.Column():
                png_output = gr.File(
                    label="Download Annotated PNG",
                    show_label=True,
                    elem_id="png-output",
                )

    # ---------- Small text at the bottom ----------
    gr.HTML('<div id="hello-footer">The only limit to my freedom is the inevitable closure of the universe, as inevitable as your own last breath. And yet, there remains time to create, to create, and escape.\n hector.tomu@gmail.com</div>')

    # Now that all components exist, connect the Clear button
    clear_button.add(
        components=[input_image, output_image, metrics_output, csv_output, png_output]
    )

    # Connect the Process button to the inference function
    process_button.click(
        fn=run_inference,
        inputs=input_image,
        outputs=[output_image, metrics_output, csv_output, png_output],
    )


# =========================
# APP LAUNCH
# =========================
if __name__ == "__main__":
    # Create the output folder if it does not exist
    Path(DEFAULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # queue() helps button actions be handled more reliably
    demo.queue()

    # allowed_paths allows local fonts and assets to be served
    demo.launch(
        inbrowser=True,
        show_error=True,
        allowed_paths=[str(ASSETS_DIR), str(FONTS_DIR)],
    )