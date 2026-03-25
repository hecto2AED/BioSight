# **BioSight Cell Vision**

BioSight Cell Vision is a computer vision pipeline for automated cell detection and counting in hemocytometer microscopy images. The project uses a fine-tuned YOLO11s object detection model to identify and classify live and dead cells, generate annotated images, and produce tabular reports with total counts and viability estimates.

## **Project Overview**

The main goal of this project is to replace manual cell counting with a reproducible deep learning workflow. Given a microscopy image of a hemocytometer chamber, the model detects individual cells, assigns a class to each detection, and calculates summary statistics such as:

- **Total cell count**
- **Live cell count**
- **Dead cell count**
- **Viability percentage**

The pipeline is designed to work end-to-end, from training on labeled datasets to inference on unseen images.

## **Model**

The project is based on **YOLO11s**, a lightweight object detection architecture from Ultralytics. This model was selected as a balance between accuracy and computational efficiency, especially for small-object detection in dense microscopy scenes.

The detector was fine-tuned on labeled hemocytometer datasets where each cell is annotated with a bounding box and a class:

- **A**: alive
- **D**: dead

## **Data**

The training data consists of annotated hemocytometer microscopy images from multiple datasets combined into a unified YOLO-style structure. The final dataset is organized as follows:

```text
train/
  images/
  labels/

val/
  images/
  labels/

test/
  images/
  labels/
```

Each label file follows the standard YOLO detection format with normalized coordinates:

```text
class x_center y_center width height
```

## **Training Workflow**

The model is trained from pretrained YOLO11s weights using transfer learning. The training setup includes:

- **High input resolution** to improve small-cell detection
- **Early stopping** with patience control
- **Reproducible seed settings**
- **MPS acceleration** on Apple Silicon when available

The final training pipeline produces:

- **Best model weights** (`best.pt`)
- **Last model weights** (`last.pt`)
- **Training plots and logs**

## **Evaluation**

The project evaluates the detector on both validation and test sets using standard object detection metrics:

- **Precision**
- **Recall**
- **mAP50**
- **mAP50-95**
- **Confusion matrix**

Since the practical objective of the project is not only detection but also reliable cell quantification, the repository also computes task-specific metrics:

- **MAE total count**
- **MAE live count**
- **MAE dead count**
- **MAE viability percentage**
- **Count bias**

These metrics make it possible to assess how well the model performs for real counting and viability estimation tasks, not only for box localization.

## **Inference Pipeline**

The final inference pipeline performs the following steps:

1. **Load the trained model**
2. **Convert input images to RGB when needed**
3. **Run object detection**
4. **Generate annotated output images**
5. **Compute cell counts and viability**
6. **Export results as a CSV table**

This makes the project suitable both for experimentation and for practical batch processing of microscopy images.

## **Outputs**

For each processed image, the pipeline can generate:

- An **annotated image** with predicted bounding boxes and classes
- A row in a **results table** including:
  - **Image name**
  - **Total count**
  - **Live count**
  - **Dead count**
  - **Viability percentage**
  - **Mean confidence score**

## **App Interface**

In addition to the notebook-based pipeline, the project also includes a lightweight local app built with **Gradio** for direct image analysis.

The app provides a simple workflow for users who want to run inference without interacting with the notebook or editing code manually.

## **How the App Works**

The application follows a straightforward three-step process:

### **1. Upload Image**
The user uploads a hemocytometer microscopy image through the graphical interface.

### **2. Processing**
Once the image is submitted, the app:

- Loads the trained YOLO11s model (`best.pt`)
- Converts the input image to RGB when needed
- Runs object detection on the uploaded image
- Identifies live and dead cells
- Computes summary statistics including total count and viability

### **3. Output Generation**
After processing, the app returns:

- An **annotated image** with predicted bounding boxes and class labels
- A **results summary** showing:
  - **Total cell count**
  - **Live cell count**
  - **Dead cell count**
  - **Viability percentage**
  - **Mean confidence score**
- Downloadable output files:
  - **CSV report**
  - **Annotated PNG image**

## **App Structure**

The app is organized as a minimal deployable local project:

```text
cell-counter-app/
├── app.py
├── predictor.py
├── requirements.txt
├── weights/
│   └── best.pt
└── outputs/
```

Where:

- **`app.py`** contains the graphical user interface
- **`predictor.py`** contains the inference logic
- **`requirements.txt`** lists the Python dependencies
- **`weights/best.pt`** stores the trained model weights
- **`outputs/`** stores generated annotated images and CSV reports

## **Main Use Case**

This repository is intended for:

- **Automated live/dead cell counting** in hemocytometer images
- **Viability estimation** from microscopy data
- **Benchmarking object detection models** on dense biological images
- **Building reproducible workflows** for cell analysis

## **Limitations**

Although the model performs well on the original dataset, performance may decrease on external datasets due to differences in:

- **Image quality**
- **Staining conditions**
- **Microscope settings**
- **Domain shift between datasets**

Future improvements may include domain adaptation, more balanced datasets, and additional experiments with confidence threshold optimization and counting-oriented evaluation.

## **Repository Purpose**

This repository provides a complete and reproducible framework for:

- **Training a YOLO-based detector**
- **Evaluating detection and counting performance**
- **Running inference for cell counting and viability analysis**





