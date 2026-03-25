**BioSight Cell Vision**

BioSight Cell Vision is a computer vision pipeline for automated cell detection and counting in hemocytometer microscopy images. The project uses a fine-tuned YOLO11s object detection model to identify and classify live and dead cells, generate annotated images, and produce tabular reports with total counts and viability estimates.

**Project overview**

The main goal of this project is to replace manual cell counting with a reproducible deep learning workflow. Given a microscopy image of a hemocytometer chamber, the model detects individual cells, assigns a class to each detection, and calculates summary statistics such as:

+ total cell count

+ live cell count

+ dead cell count

+ viability percentage

The pipeline is designed to work end-to-end, from training on labeled datasets to inference on unseen images.

**Model**

The project is based on YOLO11s, a lightweight object detection architecture from Ultralytics. This model was selected as a balance between accuracy and computational efficiency, especially for small-object detection in dense microscopy scenes.

The detector was fine-tuned on labeled hemocytometer datasets where each cell is annotated with a bounding box and a class:

+ A: alive

+ D: dead

**Data**

The training data consists of annotated hemocytometer microscopy images from multiple datasets combined into a unified YOLO-style structure. The final dataset is organized into:


train/
  images/
  labels/

val/
  images/
  labels/

test/
  images/
  labels/

Each label file follows the standard YOLO detection format with normalized coordinates: `class x_center y_center width height`

**Training workflow**

The model is trained from pretrained YOLO11s weights using transfer learning. The training setup includes:

+ high input resolution to improve small-cell detection

+ early stopping with patience control

+ reproducible seed settings

+ MPS acceleration on Apple Silicon when available

The final training pipeline produces:

+ best model weights (best.pt)

+ last model weights (last.pt)

+ training plots and logs

**Evaluation**

The project evaluates the detector on both validation and test sets using standard object detection metrics:

+ Precision

+ Recall

+ mAP50

+ mAP50-95

+ Confusion matrix

Since the practical objective of the project is not only detection but also reliable cell quantification, the repository also computes task-specific metrics:

+ MAE total count

+ MAE live count

+ MAE dead count

+ MAE viability percentage

+ count bias

These metrics make it possible to assess how well the model performs for real counting and viability estimation tasks, not only for box localization.

**Inference pipeline**

The final inference pipeline performs the following steps:

1. load the trained model

2. convert input images to RGB when needed

3. run object detection

4. generate annotated output images

5. compute cell counts and viability

6. export results as a CSV table

This makes the project suitable both for experimentation and for practical batch processing of microscopy images.

**Outputs**

For each processed image, the pipeline can generate:

an annotated image with predicted bounding boxes and classes
and a row in a results table including:

+ image name

+ total count

+ live count

+ dead count

+ viability percentage

+ mean confidence score

**Main use case**

This repository is intended for:

+ automated live/dead cell counting in hemocytometer images

+ viability estimation from microscopy data

+ benchmarking object detection models on dense biological images

+ building reproducible workflows for cell analysis

**Limitations**

Although the model performs well on the original dataset, performance may decrease on external datasets due to differences in:

+ image quality

+ staining conditions

+ microscope settings

+ domain shift between datasets

Future improvements may include domain adaptation, more balanced datasets, and additional experiments with confidence threshold optimization and counting-oriented evaluation.

**Repository purpose**

This repository provides a complete and reproducible framework for:

+ training a YOLO-based detector
+ evaluating detection and counting performance
+ running inference for cell counting and viability analysis













