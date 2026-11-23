# Rice Leaf Disease Detector
An image classification project that identifies 10 types of rice leaf conditions from a single leaf photo. It uses transfer learning with MobileNet models in Python (PyTorch), then validates the results with a statistical analysis in R.

This project was developed for CSST 101 (R and data analysis) and CSST 102 (Python and machine learning).

## Table of Contents
- Project Overview
- Repository Structure
- Dataset
- Methodology
- Results
- Statistical Analysis in R
- Advice Layer and diagnose()
- How to Run
- Limitations
- Future Work
- Tech Stack
- Acknowledgments
- Authors
- License

## Project Overview
Rice diseases and pests reduce crop yield, and many of them look very similar to the human eye. This project builds a lightweight classifier that:

- Takes a photo of one rice leaf as input.
- Predicts one of 10 classes (nine diseases or pests, and one healthy class).
- Returns a confidence score and flags low-confidence predictions.
- Produces a short report with an urgency level and general management advice.

The workflow has two parts:
- **Machine learning pipeline (Python):** data cleaning, model training, evaluation, error analysis, and a demo.
- **Statistical analysis (R):** confidence intervals, hypothesis tests, calibration checks, and independent verification of the Python results.

The Python notebook exports its predictions and metrics as CSV and JSON files. The R notebook reads these files and recomputes every metric independently to verify the results.

## Repository Structure
- `Rice_Leaf_Disease_Python.ipynb`: machine learning pipeline (CSST 102)
- `Rice_Leaf_Disease_R.ipynb`: statistical analysis (CSST 101)
- `README.md`: project documentation

Both notebooks are designed to run on Google Colab.

## Dataset
The dataset is a collection of rice leaf images organized in class folders:

- `Orig/`: original photos, used for training and evaluation
- `Altered/`: augmented copies, not used for splitting, to prevent data leakage

The 10 classes are: Healthy, bacterial_leaf_blight, brown_spot, leaf_blast, leaf_scald, narrow_brown_spot, neck_blast, rice_hispa, sheath_blight, and tungro.

Dataset size after each step:
- Original images found: 8,359
- After removing 158 exact duplicates and 1 label conflict: 8,200
- Stratified split (70/15/15): 5,740 training, 1,230 validation, and 1,230 test images

**Source:** Add the dataset name, author, link, and license here. Follow the original dataset's license and citation requirements.

## Methodology

### Data Preparation
- Every image was checked for corrupt files, color mode, and dimensions.
- MD5 hashing was used to detect exact duplicates and images filed under conflicting labels.
- A shortcut check was performed: a Random Forest trained only on image metadata (width, height, aspect ratio, file size) reached 43.6% accuracy, compared with 10% by chance. This shows that photo properties leak some class information, so all images are resized to the same square size.
- A stratified split was used, and the notebook asserts that the training, validation, and test sets do not overlap by file path or hash.

### Preprocessing
- Resize to 256 x 256 pixels
- Normalize with ImageNet mean and standard deviation
- Basic data augmentation for training images

### Models
- **MobileNetV4 Conv Medium** (`mobilenetv4_conv_medium.e500_r256_in1k`): 8.45 million parameters
- **MobileNetV3 Large** (`mobilenetv3_large_100.ra_in1k`): 4.21 million parameters

Both models use ImageNet pretrained weights loaded through the `timm` library.

### Training Setup
- **Baseline:** frozen backbone, training only the classification head for 4 epochs.
- **Fine-tuning:** unfrozen network with separate learning rates for the head and the backbone, up to 24 epochs with early stopping.
- Mixed precision (AMP) on GPU and a fixed random seed (`SEED = 42`) for reproducibility.

### Evaluation
The evaluation uses accuracy, balanced accuracy, macro precision, macro recall, macro-F1, top-3 accuracy, log loss, confusion matrices, inference latency, expected calibration error (ECE), and a robustness test.

Macro-F1 is the main metric because the classes are imbalanced (the largest class is 1.76 times the size of the smallest), and macro-F1 gives each class equal weight.

## Results

### Model Comparison (Test Set, 1,230 Images)
- **MobileNetV4, frozen backbone (`v4_baseline_frozen`):** 74.4% accuracy, 0.749 macro-F1, 93.0% top-3 accuracy, 57.4 ms latency per image.
- **MobileNetV3-Large, frozen backbone (`v3_frozen`):** 80.9% accuracy, 0.812 macro-F1, 96.1% top-3 accuracy, 30.7 ms latency per image.

The fine-tuned MobileNetV4 run reached a validation macro-F1 of about 0.976 before early stopping (see the training log in Week 5.1 of the Python notebook).

**Note:** The exported comparison tables and the R analysis in the current notebooks include only the two frozen runs. Re-run Week 5 to Week 7 in the Python notebook and re-export the files so that the fine-tuned model appears in all final tables and in the R report.

### Per-Class Behavior (Frozen MobileNetV4 Baseline)
- Best class: neck_blast (F1 = 0.98)
- Weakest class: leaf_blast (F1 = 0.52, recall = 42%)
- Most frequent error: leaf_blast predicted as brown_spot (23 images)

### Robustness Test (Macro-F1 Under Image Changes)
- Clean images: 0.749
- Darker (brightness x 0.6): 0.733
- Brighter (brightness x 1.5): 0.705
- Blurry: 0.386
- Low resolution (64 px, upscaled): 0.407

Blur and low resolution reduce performance the most. Users should take sharp, well-lit photos.

### Confidence and Calibration
- Expected calibration error (ECE): 0.102
- Confidence threshold selected on validation data: 0.60
- On the test set, 83% of images pass the threshold, with 81.9% accuracy among them. The remaining images are reported as low confidence and flagged for a second opinion.

## Statistical Analysis in R
The R notebook imports the Python outputs and answers six research questions:
- **Q1. How accurate is the model, and how certain are we?** Bootstrap 95% confidence intervals and Wilson intervals for per-class recall.
- **Q2. Is one model significantly better than another?** Exact McNemar test on paired outcomes, plus a paired bootstrap of the macro-F1 difference.
- **Q3. Are errors spread evenly across classes?** Chi-square test with a Monte Carlo p-value, and Cramer's V as the effect size.
- **Q4. Do image properties predict errors?** Logistic regression (odds ratios) and Mann-Whitney tests.
- **Q5. Can the confidence scores be trusted?** Reliability diagram, ECE, and Brier score.
- **Q6. Is there over-fitting?** Train-validation gaps from the training history.

P-values that belong to the same family of tests are adjusted with the Holm method (alpha = 0.05).

Key findings from the exported frozen-model runs:
- Accuracy is 74.4% (95% CI: 72.1% to 76.9%), and macro-F1 is 0.749 (95% CI: 0.724 to 0.773).
- MobileNetV3-Large performs significantly better than the frozen MobileNetV4 baseline (McNemar p < 0.001).
- Errors are not evenly distributed across classes (chi-square p < 0.001, Cramer's V = 0.35).
- Brightness, saturation, and color-ratio features are associated with errors after Holm correction.
- R independently reproduced all metrics reported by Python (accuracy, macro-F1, and top-3 accuracy).

## Advice Layer and diagnose()
On top of the classifier, the project includes a rule-based knowledge base. It is not learned from data. Each class has a disease type, an urgency level, and simple management advice based on public rice-extension guidance, such as the IRRI Rice Knowledge Bank.

Urgency levels:
- **None:** no action needed
- **Low:** monitor the field
- **Moderate:** act within about one week
- **High:** act within 1 to 3 days

Example usage:
```python
diagnose("path/to/leaf_photo.jpg")
```

The function prints the prediction, confidence, top-3 guesses, urgency level, and a list of recommended actions. If the confidence is below the threshold, it asks the user to retake the photo.

An optional Gradio web demo is available. Set `RUN_GRADIO = True` in the CONFIG cell to enable it.

**Disclaimer:** This tool provides general guidance only. Treatments and chemical regulations vary by country and crop stage. Always confirm with a local agricultural technician.

## How to Run

### Part 1: Python Notebook (Google Colab)
1. Open `Rice_Leaf_Disease_Python.ipynb` in Google Colab.
2. Set the runtime to GPU: Runtime > Change runtime type > T4 GPU.
3. Upload the dataset to Google Drive at `MyDrive/Rice_Leaf_Disease_Project/Rice Leaf Disease/`, with the `Orig/` and `Altered/` folders inside.
4. Edit only the CONFIG cell. The main settings are:
   - `DATA_MODE`: `'drive'`, `'upload_zip'`, `'kaggle'`, or a local folder
   - `QUICK_RUN`: set to `True` for a fast test with fewer images and 2 epochs
   - `EPOCHS_BASELINE` and `EPOCHS_FINETUNE`: training length
   - `RUN_GRADIO`: launch the web demo
5. Select Runtime > Restart session and run all.
6. In Week 7.8, the notebook creates `CSST102_outputs_for_R.zip`. Download it for Part 2.

The notebook installs its own dependencies: `timm`, `tqdm`, `scikit-learn`, and `seaborn`. PyTorch is already available on Colab.

### Part 2: R Notebook (Google Colab)
1. Open `Rice_Leaf_Disease_R.ipynb` and select the R runtime.
2. Upload `CSST102_outputs_for_R.zip` to the Colab Files panel.
3. Edit the CONFIG cell: set `ZIP_PATH` and fill in `DATASET_SOURCE`.
4. Select Runtime > Restart session and run all.
5. Results are saved in `/content/outputs_R/`, including tables, figures, and a draft report. Download them before the session ends.

The R packages `tidyverse` and `jsonlite` are installed automatically.

If you retrain the model in Python, re-export the zip file and re-run the R notebook so that all numbers stay consistent.

## Limitations
- **Dataset bias:** The photos come from a single dataset, often with plain backgrounds. Real field accuracy will likely be lower.
- **Look-alike diseases:** brown_spot, narrow_brown_spot, and leaf_blast are easily confused.
- **No "not a rice leaf" class:** Any photo receives one of the 10 labels. The confidence threshold gives only partial protection.
- **Near-duplicate images:** Only exact duplicates were removed, so near-duplicates may still exist across splits and make scores optimistic.
- **Single leaf versus field severity:** One photo cannot show how much of a field is affected.
- **Rice hispa is an insect pest, not a disease.** Its treatment differs from fungal or bacterial diseases.
- **Single split and seed:** Results may change slightly with different random settings.
- **Generic advice:** The advice layer is rule-based and was not evaluated statistically.

## Future Work
- Collect field photos from local farms to test real-world performance.
- Add a "not rice" class and an out-of-distribution check.
- Estimate disease severity in addition to the class.
- Try stronger augmentation, class weights, and label smoothing.
- Run multiple seeds and cross-validation for more reliable statistics.
- Deploy the model as a mobile or web application.

## Tech Stack
- **Python:** PyTorch, timm, scikit-learn, pandas, NumPy, Pillow, Matplotlib, Seaborn, Gradio (optional)
- **R:** tidyverse (dplyr, ggplot2, purrr, readr), jsonlite
- **Platform:** Google Colab and Google Drive

## Acknowledgments
- Md Mahamudul Hasan, Md Shakib Hossain, and Jahanur Biswas for the Mendeley Data rice leaf disease dataset (CC BY 4.0)
- [https://data.mendeley.com/datasets/n8jpkpfz5p/1]
- timm, by Ross Wightman, for the pretrained MobileNet models
- IRRI Rice Knowledge Bank for general rice management guidance
- Mr. Mark Bernardino & Mr. Rolan Sotomayor (CSST 101 and CSST 102 Instructors)

## Authors
- Ballinas, Mathew
- Cabanit, Rebecca
- Cueto, John Lei

## License
The datasets used in this project are not covered by this license. The Mendeley Data dataset is licensed under CC BY 4.0, which requires attribution to the original authors. The Kaggle dataset(s) have their own licenses. Please follow each source's terms when you reuse or redistribute the data.
