# CompositeGuard — Composite Damage Classification

CompositeGuard is a Streamlit web application for exploring two **independent** machine-learning classifiers for composite-material damage. The image and acoustic-emission (AE) signal models run separately; this project does not perform model fusion, segmentation, localization, bounding-box detection, or severity regression.

## Features

- **Image classification:** predicts `healthy` or `damaged` from an uploaded image.
- **AE signal classification:** predicts one of `healthy`, `matrix_crack`, `delamination`, or `fiber_break` from an uploaded signal CSV.
- **Signal inspection:** displays waveform and spectrogram visualizations where supported.
- **Prediction reporting:** presents predicted class and model scores, with a downloadable result where implemented.
- **Separate model pipelines:** image and signal inputs are handled independently.

## Project models

### Image model

The image branch uses ResNet-18 for binary classification. Images are resized to 224 × 224, converted to RGB, and normalized using ImageNet statistics. The project archive includes `image_model.pth`; its architecture and class mapping must match the inference code.

### Signal model

The signal branch converts each amplitude record into a log-power spectrogram and uses ResNet-18 to classify four classes. The current preprocessing assumes a sampling rate of **2048 Hz**. Verify this against acquisition metadata before interpreting frequency values physically.

The updated training script uses a stratified 70/15/15 train/validation/locked-test split, training-only augmentation, per-record amplitude normalization, partial backbone freezing, dropout, label smoothing, weight decay, gradient clipping, and early stopping. The best checkpoint is selected using validation macro-F1. The locked test set is evaluated after model selection.

## Dataset and limitations

The supplied data contains 120 synthetic images (60 healthy and 60 damaged) and 300 synthetic signal records (75 per signal class). The image and signal datasets are not documented as paired and must be evaluated independently.

The signal dataset does not document specimen/session IDs, so its current split is at the sample level. Related samples or shared synthetic-generation patterns could still cross splits. The reported held-out score is exploratory and **does not establish generalization to real composite structures**. Validate on independently collected specimens before making operational or engineering claims.

## Requirements

Use Python 3.10 or newer in a virtual environment. Install the project dependencies from the project root:

```bash
pip install -r requirements-webapp.txt
```

If you need a CUDA-enabled PyTorch installation, choose the command appropriate for your system from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Run the web app locally

1. Extract the project archive and open a terminal in the `damage_detection_updated` directory.
2. Install dependencies as described above.
3. Confirm `image_model.pth` is present.
4. Make sure the signal checkpoint exists. If it has not been trained yet, run:

   ```bash
   python starter_train_signal.py
   ```

   This creates `signal_model.pth`. If training was performed on another machine, copy the exact checkpoint produced by that script into this directory.
5. Start the application:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed by Streamlit in your browser.

The app requires the checkpoint architecture, preprocessing, and class ordering to match the training code. It should report checkpoint-loading errors rather than silently ignoring missing model weights.

## Input formats

### Image classifier

Upload an image file in a format supported by the app (for example PNG or JPG).

### Signal classifier

Upload a CSV containing a numeric amplitude column. The default expected column is `amplitude`; the app allows selection of another numeric column if available. Ensure the file represents one signal record and that its format is consistent with the training data.

## Training and evaluation

From the project directory:

```bash
python starter_train_image.py
python starter_train_signal.py
```

The signal training script reports validation metrics and evaluates the locked test set after training. The image script uses its own image-classification pipeline and data split. Review the current scripts for their exact split and metric details before comparing the models; their scores are not directly comparable unless the evaluation protocols are aligned.

Recommended next steps:

1. Audit exact and near-duplicate samples and potential synthetic-generation shortcuts.
2. Add specimen/session-level splitting when source identifiers are available.
3. Compare the spectrogram ResNet with a Random Forest on engineered features and a 1D-CNN on raw waveforms using identical partitions.
4. Investigate matrix-crack versus delamination errors.
5. Test the frozen model on real signals from specimens absent from training.
6. Confirm the true AE sampling rate and preprocessing requirements from acquisition metadata.

## Disclaimer

This application is a research prototype, not a certified inspection system. Predictions are model outputs and are not a substitute for qualified engineering assessment or validated non-destructive testing procedures.
