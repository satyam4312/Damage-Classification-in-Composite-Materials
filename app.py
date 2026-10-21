from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from scipy.signal import spectrogram
from torch import nn
from torchvision import models, transforms

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
IMAGE_MODEL_PATH = ROOT / 'image_model.pth'
SIGNAL_MODEL_PATH = ROOT / 'signal_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_CLASSES = ['healthy', 'damaged']
SIGNAL_CLASSES = ['healthy', 'matrix_crack', 'delamination', 'fiber_break']
SAMPLE_RATE = 2048          # Hz, must match the training script
LOW_CONFIDENCE = 0.65
MIN_SAMPLES = 16            # enough to plot and compute statistics
MIN_CLASSIFY_SAMPLES = 512  # spectrogram uses 256-sample windows; need several frames to be meaningful

st.set_page_config(page_title='CompositeGuard | Damage Intelligence', page_icon='🧪', layout='wide')

st.markdown('''<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px;}
.hero {padding: 1.8rem 2rem; border-radius: 18px; margin-bottom: 1.2rem; color: #fff;
       background: linear-gradient(120deg,#0b2239 0%,#12507a 55%,#1b8a9a 100%);
       box-shadow: 0 8px 24px rgba(16,42,67,.18);}
.hero h1 {font-size: 2.2rem; margin: 0 0 .4rem 0; color: #fff; letter-spacing: -.5px;}
.hero p {margin: 0; opacity: .92; font-size: 1.04rem;}
.result-card {padding: 1.1rem 1.3rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);
              border-left-width: 6px; margin-bottom: .8rem;}
.result-card .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65;}
.result-card .value {font-size: 1.7rem; font-weight: 700; margin: .1rem 0;}
.result-card .conf {font-size: .95rem; opacity: .8;}
.result-card.good {border-left-color: #16a34a;}
.result-card.bad {border-left-color: #dc2626;}
.result-card.warn {border-left-color: #d97706;}
.small-note {color: #64748b; font-size: .88rem;}
</style>''', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def build_model(task: str) -> tuple[nn.Module, list[str]]:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    if task == 'image':
        model.fc = nn.Linear(in_features, len(IMAGE_CLASSES))
        return model, IMAGE_CLASSES
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, len(SIGNAL_CLASSES)))
    return model, SIGNAL_CLASSES


@st.cache_resource(show_spinner='Loading model…')
def load_model(path_str: str, task: str, mtime: float):
    """Load a checkpoint. `mtime` is part of the cache key so replacing the file reloads it."""
    path = Path(path_str)
    if not path.exists():
        return None, f'{path.name} not found in the project folder.'
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        is_dict = isinstance(checkpoint, dict)
        state = checkpoint.get('state_dict', checkpoint) if is_dict else checkpoint
        saved_classes = checkpoint.get('classes') if is_dict else None

        model, expected = build_model(task)
        state = {k.removeprefix('module.'): v for k, v in state.items()}  # DataParallel checkpoints
        model.load_state_dict(state, strict=True)

        if saved_classes and list(saved_classes) != expected:
            return None, f'Class order in {path.name} {list(saved_classes)} does not match {expected}.'
        return model.to(DEVICE).eval(), None
    except Exception as e:
        return None, f'Could not load {path.name}: {type(e).__name__}: {e}'


def get_model(path: Path, task: str):
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return load_model(str(path), task, mtime)


@st.cache_resource(show_spinner=False)
def image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229]),
    ])


@st.cache_resource(show_spinner=False)
def signal_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
    ])


def predict(model, tensor, classes):
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).to(DEVICE))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return classes[int(np.argmax(probs))], probs


# ----------------------------------------------------------------------------
# Signal processing
# ----------------------------------------------------------------------------
def make_spectrogram(values):
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) < MIN_CLASSIFY_SAMPLES:
        raise ValueError(f'The signal needs at least {MIN_CLASSIFY_SAMPLES} samples to classify (got {len(arr)}).')
    arr = arr - np.median(arr)
    scale = np.percentile(np.abs(arr), 95) + 1e-8
    arr = np.clip(arr / scale, -5, 5)
    nperseg = min(256, len(arr))
    noverlap = min(192, max(0, nperseg - 1))
    _, _, power = spectrogram(arr, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
    spec = np.log1p(power).astype(np.float32)
    lo, hi = np.percentile(spec, [1, 99])
    spec = np.clip((spec - lo) / (hi - lo + 1e-8), 0, 1)
    image = Image.fromarray(np.uint8(spec * 255)).convert('RGB')
    # Fixed 224x224 size: keeps the preview square and matches the model input.
    return image.resize((224, 224), Image.BILINEAR)


def signal_stats(values: np.ndarray) -> dict:
    centered = values - np.mean(values)
    return {
        'Samples': f'{len(values):,}',
        'Duration': f'{len(values) / SAMPLE_RATE:.3f} s',
        'Peak amplitude': f'{np.max(np.abs(values)):.4g}',
        'RMS': f'{np.sqrt(np.mean(centered ** 2)):.4g}',
    }


def sample_csv() -> str:
    t = np.arange(2048) / SAMPLE_RATE
    rng = np.random.default_rng(0)
    amp = 0.05 * rng.standard_normal(t.size)
    burst = (t > 0.4) & (t < 0.5)
    amp[burst] += np.sin(2 * np.pi * 300 * t[burst]) * np.exp(-(t[burst] - 0.4) * 40)
    return pd.DataFrame({'amplitude': amp}).to_csv(index=False)


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def pretty(name: str) -> str:
    return name.replace('_', ' ').title()


def result_card(label: str, probs: np.ndarray, healthy_label: str = 'healthy'):
    conf = float(probs.max())
    if conf < LOW_CONFIDENCE:
        css = 'warn'
    else:
        css = 'good' if label == healthy_label else 'bad'
    st.markdown(
        f'<div class="result-card {css}"><div class="label">Predicted class</div>'
        f'<div class="value">{pretty(label)}</div>'
        f'<div class="conf">Confidence {conf:.1%}</div></div>',
        unsafe_allow_html=True,
    )
    if conf < LOW_CONFIDENCE:
        st.warning('Low-confidence prediction. Review manually and request expert verification.')


def probability_chart(classes, probs):
    df = pd.DataFrame({'Class': [pretty(c) for c in classes], 'Probability': probs})
    st.bar_chart(df.set_index('Class'), y='Probability', height=220)
    with st.expander('Probability table'):
        table = df.sort_values('Probability', ascending=False).copy()
        table['Probability'] = table['Probability'].map(lambda x: f'{x:.2%}')
        st.dataframe(table, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
image_model, image_error = get_model(IMAGE_MODEL_PATH, 'image')
signal_model, signal_error = get_model(SIGNAL_MODEL_PATH, 'signal')

st.markdown(
    '<div class="hero"><h1>🧪 CompositeGuard</h1>'
    '<p>AI-assisted screening for composite material damage, from specimen images and acoustic-emission signals.</p>'
    '</div>',
    unsafe_allow_html=True,
)

st.subheader('About')
about_left, about_right = st.columns(2, gap='large')
with about_left:
    st.markdown('**🖼️ Image screening**  \nA ResNet-18 classifier labels a specimen image as *healthy* or *damaged*.')
with about_right:
    st.markdown('**〰️ AE signal analysis**  \nThe signal is normalized, converted to a log-power spectrogram, and classified by '
                'ResNet-18 as *healthy*, *matrix crack*, *delamination* or *fiber break*.')

tab_image, tab_signal = st.tabs(['🖼️ Image screening', '〰️ AE signal analysis'])

# ---- Image tab --------------------------------------------------------------
with tab_image:
    left, right = st.columns(2, gap='large')
    with left:
        st.subheader('Specimen image')
        uploaded_image = st.file_uploader('PNG, JPG or JPEG', type=['png', 'jpg', 'jpeg'], key='image_upload')
        pil_image = None
        if uploaded_image:
            try:
                pil_image = Image.open(uploaded_image).convert('RGB')
                st.image(pil_image, caption=f'{uploaded_image.name} · {pil_image.width}×{pil_image.height}px',
                         use_container_width=True)
            except Exception as e:
                st.error(f'Could not read this image: {e}')
    with right:
        st.subheader('Prediction')
        if image_error:
            st.warning(image_error)
            st.info('Train and export the image model, then refresh this page.')
        elif pil_image is not None:
            label, probs = predict(image_model, image_transform()(pil_image), IMAGE_CLASSES)
            result_card(label, probs)
            probability_chart(IMAGE_CLASSES, probs)
        else:
            st.info('Upload an image to run the classifier.')

# ---- Signal tab -------------------------------------------------------------
with tab_signal:
    head, dl = st.columns([3, 1])
    head.subheader('Acoustic-emission signal')
    dl.download_button('⬇ Sample CSV', data=sample_csv(), file_name='sample_signal.csv',
                       mime='text/csv', use_container_width=True)
    st.caption(f'Upload a CSV with a numeric `amplitude` column (other numeric columns can be selected). '
               f'Sampling rate is assumed to be {SAMPLE_RATE} Hz, matching the training script.')

    uploaded_csv = st.file_uploader('Signal CSV', type=['csv'], key='signal_upload')
    if not uploaded_csv:
        st.info('Upload a CSV to begin.')
    else:
        try:
            df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f'Could not read CSV: {e}')
            df = None

        numeric_cols = list(df.select_dtypes(include=[np.number]).columns) if df is not None else []
        if df is not None and df.empty:
            st.error('The CSV is empty.')
        elif df is not None and not numeric_cols:
            st.error('No numeric columns found in the CSV.')
        elif df is not None:
            default = numeric_cols.index('amplitude') if 'amplitude' in numeric_cols else 0
            col = st.selectbox('Amplitude column', numeric_cols, index=default)
            raw = pd.to_numeric(df[col], errors='coerce').to_numpy()
            values = raw[np.isfinite(raw)]

            if len(values) < MIN_SAMPLES:
                st.error(f'Choose a column with at least {MIN_SAMPLES} valid samples.')
            else:
                for (name, val), c in zip(signal_stats(values).items(), st.columns(4)):
                    c.metric(name, val)

                fig, ax = plt.subplots(figsize=(10, 2.6))
                ax.plot(np.arange(len(values)) / SAMPLE_RATE, values, linewidth=0.8, color='#12507a')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(alpha=0.2)
                ax.margins(x=0)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                if 'amplitude' not in numeric_cols:
                    st.warning('No `amplitude` column found. Make sure the selected column is a raw signal '
                               'waveform and not labels or metadata.')

                if len(values) < MIN_CLASSIFY_SAMPLES:
                    st.error(f'This column has only {len(values)} samples. Classification needs at least '
                             f'{MIN_CLASSIFY_SAMPLES} (the training signals were sampled at {SAMPLE_RATE} Hz). '
                             'Upload a raw AE waveform instead.')
                elif signal_error:
                    st.warning(signal_error)
                    st.info('Run `python starter_train_signal.py` to create signal_model.pth, then refresh.')
                elif st.button('Classify signal', type='primary', use_container_width=True):
                    try:
                        with st.spinner('Analysing signal…'):
                            spec_image = make_spectrogram(values)
                            label, probs = predict(signal_model, signal_transform()(spec_image), SIGNAL_CLASSES)
                        a, b = st.columns(2, gap='large')
                        with a:
                            st.image(spec_image, caption='Model input: normalized log-power spectrogram',
                                     use_container_width=True)
                        with b:
                            result_card(label, probs)
                            probability_chart(SIGNAL_CLASSES, probs)
                        summary = pd.DataFrame({'class': SIGNAL_CLASSES, 'probability': probs})
                        st.download_button('Download prediction summary', data=summary.to_csv(index=False),
                                           file_name='signal_prediction.csv', mime='text/csv')
                    except Exception as e:
                        st.error(f'Could not process this signal: {e}')

st.divider()
st.caption('CompositeGuard · Research prototype')