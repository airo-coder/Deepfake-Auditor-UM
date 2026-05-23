import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import joblib
import io
import soundfile as sf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# ─────────────────────────────────────────────────────────
# Google Gemini API for dynamic AI-powered reasoning
# ─────────────────────────────────────────────────────────
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ─────────────────────────────────────────────────────────
# Page Configuration & Custom CSS
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Auditor V4 Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2230;
        border: 1px solid #2e3440;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-title {
        color: #88c0d0;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #eceff4;
    }
    .metric-status {
        font-size: 0.8rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    .status-normal { color: #a3be8c; }
    .status-warning { color: #ebcb8b; }
    .status-critical { color: #bf616a; }
    .banner-bonafide {
        background: linear-gradient(135deg, #1e3d2f 0%, #2e7d32 100%);
        border-left: 8px solid #4caf50;
        padding: 1.5rem;
        border-radius: 8px;
        color: #e8f5e9;
        margin-bottom: 1.5rem;
    }
    .banner-spoofed {
        background: linear-gradient(135deg, #4d1d24 0%, #c62828 100%);
        border-left: 8px solid #e53935;
        padding: 1.5rem;
        border-radius: 8px;
        color: #ffebee;
        margin-bottom: 1.5rem;
    }
    .audit-reason-box {
        background-color: #1e1e24;
        border-radius: 6px;
        padding: 10px;
        margin-top: 8px;
        border-left: 3px solid #88c0d0;
    }
    .edu-box {
        background-color: #1a1d2e;
        border: 1px solid #3b4252;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #d8dee9;
    }
    .edu-box strong { color: #88c0d0; }
    .gemini-box {
        background: linear-gradient(135deg, #1a1d2e 0%, #252a3a 100%);
        border: 1px solid #5e81ac;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #eceff4;
    }
    .gemini-box h4 { color: #88c0d0; margin-top: 0; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# 1. Custom Attention Layer
# ─────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable()
class SelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(SelfAttention, self).build(input_shape)
    def call(self, x):
        et = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        at = tf.nn.softmax(tf.squeeze(et, axis=-1), axis=1)
        at = tf.expand_dims(at, axis=-1)
        return tf.reduce_sum(x * at, axis=1)

# ─────────────────────────────────────────────────────────
# 2. Load Models
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    brain = tf.keras.models.load_model('enhanced_attention_model_v4_robust.keras',
                                       custom_objects={'SelfAttention': SelfAttention},
                                       compile=False)
    feat_extractor = tf.keras.Model(inputs=brain.input, outputs=brain.layers[-3].output)
    xgb_judge = joblib.load('xgboost_refiner_v4_robust.pkl')
    return feat_extractor, xgb_judge

feat_ext, xgb = load_models()

# ─────────────────────────────────────────────────────────
# 3. Enhanced Neural Feature Extractor (60-D)
# ─────────────────────────────────────────────────────────
def extract_60d_features(audio_data):
    audio = librosa.effects.preemphasis(audio_data)
    audio = librosa.util.normalize(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate([mfcc, delta, delta2], axis=0)
    if combined.shape[1] < 200:
        combined = np.pad(combined, ((0,0), (0, 200 - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :200]
    return combined.T

# ─────────────────────────────────────────────────────────
# 4. Physical Acoustic Feature Diagnostic
# ─────────────────────────────────────────────────────────
def compute_explainable_metrics(audio_data, sr=16000):
    metrics = {}
    try:
        f0 = librosa.yin(audio_data, fmin=75, fmax=400, sr=sr)
        f0_clean = f0[np.isfinite(f0) & (f0 > 0)]
        if len(f0_clean) > 5:
            metrics['pitch_mean'] = float(np.mean(f0_clean))
            metrics['pitch_std'] = float(np.std(f0_clean))
            metrics['pitch_regularity'] = 1.0 / (metrics['pitch_std'] + 1e-5)
            metrics['jitter_approx'] = float(np.std(np.abs(np.diff(f0_clean))))
        else:
            metrics['pitch_mean'] = 0.0
            metrics['pitch_std'] = 0.0
            metrics['pitch_regularity'] = 0.0
            metrics['jitter_approx'] = 0.0
    except Exception:
        metrics['pitch_mean'] = 0.0
        metrics['pitch_std'] = 0.0
        metrics['pitch_regularity'] = 0.0
        metrics['jitter_approx'] = 0.0

    flatness = librosa.feature.spectral_flatness(y=audio_data)
    metrics['spectral_flatness_mean'] = float(np.mean(flatness))
    metrics['spectral_flatness_std'] = float(np.std(flatness))

    centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    metrics['spectral_centroid_mean'] = float(np.mean(centroid))
    metrics['spectral_centroid_std'] = float(np.std(centroid))

    zcr = librosa.feature.zero_crossing_rate(audio_data)
    metrics['zcr_mean'] = float(np.mean(zcr))
    metrics['zcr_std'] = float(np.std(zcr))

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_stds = np.std(mfcc, axis=1)
    metrics['mfcc_var_mean'] = float(np.mean(mfcc_stds))

    diffs_amp = np.abs(np.diff(audio_data))
    metrics['shimmer_approx'] = float(np.std(diffs_amp))

    rms = librosa.feature.rms(y=audio_data)
    metrics['rms_mean'] = float(np.mean(rms))
    metrics['rms_std'] = float(np.std(rms))

    # Audio duration
    metrics['duration_sec'] = float(len(audio_data) / sr)

    return metrics

# ─────────────────────────────────────────────────────────
# 5. Heuristic Suspicion Scoring
# ─────────────────────────────────────────────────────────
def compute_heuristic_suspicion(metrics):
    suspicion_points = 0.0
    max_points = 6.0
    details = []

    # Feature 1: MFCC Timbre Variance
    if metrics['mfcc_var_mean'] < 2.5:
        suspicion_points += 1.5
        details.append({"name": "Vocal Timbre Consistency", "val_str": f"{metrics['mfcc_var_mean']:.2f}",
                         "status": "🔴 HIGHLY SUSPICIOUS", "flag": "critical"})
    elif metrics['mfcc_var_mean'] < 3.2:
        suspicion_points += 0.8
        details.append({"name": "Vocal Timbre Consistency", "val_str": f"{metrics['mfcc_var_mean']:.2f}",
                         "status": "🟡 SUSPICIOUS", "flag": "warning"})
    else:
        details.append({"name": "Vocal Timbre Consistency", "val_str": f"{metrics['mfcc_var_mean']:.2f}",
                         "status": "🟢 NORMAL", "flag": "normal"})

    # Feature 2: Pitch standard deviation
    if metrics['pitch_mean'] > 0:
        if metrics['pitch_std'] < 12.0:
            suspicion_points += 1.2
            details.append({"name": "Intonation Melody (F0)", "val_str": f"σ = {metrics['pitch_std']:.1f} Hz",
                             "status": "🔴 HIGHLY SUSPICIOUS", "flag": "critical"})
        elif metrics['pitch_std'] < 22.0:
            suspicion_points += 0.6
            details.append({"name": "Intonation Melody (F0)", "val_str": f"σ = {metrics['pitch_std']:.1f} Hz",
                             "status": "🟡 SUSPICIOUS", "flag": "warning"})
        else:
            details.append({"name": "Intonation Melody (F0)", "val_str": f"σ = {metrics['pitch_std']:.1f} Hz",
                             "status": "🟢 NORMAL", "flag": "normal"})
    else:
        details.append({"name": "Intonation Melody (F0)", "val_str": "Unvoiced / Whisper",
                         "status": "⚪ AUDIT SKIPPED", "flag": "skipped"})

    # Feature 3: Spectral Flatness
    if metrics['spectral_flatness_mean'] < 0.0004:
        suspicion_points += 1.0
        details.append({"name": "Spectral Envelope Flatness", "val_str": f"{metrics['spectral_flatness_mean']:.6f}",
                         "status": "🔴 HIGHLY SUSPICIOUS", "flag": "critical"})
    elif metrics['spectral_flatness_mean'] < 0.0012:
        suspicion_points += 0.5
        details.append({"name": "Spectral Envelope Flatness", "val_str": f"{metrics['spectral_flatness_mean']:.6f}",
                         "status": "🟡 SUSPICIOUS", "flag": "warning"})
    else:
        details.append({"name": "Spectral Envelope Flatness", "val_str": f"{metrics['spectral_flatness_mean']:.6f}",
                         "status": "🟢 NORMAL", "flag": "normal"})

    # Feature 4: ZCR Variance
    if metrics['zcr_std'] < 0.012:
        suspicion_points += 0.8
        details.append({"name": "Acoustic Silence Gating", "val_str": f"σ = {metrics['zcr_std']:.4f}",
                         "status": "🔴 HIGHLY SUSPICIOUS", "flag": "critical"})
    else:
        details.append({"name": "Acoustic Silence Gating", "val_str": f"σ = {metrics['zcr_std']:.4f}",
                         "status": "🟢 NORMAL", "flag": "normal"})

    # Feature 5: Shimmer
    if metrics['shimmer_approx'] < 0.02:
        suspicion_points += 1.0
        details.append({"name": "Micro-Amplitude Instability (Shimmer)", "val_str": f"{metrics['shimmer_approx']:.4f}",
                         "status": "🔴 HIGHLY SUSPICIOUS", "flag": "critical"})
    else:
        details.append({"name": "Micro-Amplitude Instability (Shimmer)", "val_str": f"{metrics['shimmer_approx']:.4f}",
                         "status": "🟢 NORMAL", "flag": "normal"})

    # Feature 6: Jitter
    if metrics['pitch_mean'] > 0:
        if metrics['jitter_approx'] < 1.2:
            suspicion_points += 0.5
            details.append({"name": "Micro-Pitch Jitter", "val_str": f"{metrics['jitter_approx']:.2f} Hz",
                             "status": "🔴 HIGHLY SUSPICIOUS", "flag": "critical"})
        else:
            details.append({"name": "Micro-Pitch Jitter", "val_str": f"{metrics['jitter_approx']:.2f} Hz",
                             "status": "🟢 NORMAL", "flag": "normal"})

    suspicion_score = min(float(suspicion_points / max_points), 1.0)
    return suspicion_score, details

# ─────────────────────────────────────────────────────────
# 6. Gemini AI Dynamic Analysis Engine
# ─────────────────────────────────────────────────────────
def _detect_gemini_model(client):
    """Auto-detect a working Gemini model by listing available models."""
    # Check session cache first
    if 'gemini_model_name' in st.session_state and st.session_state['gemini_model_name']:
        return st.session_state['gemini_model_name']

    # Preferred model names in priority order
    preferred = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash",
                 "gemini-2.5-pro", "gemini-2.0-flash-001"]
    try:
        available = [m.name for m in client.models.list()]
        # Try preferred models first
        for pref in preferred:
            if pref in available or f"models/{pref}" in available:
                st.session_state['gemini_model_name'] = pref
                return pref
        # Fallback: pick first model that supports generateContent
        for m in client.models.list():
            name = m.name.replace("models/", "") if m.name.startswith("models/") else m.name
            if "gemini" in name.lower() and "flash" in name.lower():
                st.session_state['gemini_model_name'] = name
                return name
        # Last resort: just use the first gemini model
        for m in client.models.list():
            name = m.name.replace("models/", "") if m.name.startswith("models/") else m.name
            if "gemini" in name.lower():
                st.session_state['gemini_model_name'] = name
                return name
    except Exception:
        pass

    # Absolute fallback
    st.session_state['gemini_model_name'] = "gemini-2.5-flash"
    return "gemini-2.5-flash"


def _call_gemini(client, prompt):
    """Call Gemini using the auto-detected model. Returns (text, error)."""
    model_name = _detect_gemini_model(client)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text.strip(), None
    except Exception as e:
        # If detected model fails, clear cache and try once more with re-detection
        st.session_state.pop('gemini_model_name', None)
        model_name = _detect_gemini_model(client)
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text.strip(), None
        except Exception as e2:
            return None, f"Model '{model_name}': {type(e2).__name__}: {e2}"


def generate_gemini_verdict_reasoning(client, verdict_label, ensemble_prob, model_prob,
                                       heuristic_prob, metrics, audit_details):
    """Generates a dynamic, context-aware verdict explanation using Gemini."""
    flags_summary = "\n".join(
        [f"  - {d['name']}: {d['status']} (value: {d['val_str']})" for d in audit_details]
    )

    prompt = f"""You are an expert audio forensics analyst specializing in deepfake voice detection.

An audio sample was just analyzed by our detection system. Here are the results:

FINAL VERDICT: {verdict_label}
- Combined Spoof Probability: {ensemble_prob*100:.1f}%
- Neural Model Spoof Probability: {model_prob*100:.1f}%
- Acoustic Heuristic Suspicion Score: {heuristic_prob*100:.1f}%

Raw Acoustic Measurements:
- Fundamental Pitch (F0): mean={metrics['pitch_mean']:.1f} Hz, std={metrics['pitch_std']:.2f} Hz
- Spectral Flatness: {metrics['spectral_flatness_mean']:.6f}
- MFCC Temporal Variance: {metrics['mfcc_var_mean']:.3f}
- Zero Crossing Rate std: {metrics['zcr_std']:.4f}
- Shimmer (amplitude instability): {metrics['shimmer_approx']:.4f}
- Jitter (pitch micro-variation): {metrics['jitter_approx']:.2f} Hz
- RMS Energy: mean={metrics['rms_mean']:.4f}, std={metrics['rms_std']:.4f}
- Audio Duration: {metrics['duration_sec']:.2f} seconds

Diagnostic Flag Summary:
{flags_summary}

Based on these actual measurements, write a clear 3-4 sentence forensic reasoning paragraph explaining WHY this audio was classified as {verdict_label}. Reference the specific metric values that most influenced the decision. Use plain language that a university student can understand. Do NOT use markdown formatting, just plain text."""

    text, error = _call_gemini(client, prompt)
    if error:
        st.session_state['gemini_last_error'] = error
    return text


def generate_gemini_feature_explanation(client, feature_name, feature_value, feature_status):
    """Generates a dynamic explanation for a single diagnostic feature using Gemini."""
    prompt = f"""You are an expert audio forensics teacher explaining deepfake voice detection to university students.

A diagnostic test called "{feature_name}" was run on an audio sample.
- Measured Value: {feature_value}
- Status: {feature_status}

In exactly 2-3 sentences, explain:
1. What this test measures in simple terms (relate it to how the human voice works physically).
2. Why this specific result ({feature_status}) matters for detecting whether the voice is real or AI-generated.

Use plain, conversational language. Do NOT use markdown formatting."""

    text, error = _call_gemini(client, prompt)
    if error:
        st.session_state['gemini_last_error'] = error
    return text


def generate_gemini_section_description(client, section_name, context_info):
    """Generates an educational description for a visualization section."""
    prompt = f"""You are a friendly audio science teacher writing a short description panel for a web app dashboard.

The section is called: "{section_name}"
Context: {context_info}

Write exactly 2-3 sentences explaining what this visualization shows and why it matters for detecting deepfake voices. A first-year university student should understand it. Be concise and informative. Do NOT use markdown formatting."""

    text, error = _call_gemini(client, prompt)
    if error:
        st.session_state['gemini_last_error'] = error
    return text


# Fallback static descriptions (used when Gemini API is unavailable)
FALLBACK_DESCRIPTIONS = {
    "waveform": "This chart shows how loud the audio signal is over time. The vertical axis represents sound pressure (amplitude) and the horizontal axis is time in seconds. Human voices typically show irregular amplitude patterns with natural pauses and breaths, while AI-generated voices sometimes show unnaturally consistent or overly smooth amplitude envelopes.",
    "vocal_texture": "This heatmap visualizes the 60-dimensional spectral fingerprint of the voice — it combines 20 MFCCs (which capture the tonal 'color' of the voice), plus their first and second derivatives (which capture how quickly that color changes). Each horizontal band is one feature dimension and colors show intensity over time. Human voices produce rich, constantly shifting patterns, whereas AI-synthesized voices often appear smoother and more repetitive.",
    "pitch_tracking": "This plot traces the fundamental frequency (F0) of the speaker's voice across time — essentially the 'musical note' the vocal cords are vibrating at in each moment. Human speakers naturally vary their pitch as they emphasize words, ask questions, or express emotions. AI-generated voices often show unnaturally flat or mechanically stepped pitch contours because speech synthesis models struggle to replicate the organic randomness of real vocal cord vibration.",
    "model_embedding": "This bar chart shows the internal neural 'fingerprint' that our deep learning model extracted from the audio. Each bar represents one dimension of the model's bottleneck layer — the compressed representation the neural network learned to distinguish real from fake voices. Certain activation patterns are characteristic of synthetic speech. The model uses this vector to make its classification decision.",
    "verdict_spoof": "Artificial voice patterns were detected. The combination of acoustic measurements suggests this audio lacks the natural physical imperfections that come from a biological vocal tract. Key indicators include overly stable pitch, low timbral variation, and absence of muscular micro-tremors.",
    "verdict_bonafide": "The audio exhibits natural characteristics consistent with human speech production. The measured pitch variations, spectral dynamics, and amplitude micro-fluctuations are within ranges expected from biological vocal cords interacting with a real vocal tract."
}

FALLBACK_FEATURE_EXPLANATIONS = {
    "Vocal Timbre Consistency": {
        "critical": "The voice has an unnaturally uniform tonal color over time. When humans speak, the shape of our mouth, tongue, and throat constantly shifts, creating rich variation in the MFCC features. This sample's low variance suggests a vocoder or neural synthesizer that produces overly smooth, repetitive vocal textures.",
        "warning": "The tonal texture of this voice is smoother than typical human speech but not definitively artificial. This could indicate a professional studio recording with noise reduction, or it could be a sign of high-quality neural speech synthesis.",
        "normal": "The voice shows healthy variation in its tonal characteristics across time. This kind of dynamic, shifting vocal texture is produced naturally when air passes through the constantly reshaping human vocal tract."
    },
    "Intonation Melody (F0)": {
        "critical": "The pitch of this voice barely changes throughout the recording. Human speech naturally rises and falls as we emphasize words, ask questions, or express emotions. A standard deviation below 12 Hz indicates an almost monotone delivery, which is a hallmark of older text-to-speech systems and some neural voice cloners.",
        "warning": "The pitch variation in this voice is below the typical range for conversational speech. While some people naturally speak in a flatter tone (especially when reading), this level of monotony is also common in AI-generated voices.",
        "normal": "The pitch shows natural expressive variation. The speaker's voice rises and falls in ways consistent with genuine human emotional expression and speech prosody.",
        "skipped": "The pitch tracking algorithm could not detect a reliable fundamental frequency. This can happen with whispered speech, very noisy recordings, or extremely short audio clips."
    },
    "Spectral Envelope Flatness": {
        "critical": "The frequency spectrum of this audio is unnaturally uniform. Real human voices produce complex, peaked spectral shapes because of the resonances of our vocal tract (formants). A very low spectral flatness value suggests the presence of vocoder artifacts or artificial spectral shaping.",
        "warning": "The spectral shape is flatter than typical speech, possibly indicating some form of digital audio processing or a vocoder-based synthesis method that hasn't fully replicated natural formant structure.",
        "normal": "The spectral envelope shows the complex, peaked structure typical of natural human speech. The presence of distinct formant regions indicates genuine acoustic resonance from a biological vocal tract."
    },
    "Acoustic Silence Gating": {
        "critical": "The transitions between voiced sounds and silence are unnaturally consistent. Human speech naturally includes varied breathing noises, lip smacks, and irregular silence patterns. The very low zero-crossing rate variance suggests synthetic noise gating — where an algorithm artificially cleans or pads silent regions.",
        "normal": "The silence and breathing patterns in this audio show natural variation. The transitions between speech and quiet moments have the irregular, organic quality expected from real recordings of human speakers."
    },
    "Micro-Amplitude Instability (Shimmer)": {
        "critical": "The amplitude of the waveform is unnervingly stable cycle-to-cycle. Real vocal cords are muscles that vibrate slightly differently with each cycle, producing natural shimmer (tiny loudness fluctuations). The absence of this shimmer strongly suggests the waveform was mathematically generated rather than physically produced.",
        "normal": "The audio shows natural cycle-to-cycle amplitude variations (shimmer). This micro-instability is a hallmark of real vocal cords — biological muscles that can never vibrate with perfect mathematical precision."
    },
    "Micro-Pitch Jitter": {
        "critical": "The pitch is unnervingly precise from cycle to cycle. Real vocal cord vibration has natural micro-instability (jitter) because biological tissue cannot vibrate at exactly the same frequency twice. Very low jitter is a tell-tale sign of synthetic waveform generation where pitch is controlled by a mathematical function.",
        "normal": "The pitch shows healthy cycle-to-cycle micro-variation (jitter). This natural instability is produced by the physical imprecision of biological vocal cord vibration and is very difficult for AI synthesizers to replicate perfectly."
    }
}

# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Deepfake Audio Auditor v4 Pro")
    st.markdown("---")

    st.markdown("### 🤖 Gemini AI Analysis")
    gemini_api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Enter your Google Gemini API key to enable AI-powered dynamic analysis. Get a free key at https://aistudio.google.com/apikey"
    )

    gemini_client = None
    if gemini_api_key and GEMINI_AVAILABLE:
        try:
            gemini_client = genai.Client(api_key=gemini_api_key)
            # Auto-detect the correct model for this API key
            detected_model = _detect_gemini_model(gemini_client)
            st.success(f"✅ Gemini AI connected (model: `{detected_model}`)")
        except Exception as e:
            st.error(f"❌ Failed to connect: {e}")
    elif not GEMINI_AVAILABLE:
        st.warning("⚠️ google-genai package not installed")
    else:
        st.info("💡 Add your API key for AI-powered explanations. Without it, built-in analysis will be used.")

    st.markdown("---")
    st.info("""
    **Core Pipeline:**
    - 🎵 60-D Spectral Feature Extraction (MFCC + Δ + Δ²)
    - 🧠 Neural Attention Network (Keras)
    - 🌲 Gradient Boosting Refiner (XGBoost)
    - 🔬 Physiological Acoustic Diagnostics
    - 🤖 Gemini AI Contextual Reasoning
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Detection Sensitivity")
    st_sensitivity = st.slider(
        "Spoof Sensitivity Threshold", 0.30, 0.70, 0.45, step=0.05,
        help="Lower values = more sensitive to AI deepfakes (fewer false negatives). Higher values = fewer false alarms. Default 0.45 is tuned for high sensitivity."
    )
    st.markdown("---")
    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
        **Step 1 — Feature Extraction:** The audio is converted into a 60-dimensional spectral representation using MFCCs (Mel-Frequency Cepstral Coefficients), capturing the voice's tonal fingerprint.

        **Step 2 — Neural Analysis:** A Keras neural network with a custom self-attention layer processes the spectral features and compresses them into a dense embedding vector.

        **Step 3 — Classification:** An XGBoost gradient boosting model reads the neural embedding and outputs a probability that the voice is synthetic.

        **Step 4 — Acoustic Diagnostics:** Six independent physical measurements (pitch jitter, shimmer, spectral flatness, etc.) are analyzed as heuristic cross-checks.

        **Step 5 — Ensemble Fusion:** The neural model probability (65% weight) is combined with the acoustic heuristic score (35% weight) for a robust final verdict.

        **Step 6 — AI Reasoning:** When a Gemini API key is provided, the system sends all metrics to Google Gemini to generate a unique, contextual forensic explanation for each specific audio sample.
        """)

# ─────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────
st.title("🛡️ Deepfake Audio Auditor: Specialist V4 Pro")
st.markdown("Advanced Voice Authenticity Diagnostics with AI-Powered Explainability")
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload a Speech Sample (.mp3, .wav, .flac)", type=['flac', 'mp3', 'wav'])

if uploaded_file:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes)
    audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    if st.button("🚀 RUN COMPREHENSIVE BIOMETRIC SPEECH AUDIT", use_container_width=True):
        with st.spinner("🔬 Extracting spectral features, running neural inference, computing acoustic diagnostics..."):

            # ── Pipeline Execution ──
            features = extract_60d_features(audio_data)
            deep_feats = feat_ext.predict(np.expand_dims(features, axis=0), verbose=0)
            model_prob = float(xgb.predict_proba(deep_feats)[0][1])
            metrics = compute_explainable_metrics(audio_data, sr=16000)
            heuristic_prob, audit_details = compute_heuristic_suspicion(metrics)
            ensemble_prob = 0.65 * model_prob + 0.35 * heuristic_prob
            is_ai = ensemble_prob > st_sensitivity
            verdict_label = "SPOOFED / AI-GENERATED" if is_ai else "BONAFIDE / HUMAN"

            # ── Generate Gemini reasoning (if available) ──
            gemini_verdict_text = None
            if gemini_client:
                with st.spinner("🤖 Gemini is analyzing your audio forensics data..."):
                    gemini_verdict_text = generate_gemini_verdict_reasoning(
                        gemini_client, verdict_label, ensemble_prob, model_prob,
                        heuristic_prob, metrics, audit_details
                    )

        # ── UI Tabs ──
        tab_verdict, tab_explain, tab_metrics, tab_neural = st.tabs([
            "📊 Verdict & Signals",
            "🔍 AI-Powered Diagnostics",
            "📈 Physical Signal Metrics",
            "🧠 Model Embedding"
        ])

        # ═══════════════════════════════════════════════
        # TAB 1: Verdict & Signals
        # ═══════════════════════════════════════════════
        with tab_verdict:
            st.subheader("Authenticity Analysis Verdict")

            v_col1, v_col2, v_col3 = st.columns([2, 1, 1])

            with v_col1:
                if is_ai:
                    st.markdown("""
                    <div class="banner-spoofed">
                        <h2 style="margin:0; color:#ffffff;">🚨 SPOOFED / DEEPFAKE SPEECH DETECTED</h2>
                        <p style="margin:5px 0 0 0; font-size:1.1rem; opacity:0.9;">
                            This voice has been flagged as <b>SYNTHETIC / AI-GENERATED</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="banner-bonafide">
                        <h2 style="margin:0; color:#ffffff;">✅ BONAFIDE HUMAN SPEECH VERIFIED</h2>
                        <p style="margin:5px 0 0 0; font-size:1.1rem; opacity:0.9;">
                            This voice is determined to be <b>GENUINE / HUMAN-SPOKEN</b>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with v_col2:
                st.metric("Synthesized Probability", f"{ensemble_prob*100:.1f}%")
                st.progress(float(ensemble_prob))

            with v_col3:
                if ensemble_prob > 0.70:
                    risk_level = "HIGH 🔴"
                elif ensemble_prob > st_sensitivity:
                    risk_level = "MEDIUM 🟡"
                else:
                    risk_level = "LOW 🟢"
                st.metric("Risk Level", risk_level)

            # ── AI-Generated Reasoning ──
            st.markdown("#### 🤖 Forensic Reasoning")
            if gemini_verdict_text:
                st.markdown(f'<div class="gemini-box"><h4>✨ Gemini AI Analysis</h4>{gemini_verdict_text}</div>',
                            unsafe_allow_html=True)
            else:
                fallback_key = "verdict_spoof" if is_ai else "verdict_bonafide"
                st.info(FALLBACK_DESCRIPTIONS[fallback_key])
                if gemini_client:
                    gemini_err = st.session_state.get('gemini_last_error', 'Unknown error')
                    st.warning(f"⚠️ Gemini API call failed — showing built-in analysis instead.\n\n**Error:** `{gemini_err}`")
                else:
                    st.caption("💡 _Add a Gemini API key in the sidebar to get AI-generated analysis tailored to this specific audio sample._")

            st.markdown("---")

            # ── Waveform & Vocal Texture ──
            col_w1, col_w2 = st.columns(2)

            with col_w1:
                st.subheader("🎙️ Waveform Analysis")
                # Educational description
                if gemini_client:
                    desc = generate_gemini_section_description(
                        gemini_client, "Waveform Analysis",
                        f"Audio duration: {metrics['duration_sec']:.2f}s, RMS energy mean: {metrics['rms_mean']:.4f}, RMS std: {metrics['rms_std']:.4f}"
                    )
                    if desc:
                        st.markdown(f'<div class="edu-box">{desc}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="edu-box">{FALLBACK_DESCRIPTIONS["waveform"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="edu-box">{FALLBACK_DESCRIPTIONS["waveform"]}</div>', unsafe_allow_html=True)

                fig_wav, ax_wav = plt.subplots(figsize=(10, 4.5))
                plt.style.use('dark_background')
                librosa.display.waveshow(audio_data, sr=16000, ax=ax_wav, color='#88c0d0')
                ax_wav.set_title("Amplitude vs. Time", fontsize=12, pad=10)
                ax_wav.set_xlabel("Time (s)", color='#eceff4')
                ax_wav.set_ylabel("Amplitude", color='#eceff4')
                ax_wav.grid(True, linestyle='--', alpha=0.3, color='#4c566a')
                fig_wav.patch.set_facecolor('#1e2230')
                ax_wav.set_facecolor('#1e2230')
                st.pyplot(fig_wav)

            with col_w2:
                st.subheader("🌈 Vocal Texture Map")
                if gemini_client:
                    desc = generate_gemini_section_description(
                        gemini_client, "Vocal Texture Map (60-D Spectral Fingerprint)",
                        f"60 feature dimensions (20 MFCCs + 20 delta + 20 delta-delta) plotted over 200 time frames. MFCC variance mean: {metrics['mfcc_var_mean']:.3f}"
                    )
                    if desc:
                        st.markdown(f'<div class="edu-box">{desc}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="edu-box">{FALLBACK_DESCRIPTIONS["vocal_texture"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="edu-box">{FALLBACK_DESCRIPTIONS["vocal_texture"]}</div>', unsafe_allow_html=True)

                fig_spec, ax_spec = plt.subplots(figsize=(10, 4.5))
                plt.style.use('dark_background')
                img = librosa.display.specshow(
                    features.T, x_axis='time', sr=16000, ax=ax_spec, cmap='magma',
                    vmax=np.percentile(features, 95), vmin=np.percentile(features, 5)
                )
                ax_spec.set_title("60-D Spectral Fingerprint (MFCC + Δ + Δ²)", fontsize=12, pad=10)
                fig_spec.colorbar(img, ax=ax_spec, format="%+2.f")
                fig_spec.patch.set_facecolor('#1e2230')
                ax_spec.set_facecolor('#1e2230')
                st.pyplot(fig_spec)

        # ═══════════════════════════════════════════════
        # TAB 2: AI-Powered Diagnostics
        # ═══════════════════════════════════════════════
        with tab_explain:
            st.subheader("🔍 Acoustic Justification & Auditing Report")
            st.markdown("Each diagnostic test below examines a different physical property of the voice. "
                        "The explanations are generated dynamically based on the actual measurements from your audio sample.")
            st.markdown("---")

            for detail in audit_details:
                status_class = "status-normal"
                if "HIGHLY" in detail["status"]:
                    status_class = "status-critical"
                elif "SUSPICIOUS" in detail["status"]:
                    status_class = "status-warning"

                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                        <span style="font-weight:bold; font-size:1.1rem; color:#eceff4;">{detail['name']}</span>
                        <span class="{status_class}" style="font-weight:bold; padding: 4px 10px; border-radius: 4px; border: 1px solid currentColor;">
                            {detail['status']} (value: {detail['val_str']})
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Generate dynamic explanation per feature
                if gemini_client:
                    explanation = generate_gemini_feature_explanation(
                        gemini_client, detail['name'], detail['val_str'], detail['status']
                    )
                    if explanation:
                        st.markdown(f'<div class="gemini-box"><h4>✨ Gemini Explanation</h4>{explanation}</div>',
                                    unsafe_allow_html=True)
                    else:
                        # Fallback
                        fb = FALLBACK_FEATURE_EXPLANATIONS.get(detail['name'], {}).get(detail['flag'], "")
                        if fb:
                            st.markdown(f'<div class="audit-reason-box"><p style="margin:0; color:#eceff4;">{fb}</p></div>',
                                        unsafe_allow_html=True)
                else:
                    fb = FALLBACK_FEATURE_EXPLANATIONS.get(detail['name'], {}).get(detail['flag'], "")
                    if fb:
                        st.markdown(f'<div class="audit-reason-box"><p style="margin:0; color:#eceff4;">{fb}</p></div>',
                                    unsafe_allow_html=True)

                st.markdown("")  # spacing

        # ═══════════════════════════════════════════════
        # TAB 3: Physical Signal Metrics
        # ═══════════════════════════════════════════════
        with tab_metrics:
            st.subheader("📈 Raw Acoustic Diagnostics")
            st.markdown("These are the raw numerical measurements extracted from the audio signal. "
                        "Each metric captures a different physical property of how the voice was produced.")
            st.markdown("---")

            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Fundamental Pitch (Mean)</div>
                    <div class="metric-value">{metrics['pitch_mean']:.1f} Hz</div>
                    <div class="metric-status status-normal">The average vibration frequency of the vocal cords. Typical male: 85-180 Hz, female: 165-255 Hz.</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Spectral Flatness (Mean)</div>
                    <div class="metric-value">{metrics['spectral_flatness_mean']:.5f}</div>
                    <div class="metric-status">Measures how 'noise-like' vs 'tone-like' the signal is. Values near 0 = tonal (speech), near 1 = noisy (hiss). Very low values can indicate vocoder artifacts.</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Intonation Standard Dev</div>
                    <div class="metric-value">{metrics['pitch_std']:.2f} Hz</div>
                    <div class="metric-status">How much the pitch varies over time. Low values (&lt;12 Hz) suggest monotone/robotic speech. Natural speech: 20-60 Hz.</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">MFCC Texture Variance</div>
                    <div class="metric-value">{metrics['mfcc_var_mean']:.3f}</div>
                    <div class="metric-status">Captures the variety of vocal timbres over time. Low values mean the voice sounds repetitively 'same' — a common trait of AI voices.</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Waveform Shimmer</div>
                    <div class="metric-value">{metrics['shimmer_approx']:.4f}</div>
                    <div class="metric-status">Micro-fluctuations in loudness between adjacent wave cycles. Real vocal cords produce measurable shimmer; synthesizers tend to produce perfectly stable amplitudes.</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Noise Stability (ZCR std)</div>
                    <div class="metric-value">{metrics['zcr_std']:.4f}</div>
                    <div class="metric-status">Variation in how often the signal crosses zero. Low values suggest artificially gated silence regions rather than natural breathing and pauses.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Pitch Contour Plot ──
            st.subheader("🎵 Pitch Tracking (F0 Contour)")
            pitch_desc_key = "pitch_tracking"
            if gemini_client:
                desc = generate_gemini_section_description(
                    gemini_client, "Pitch Tracking (F0 Contour)",
                    f"Pitch mean: {metrics['pitch_mean']:.1f} Hz, std: {metrics['pitch_std']:.2f} Hz, jitter: {metrics['jitter_approx']:.2f} Hz. This plot shows the fundamental frequency over time."
                )
                if desc:
                    st.markdown(f'<div class="edu-box">{desc}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="edu-box">{FALLBACK_DESCRIPTIONS[pitch_desc_key]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="edu-box">{FALLBACK_DESCRIPTIONS[pitch_desc_key]}</div>', unsafe_allow_html=True)

            try:
                f_yin = librosa.yin(audio_data, fmin=75, fmax=400, sr=16000)
                fig_p, ax_p = plt.subplots(figsize=(10, 3.5))
                plt.style.use('dark_background')
                ax_p.plot(f_yin, color='#a3be8c', linewidth=2, label="F0 Pitch Contour")
                ax_p.set_ylabel("Frequency (Hz)")
                ax_p.set_xlabel("Frames")
                ax_p.set_title("Fundamental Frequency Over Time", fontsize=12, pad=10)
                ax_p.grid(True, linestyle=':', alpha=0.4)
                ax_p.legend()
                fig_p.patch.set_facecolor('#1e2230')
                ax_p.set_facecolor('#1e2230')
                st.pyplot(fig_p)
            except Exception:
                st.info("Pitch contour could not be plotted — the fundamental frequency (F0) was undetectable in this audio.")

        # ═══════════════════════════════════════════════
        # TAB 4: Model Embedding
        # ═══════════════════════════════════════════════
        with tab_neural:
            st.subheader("🧠 Deep Neural Representation (Bottleneck Features)")

            if gemini_client:
                emb_stats = f"Embedding dimensions: {len(deep_feats[0])}, max activation: {np.max(deep_feats[0]):.3f}, min: {np.min(deep_feats[0]):.3f}, mean: {np.mean(deep_feats[0]):.3f}"
                desc = generate_gemini_section_description(
                    gemini_client, "Neural Network Bottleneck Embedding",
                    f"This is the internal representation from the attention-pooled dense layer. {emb_stats}. The XGBoost classifier reads this vector to predict spoof probability."
                )
                if desc:
                    st.markdown(f'<div class="edu-box">{desc}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="edu-box">{FALLBACK_DESCRIPTIONS["model_embedding"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="edu-box">{FALLBACK_DESCRIPTIONS["model_embedding"]}</div>', unsafe_allow_html=True)

            fig_emb, ax_emb = plt.subplots(figsize=(10, 3.5))
            plt.style.use('dark_background')
            emb_feats = deep_feats[0]
            colors = ['#bf616a' if v > np.percentile(emb_feats, 90) else '#ebcb8b' if v > np.percentile(emb_feats, 50) else '#5e81ac' for v in emb_feats]
            ax_emb.bar(range(len(emb_feats)), emb_feats, color=colors, edgecolor='none', width=0.8)
            ax_emb.set_xlabel("Embedding Dimension Index")
            ax_emb.set_ylabel("Activation Value")
            ax_emb.set_title("Neural Activation Signature Vector", fontsize=12, pad=10)
            fig_emb.patch.set_facecolor('#1e2230')
            ax_emb.set_facecolor('#1e2230')
            st.pyplot(fig_emb)

            st.caption("🔴 Red bars = top 10% activations (strongest signal) · 🟡 Yellow = above median · 🔵 Blue = below median")

            st.markdown("---")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.info(f"**Model Diagnostic Outputs:**\n"
                        f"- Neural model raw spoof probability: **{model_prob*100:.2f}%**\n"
                        f"- Acoustic heuristic suspicion score: **{heuristic_prob*100:.2f}%**\n"
                        f"- Embedding dimensions: **{len(emb_feats)}**")
            with col_d2:
                st.info(f"**Ensemble Fusion Formula:**\n"
                        f"- 0.65 × Neural Model ({model_prob*100:.1f}%) + 0.35 × Heuristics ({heuristic_prob*100:.1f}%)\n"
                        f"- **Final Score: {ensemble_prob*100:.2f}%** (threshold: {st_sensitivity:.2f})")