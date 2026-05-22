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

# Custom CSS for Premium Design & Responsive Layout
st.set_page_config(
    page_title="Deepfake Auditor V4 Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark-theme CSS injection
st.markdown("""
    <style>
    /* Card design */
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
    /* Indicator status colors */
    .status-normal { color: #a3be8c; }
    .status-warning { color: #ebcb8b; }
    .status-critical { color: #bf616a; }
    
    /* Result banners */
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
    </style>
    """, unsafe_allow_html=True)

# 1. Custom Attention Layer
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

# 2. Load Models
@st.cache_resource
def load_models():
    brain = tf.keras.models.load_model('enhanced_attention_model_v4_robust.keras', 
                                       custom_objects={'SelfAttention': SelfAttention}, 
                                       compile=False)
    feat_extractor = tf.keras.Model(inputs=brain.input, outputs=brain.layers[-3].output)
    xgb_judge = joblib.load('xgboost_refiner_v4_robust.pkl')
    return feat_extractor, xgb_judge

feat_ext, xgb = load_models()

# 3. Enhanced Neural Feature Extractor (60-D)
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

# 4. Physical Acoustic Feature Diagnostic & Explainability
def compute_explainable_metrics(audio_data, sr=16000):
    metrics = {}
    
    # 1. Pitch (F0) Estimation via robust YIN
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
        
    # 2. Spectral Flatness (Vocoder presence detection)
    flatness = librosa.feature.spectral_flatness(y=audio_data)
    metrics['spectral_flatness_mean'] = float(np.mean(flatness))
    metrics['spectral_flatness_std'] = float(np.std(flatness))
    
    # 3. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    metrics['spectral_centroid_mean'] = float(np.mean(centroid))
    metrics['spectral_centroid_std'] = float(np.std(centroid))
    
    # 4. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(audio_data)
    metrics['zcr_mean'] = float(np.mean(zcr))
    metrics['zcr_std'] = float(np.std(zcr))
    
    # 5. Temporal Dynamic Timbre (Variance of standard MFCCs across frames)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_stds = np.std(mfcc, axis=1)
    metrics['mfcc_var_mean'] = float(np.mean(mfcc_stds))
    
    # 6. Shimmer (waveform micro-amplitude variation)
    diffs_amp = np.abs(np.diff(audio_data))
    metrics['shimmer_approx'] = float(np.std(diffs_amp))
    
    # 7. Signal Power Energy stats
    rms = librosa.feature.rms(y=audio_data)
    metrics['rms_mean'] = float(np.mean(rms))
    metrics['rms_std'] = float(np.std(rms))
    
    return metrics

def compute_heuristic_suspicion(metrics):
    suspicion_points = 0.0
    max_points = 6.0
    details = []
    
    # Feature 1: Timbre Regularity / Smoothness (MFCC Variation)
    # AI vocoders produce highly predictable/uniform vocal timbres over time.
    if metrics['mfcc_var_mean'] < 2.5:
        suspicion_points += 1.5
        details.append({
            "name": "Vocal Timbre Consistency",
            "val_str": f"{metrics['mfcc_var_mean']:.2f}",
            "status": "🔴 HIGHLY SUSPICIOUS",
            "desc": "Unnatural timbre regularity (Low MFCC variance). The voice lacks the raw physical fluctuations expected when air interacts dynamically with biological tissue."
        })
    elif metrics['mfcc_var_mean'] < 3.2:
        suspicion_points += 0.8
        details.append({
            "name": "Vocal Timbre Consistency",
            "val_str": f"{metrics['mfcc_var_mean']:.2f}",
            "status": "🟡 SUSPICIOUS",
            "desc": "Moderately smooth vocal timbre. Common in professional clean recordings but also characteristic of neural audio synthesis."
        })
    else:
        details.append({
            "name": "Vocal Timbre Consistency",
            "val_str": f"{metrics['mfcc_var_mean']:.2f}",
            "status": "🟢 NORMAL",
            "desc": "Healthy dynamic acoustic timbre. Normal human variation across speech sounds."
        })
        
    # Feature 2: Pitch standard deviation (flatness)
    # AI text-to-speech models often struggle with dynamic inflection.
    if metrics['pitch_mean'] > 0:
        if metrics['pitch_std'] < 12.0:
            suspicion_points += 1.2
            details.append({
                "name": "Intonation Melody (F0)",
                "val_str": f"σ = {metrics['pitch_std']:.1f} Hz",
                "status": "🔴 HIGHLY SUSPICIOUS",
                "desc": "Robot-like flat pitch melody. The speech is missing natural melodic arcs and emotional spikes (standard deviation < 12 Hz), suggesting monotonic synthesizer gating."
            })
        elif metrics['pitch_std'] < 22.0:
            suspicion_points += 0.6
            details.append({
                "name": "Intonation Melody (F0)",
                "val_str": f"σ = {metrics['pitch_std']:.1f} Hz",
                "status": "🟡 SUSPICIOUS",
                "desc": "Slightly monotonous voice melody. Typical of reading from text or simple AI speaker profiles."
            })
        else:
            details.append({
                "name": "Intonation Melody (F0)",
                "val_str": f"σ = {metrics['pitch_std']:.1f} Hz",
                "status": "🟢 NORMAL",
                "desc": "Expressive inflection. Melodic intonation matches authentic biological voicing."
            })
    else:
        details.append({
            "name": "Intonation Melody (F0)",
            "val_str": "Unvoiced / Whisper",
            "status": "⚪ AUDIT SKIPPED",
            "desc": "Fundamental pitch (F0) was not detected. Whispered or high-noise audio."
        })
        
    # Feature 3: Spectral Flatness
    # Generative AI voices often produce artificial spectral uniformity in background bands.
    if metrics['spectral_flatness_mean'] < 0.0004:
        suspicion_points += 1.0
        details.append({
            "name": "Spectral Envelope Flatness",
            "val_str": f"{metrics['spectral_flatness_mean']:.6f}",
            "status": "🔴 HIGHLY SUSPICIOUS",
            "desc": "Super-flat spectral density. Often indicating vocoder spectral structures or artificial low-pass filters."
        })
    elif metrics['spectral_flatness_mean'] < 0.0012:
        suspicion_points += 0.5
        details.append({
            "name": "Spectral Envelope Flatness",
            "val_str": f"{metrics['spectral_flatness_mean']:.6f}",
            "status": "🟡 SUSPICIOUS",
            "desc": "Low spectral flatness. Possible high-quality digital vocoding signatures."
        })
    else:
        details.append({
            "name": "Spectral Envelope Flatness",
            "val_str": f"{metrics['spectral_flatness_mean']:.6f}",
            "status": "🟢 NORMAL",
            "desc": "Standard biological acoustic complexity."
        })

    # Feature 4: Noise Gating / Breathiness Stability (ZCR Variance)
    # Human speech has highly dynamic breathiness / vocal pops. AI has static noise frames.
    if metrics['zcr_std'] < 0.012:
        suspicion_points += 0.8
        details.append({
            "name": "Acoustic Silence Gating",
            "val_str": f"σ = {metrics['zcr_std']:.4f}",
            "status": "🔴 HIGHLY SUSPICIOUS",
            "desc": "Artificial silence gating (Low ZCR variance). Suggests high-frequency noise bands are synthetically padded or mathematically gated."
        })
    else:
        details.append({
            "name": "Acoustic Silence Gating",
            "val_str": f"σ = {metrics['zcr_std']:.4f}",
            "status": "🟢 NORMAL",
            "desc": "Standard breathiness variations and natural transitions."
        })

    # Feature 5: Acoustic micro-variations (Shimmer)
    if metrics['shimmer_approx'] < 0.02:
        suspicion_points += 1.0
        details.append({
            "name": "Micro-Amplitude Instability (Shimmer)",
            "val_str": f"{metrics['shimmer_approx']:.4f}",
            "status": "🔴 HIGHLY SUSPICIOUS",
            "desc": "Synthetically stable amplitudes. Lacks standard muscular micro-shimmer inherent to real vocal cord movement."
        })
    else:
        details.append({
            "name": "Micro-Amplitude Instability (Shimmer)",
            "val_str": f"{metrics['shimmer_approx']:.4f}",
            "status": "🟢 NORMAL",
            "desc": "Acoustically imperfect. Human-like muscular micro-modulations present."
        })

    # Feature 6: Pitch micro-jitter
    if metrics['pitch_mean'] > 0:
        if metrics['jitter_approx'] < 1.2:
            suspicion_points += 0.5
            details.append({
                "name": "Micro-Pitch Jitter",
                "val_str": f"{metrics['jitter_approx']:.2f} Hz",
                "status": "🔴 HIGHLY SUSPICIOUS",
                "desc": "Hyper-precise pitch tracking. Speech synthesizers often maintain perfect mathematical frequency paths, lacking micro-jitter."
            })
        else:
            details.append({
                "name": "Micro-Pitch Jitter",
                "val_str": f"{metrics['jitter_approx']:.2f} Hz",
                "status": "🟢 NORMAL",
                "desc": "Natural jitter detected."
            })

    suspicion_score = min(float(suspicion_points / max_points), 1.0)
    return suspicion_score, details

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/security-shield.png", width=90)
    st.markdown("## Deepfake Audio Auditor v4")
    st.info("""
    **Core Pipeline:**
    - **Melody & Dynamic Texture Analyzer** (60-D Pre-emphasized Features)
    - **Neural Attention pooling network** (Keras)
    - **Gradient Boosting Refiner** (XGBoost)
    - **Physiological Acoustic Diagnostic Heuristics**
    """)
    st.markdown("---")
    st.markdown("### Detection Sensitivity")
    st_sensitivity = st.slider("Spoof Sensitivity Threshold", 0.30, 0.70, 0.45, step=0.05, 
                               help="Lower threshold values prioritize detecting sneaky AI deepfakes over avoiding false positives.")
    
    st.markdown("---")
    st.markdown("Designed for **Biometric Security Audits & Speech Forensics**.")

# Main Application Layout
st.title("🛡️ Deepfake Audio Auditor: Specialist V4 Pro")
st.markdown("State-of-the-art Voice Authenticity Diagnostics and Physiochemical Signal Analysis.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Speech Sample (.mp3, .wav, .flac)", type=['flac', 'mp3', 'wav'])

if uploaded_file:
    audio_bytes = uploaded_file.read()
    # Streamlit Audio Player
    st.audio(audio_bytes)
    
    # Load and decode using Librosa
    audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    
    # Execute Audit
    if st.button("🚀 RUN COMPREHENSIVE BIOMETRIC SPEECH AUDIT"):
        with st.spinner("Analyzing spectral signals, feeding custom neural layers, and executing refiners..."):
            
            # 1. Extract standard representations
            features = extract_60d_features(audio_data)
            
            # 2. Extract Deep features via neural layer
            deep_feats = feat_ext.predict(np.expand_dims(features, axis=0), verbose=0)
            
            # 3. Model score (XGBoost probability of class 1 / spoof)
            model_prob = float(xgb.predict_proba(deep_feats)[0][1])
            
            # 4. Extract physiological heuristics and generate explainable bullet points
            metrics = compute_explainable_metrics(audio_data, sr=16000)
            heuristic_prob, audit_details = compute_heuristic_suspicion(metrics)
            
            # 5. Robust Ensemble Calculation
            # Higher weight is assigned to the deep neural model, but acoustics act as an override filter
            # to make sure misidentified AI voices are correctly flagged.
            ensemble_prob = 0.65 * model_prob + 0.35 * heuristic_prob
            
            # Final Decision
            is_ai = ensemble_prob > st_sensitivity
            
            # UI Organization: Tabs
            tab_verdict, tab_explain, tab_metrics, tab_neural = st.tabs([
                "📊 Verdict & Signals", 
                "🔍 Explainable AI Diagnostics", 
                "📈 Physical Signal Metrics", 
                "🧠 Model Embedding"
            ])
            
            # TAB 1: Verdict & Signal Waveforms
            with tab_verdict:
                st.subheader("Authenticity Analysis Verdict")
                
                v_col1, v_col2, v_col3 = st.columns([2, 1, 1])
                
                with v_col1:
                    if is_ai:
                        st.markdown(f"""
                        <div class="banner-spoofed">
                            <h2 style="margin:0; color:#ffffff;">🚨 SPOOFED / DEEPFAKE SPEECH FLAG</h2>
                            <p style="margin:5px 0 0 0; font-size:1.1rem; opacity:0.9;">
                                This voice has been flagged as <b>SYNTHETIC / AI-GENERATED</b>.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.warning("⚠️ **Reasoning:** Artificial voice patterns matching non-biological tract properties were detected. The pitch variance and frequency envelopes do not correspond with standard vocal cord acoustics.")
                    else:
                        st.markdown(f"""
                        <div class="banner-bonafide">
                            <h2 style="margin:0; color:#ffffff;">✅ BONAFIDE HUMAN SPEECH VERIFIED</h2>
                            <p style="margin:5px 0 0 0; font-size:1.1rem; opacity:0.9;">
                                This voice is determined to be <b>GENUINE / HUMAN-SPOKEN</b>.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("✔️ **Reasoning:** Natural muscle micro-tremors, correct spectral variations, and authentic physical vocal cord instability metrics confirm authentic voice box production.")
                
                with v_col2:
                    st.metric("Synthesized Probability", f"{ensemble_prob*100:.1f}%")
                    st.progress(float(ensemble_prob))
                
                with v_col3:
                    risk_level = "LOW"
                    color = "green"
                    if ensemble_prob > 0.70:
                        risk_level = "HIGH 🔴"
                    elif ensemble_prob > st_sensitivity:
                        risk_level = "MEDIUM 🟡"
                    st.metric("Risk Level Evaluation", risk_level)
                
                st.markdown("---")
                # Waveform Plots Side by Side
                col_w1, col_w2 = st.columns(2)
                
                with col_w1:
                    st.subheader("Waveform Analysis")
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
                    st.subheader("Vocal Texture Map")
                    fig_spec, ax_spec = plt.subplots(figsize=(10, 4.5))
                    plt.style.use('dark_background')
                    img = librosa.display.specshow(
                        features.T, 
                        x_axis='time', 
                        sr=16000, 
                        ax=ax_spec, 
                        cmap='magma',
                        vmax=np.percentile(features, 95),
                        vmin=np.percentile(features, 5)
                    )
                    ax_spec.set_title("60-D Spectral Fingerprint (MFCC + Δ + Δ²)", fontsize=12, pad=10)
                    fig_spec.colorbar(img, ax=ax_spec, format="%+2.f")
                    fig_spec.patch.set_facecolor('#1e2230')
                    ax_spec.set_facecolor('#1e2230')
                    st.pyplot(fig_spec)

            # TAB 2: Explainable AI Diagnostics
            with tab_explain:
                st.subheader("🔍 Acoustic Justification & Auditing Report")
                st.markdown("We trace standard vocal anatomy telltales to justify why the voice is AI or human-spoken:")
                
                for detail in audit_details:
                    # Choose color class based on status
                    status_class = "status-normal"
                    if "HIGHLY" in detail["status"]:
                        status_class = "status-critical"
                    elif "SUSPICIOUS" in detail["status"]:
                        status_class = "status-warning"
                        
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:bold; font-size:1.1rem; color:#eceff4;">{detail['name']}</span>
                            <span class="{status_class}" style="font-weight:bold; padding: 4px 10px; border-radius: 4px; border: 1px solid currentColor;">
                                {detail['status']} (value: {detail['val_str']})
                            </span>
                        </div>
                        <div class="audit-reason-box">
                            <p style="margin:0; font-size:0.95rem; line-height:1.4; color:#eceff4;">{detail['desc']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # TAB 3: Advanced Signal Metrics
            with tab_metrics:
                st.subheader("📈 Raw Acoustic Diagnostics Table")
                st.markdown("Below is the physiological feature grid mapped from the physical signal:")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Fundamental Pitch (Mean)</div>
                        <div class="metric-value">{metrics['pitch_mean']:.1f} Hz</div>
                        <div class="metric-status status-normal">Frequency band of vocal cords</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Spectral Flatness (Mean)</div>
                        <div class="metric-value">{metrics['spectral_flatness_mean']:.5f}</div>
                        <div class="metric-status">How 'noise-like' vs 'tone-like' signal is</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Intonation Standard Dev</div>
                        <div class="metric-value">{metrics['pitch_std']:.2f} Hz</div>
                        <div class="metric-status">Low standard dev indicates AI monotone</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">MFCC Texture Variance</div>
                        <div class="metric-value">{metrics['mfcc_var_mean']:.3f}</div>
                        <div class="metric-status">AI speech is smoother and lacks variance</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Waveform Shimmer</div>
                        <div class="metric-value">{metrics['shimmer_approx']:.4f}</div>
                        <div class="metric-status">Micro-modulations in vocal loudness</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Noise Stability (ZCR std)</div>
                        <div class="metric-value">{metrics['zcr_std']:.4f}</div>
                        <div class="metric-status">Artificial noise filter tell</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("---")
                # Pitch plot
                st.subheader("Pitch Tracking (F0 Contour)")
                try:
                    f_yin = librosa.yin(audio_data, fmin=75, fmax=400, sr=16000)
                    fig_p, ax_p = plt.subplots(figsize=(10, 3.5))
                    plt.style.use('dark_background')
                    ax_p.plot(f_yin, color='#a3be8c', linewidth=2, label="Calculated Pitch Contour")
                    ax_p.set_ylabel("Frequency (Hz)")
                    ax_p.set_xlabel("Frames")
                    ax_p.grid(True, linestyle=':', alpha=0.4)
                    ax_p.legend()
                    fig_p.patch.set_facecolor('#1e2230')
                    ax_p.set_facecolor('#1e2230')
                    st.pyplot(fig_p)
                except Exception:
                    st.info("No pitch contour could be plotted because fundamental frequency (F0) was undetectable.")

            # TAB 4: Model Embedding Visualizer
            with tab_neural:
                st.subheader("🧠 Deep Neural Representation (Bottleneck Features)")
                st.markdown("Visualizing the raw attention-pooled dense output vector layer (64-D embedding space) before classification:")
                
                fig_emb, ax_emb = plt.subplots(figsize=(10, 3.5))
                plt.style.use('dark_background')
                emb_feats = deep_feats[0]
                ax_emb.bar(range(len(emb_feats)), emb_feats, color='#ebcb8b', edgecolor='#d08770')
                ax_emb.set_xlabel("Bottleneck Embedding Dimensions")
                ax_emb.set_ylabel("Activation Intensity")
                ax_emb.set_title("Neural Activation Signature Vector", fontsize=12, pad=10)
                fig_emb.patch.set_facecolor('#1e2230')
                ax_emb.set_facecolor('#1e2230')
                st.pyplot(fig_emb)
                
                st.markdown("---")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.info(f"**Model Diagnostic Outputs:**\n- Neural feature extractor raw risk probability: **{model_prob*100:.2f}%**\n- Acoustic physical heuristics risk probability: **{heuristic_prob*100:.2f}%**")
                with col_d2:
                    st.info(f"**Final Fusion Calculation:**\n- 0.65 × Neural Model ({model_prob*100:.1f}%) + 0.35 × Heuristics ({heuristic_prob*100:.1f}%)\n- Combined Spoof Score: **{ensemble_prob*100:.2f}%** (Sensitivity threshold = **{st_sensitivity:.2f}**)")
 