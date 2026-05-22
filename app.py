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

# 1. Custom Attention Layer (Preserved exactly from V4)
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

# UI Setup - Main Dashboard Optimization (No Sidebar)
st.set_page_config(page_title="Deepfake Auditor V4", page_icon="🛡️", layout="wide")

# Main Application Headers
st.title("🎙️ Multi-Layer Audio Deepfake Authentication Dashboard")
st.write("Extract 60-D dynamic spectral fingerprints to validate biological speech vs. synthetic anomalies.")

# --- MAIN PERFORMANCE OVERVIEW PANEL (Replaced the Sidebar) ---
st.markdown("### 📊 System Empirical Validation Metrics")
met_c1, met_c2, met_c3, met_c4 = st.columns(4)
with met_c1:
    st.metric(label="Overall Testing Accuracy", value="95.0%")
with met_c2:
    st.metric(label="Area Under Curve (ROC-AUC)", value="0.9848")
with met_c3:
    st.metric(label="Balanced Human F1-Score", value="0.95")
with met_c4:
    st.metric(label="Balanced AI F1-Score", value="0.95")

st.caption("Engine Specifications: V4-Robust | Architecture: Hybrid CNN-BiLSTM + Self Attention with Stacking XGBoost Refiner")
st.markdown("---")

# 2. Load Models Safely
@st.cache_resource
def load_models():
    brain = tf.keras.models.load_model('enhanced_attention_model_v4_robust.keras', 
                                       custom_objects={'SelfAttention': SelfAttention}, 
                                       compile=False)
    feat_extractor = tf.keras.Model(inputs=brain.input, outputs=brain.layers[-3].output)
    xgb_judge = joblib.load('xgboost_refiner_v4_robust.pkl')
    return feat_extractor, xgb_judge

feat_ext, xgb = load_models()

# 3. Enhanced Feature Extractor (Preserved from V4)
def extract_60d_features(audio_data):
    audio = librosa.effects.preemphasis(audio_data)
    audio = librosa.util.normalize(audio)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    combined = np.concatenate([mfcc, delta, delta2], axis=0)
    
    if combined.shape[1] < 200:
        combined = np.pad(combined, ((0,0), (0, 200 -
