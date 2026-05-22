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



# UI Setup

st.set_page_config(page_title="Deepfake Auditor V4", page_icon="🛡️", layout="wide")

st.title("🛡️ Deepfake Audio Auditor: Specialist V4")

st.markdown("---")



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



# 3. Enhanced Feature Extractor

def extract_60d_features(audio_data):

    # Pre-emphasis helps separate the 'vocal ridges' from the 'yellow' background noise

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



# 4. Main App Interface

uploaded_file = st.file_uploader("Upload Audio Sample", type=['flac', 'mp3', 'wav'])



if uploaded_file:

    audio_data, sr = librosa.load(uploaded_file, sr=16000)

    

    col1, col2 = st.columns(2)

    

    with col1:

        st.subheader("Waveform Analysis")

        st.audio(uploaded_file)

        fig_wav, ax_wav = plt.subplots(figsize=(10, 4))

        librosa.display.waveshow(audio_data, sr=16000, ax=ax_wav, color='#2ca02c')

        ax_wav.set_title("Amplitude vs. Time")

        st.pyplot(fig_wav)



    if st.button("🚀 EXECUTE MULTI-DOMAIN AUDIT"):

        features = extract_60d_features(audio_data)

        

        with col2:

            st.subheader("Vocal Texture Map")

            fig_spec, ax_spec = plt.subplots(figsize=(10, 4))

            

            # THE FIX: Using 'magma' with dynamic range scaling to stop the 'yellow' block

            # We use robust scaling based on percentiles to ensure contrast

            img = librosa.display.specshow(

                features.T, 

                x_axis='time', 

                sr=16000, 

                ax=ax_spec, 

                cmap='magma',

                vmax=np.percentile(features, 95), # Ignores outliers that cause yellowing

                vmin=np.percentile(features, 5)

            )

            ax_spec.set_title("60-D Spectral Fingerprint")

            fig_spec.colorbar(img, ax=ax_spec, format="%+2.f")

            st.pyplot(fig_spec)

        

        # Predictions

        deep_feats = feat_ext.predict(np.expand_dims(features, axis=0), verbose=0)

        prob = xgb.predict_proba(deep_feats)[0]

        

        st.markdown("---")

        res_c1, res_c2 = st.columns(2)

        with res_c1:

            if prob[1] > 0.5:

                st.error("🚨 **SPOOFED (DEEPFAKE)**")

            else:

                st.success("✅ **BONAFIDE (HUMAN)**")

        with res_c2:

            st.metric("Detection Confidence", f"{prob[1]*100 if prob[1] > 0.5 else prob[0]*100:.2f}%")

            st.progress(float(prob[1]))
