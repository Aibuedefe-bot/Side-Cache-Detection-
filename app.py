import pandas as pd
pd.options.future.infer_string = False  # Disable pyarrow-backed strings (fixes LargeUtf8)

import streamlit as st
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Cache Side-Channel Detector", layout="wide")
st.title("🛡️ Cache Side-Channel Attack Detector")
st.markdown("**Feature Engineering + ML Detector** (your full project demo)")

# Load model
model = joblib.load('cache_detector_model.pkl')

# Upload or use default
uploaded = st.file_uploader("Upload new perf data (or use the one we built)", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv('small_dataset.csv')

# Convert any pyarrow-backed string columns to plain object dtype (fixes LargeUtf8 error)
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].astype(str)

st.success("✅ Using dataset with " + str(len(df)) + " samples")

# Real-time simulation
st.subheader("Live Detection Dashboard")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Samples", len(df))
    st.metric("Detected Attacks", int(df['label'].sum()) if 'label' in df.columns else 0)

with col2:
    if st.button("🚨 Run Detection on Whole Dataset"):
        expected_features = list(model.feature_names_in_)
        X = df.drop(columns=['label', 'program', 'time'], errors='ignore')
        X = X[expected_features]
        predictions = model.predict(X)
        df['prediction'] = predictions
        df['attack_detected'] = ["ATTACK" if p == 1 else "Normal" for p in predictions]

        # Build display dataframe with plain python types to avoid LargeUtf8
        display_df = pd.DataFrame({
            'time': df['time'].head(50).tolist(),
            'prediction': df['prediction'].head(50).tolist(),
            'status': df['attack_detected'][:50],
        })
        st.dataframe(display_df)

        # Summary
        n_attacks = int((predictions == 1).sum())
        n_normal = int((predictions == 0).sum())
        st.info(f"Results: {n_attacks} attacks detected, {n_normal} normal samples")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'].tolist(), y=df['LLC-load-misses'].tolist(),
                                 mode='lines', name='LLC Misses', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df['time'].tolist(), y=(df['prediction'] * 100000).tolist(),
                                 mode='lines', name='Attack Prediction', line=dict(color='red')))
        st.plotly_chart(fig, use_container_width=True)

st.caption("Your full project: Feature Engineering → Selection → Random Forest → Real-time Dashboard")