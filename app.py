# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:06:42 2025

@author: yoyop
"""

# app.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Load the KMeans model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Set Streamlit page configuration
st.set_page_config(page_title="K-Means Clustering App", layout="centered")

# Title
st.title("🔍 k-Means Clustering Visualizer")

# Display section header
st.subheader("📊 Example Data for Visualization")
st.markdown("This demo uses example 2D data to illustrate clustering results.")

# Generate synthetic data
X, _ = make_blobs(
    n_samples=300,
    centers=loaded_model.n_clusters,
    cluster_std=0.60,
    random_state=0
)

# Predict cluster labels
y_kmeans = loaded_model.predict(X)
