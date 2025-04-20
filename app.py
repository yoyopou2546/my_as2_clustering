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
import numpy as np

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

# Set number of clusters to 4 by default
num_clusters = 4

# Generate synthetic data
X, _ = make_blobs(
    n_samples=300,
    centers=num_clusters,  # Fixed number of centers
    cluster_std=0.60,
    random_state=0
)

# Predict cluster labels
y_kmeans = loaded_model.predict(X)

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.title(f'K-Means Clustering (k={num_clusters})')

# Display cluster centers with red circles
centers = loaded_model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='o', label="Centroids")  # Changed marker to 'o'

plt.legend()
st.pyplot(plt)
