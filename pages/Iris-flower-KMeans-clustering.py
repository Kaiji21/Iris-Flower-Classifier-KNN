import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris = datasets.load_iris()
data = iris.data

# Convert to DataFrame for easier handling
df = pd.DataFrame(data, columns=iris.feature_names)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)
df['Cluster'] = kmeans.labels_

# Use PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

st.title("Iris Flower K-Means Clustering")

col1,col2= st.columns(2)
col1.image("iris_flower.jpg", caption="Iris Flower")


col2.write("### A simple web application built with Streamlit that applies the K-Means clustering algorithm to group Iris flowers into three clusters based on their characteristics.")

# Streamlit interface
st.caption("K-Means clustering applied to the Iris dataset to group the flowers into three distinct clusters based on their features.")

# Add PCA1, PCA2, and cluster results into a new DataFrame for Streamlit scatter chart
chart_data = pd.DataFrame({
    'PCA1': df['PCA1'],
    'PCA2': df['PCA2'],
    'Cluster': df['Cluster'].astype(str)  # Convert clusters to strings for coloring
})

# Display scatter chart using Streamlit's native scatter_chart
st.scatter_chart(
    chart_data,
    x='PCA1',
    y='PCA2',
    color='Cluster'  # This will color the points based on the cluster
)


st.markdown(
    """
    <div style="text-align: center; font-size: 9px; color: gray;">
        <p>Created by ELHAIBA</p>
    </div>
    """,
    unsafe_allow_html=True
)