import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from kneed import KneeLocator  
from sklearn.datasets import load_iris

class IntImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill_value_ = round(np.nanmean(X))
        return self
    
    def transform(self, X):
        return np.nan_to_num(X, nan=self.fill_value_)

def get_preprocessing_pipeline(df):
    float_cols = df.select_dtypes(include=['float64']).columns.tolist()
    int_cols = df.select_dtypes(include=['int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('float', Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', RobustScaler())]), float_cols),
        ('int', Pipeline([('imputer', IntImputer()), ('scaler', RobustScaler())]), int_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])
    return preprocessor

def get_optimal_k(X, k_range=(2, 11)):
    inertia = []
    k_list = range(k_range[0], k_range[1])
    
    for k in k_list:
        model = KMeans(n_clusters=k, random_state=42, n_init=15)
        model.fit(X)
        inertia.append(model.inertia_)

    knee = KneeLocator(k_list, inertia, curve="convex", direction="decreasing")
    optimal_k = knee.elbow
    return optimal_k, inertia, k_list

def show_clusters(X, labels, centers, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=100, edgecolor='k', alpha=0.85)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='o', s=120, label='Centroids')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def run_kmeans_pipeline(df):
    print("Splitting the dataset...")
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    
    print("Preprocessing the data...")
    preprocessor = get_preprocessing_pipeline(df)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Removing outliers...")
    outlier_model = IsolationForest(contamination=0.1, random_state=42)
    outliers = outlier_model.fit_predict(X_train_processed)
    X_train_processed = X_train_processed[outliers == 1]
    
    print("Finding optimal number of clusters...")
    k, inertia, k_list = get_optimal_k(X_train_processed)
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_list, inertia, marker='o')
    plt.axvline(k, color='red', linestyle='--', label=f'Optimal K = {k}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Optimal number of clusters selected: {k}")
    
    print("Applying KMeans...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = kmeans.fit_predict(X_train_processed)
    centers = kmeans.cluster_centers_
    
    print("Visualizing clusters...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_processed)
    centers_pca = pca.transform(centers)
    
    show_clusters(X_pca, labels, centers_pca, "KMeans Clustering (PCA Reduced)")

    print("Evaluating clustering...")
    score = silhouette_score(X_train_processed, labels)
    
    plt.figure(figsize=(8, 6))
    plt.bar(['KMeans'], [score], color='blue')
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Evaluation')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()
    
    print(f"Optimal K = {k}")
    print(f"Silhouette Score = {score:.3f}")

iris = load_iris()
df = pd.DataFrame(iris.data)

run_kmeans_pipeline(df)