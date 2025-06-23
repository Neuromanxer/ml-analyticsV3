from preprocessing import preprocess_data
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
from preprocessing import preprocess_data
# Find optimal k using both elbow method and silhouette score
def find_optimal_k(data, max_k=10):
    """ Finds the best K using Elbow & Silhouette method """
    # Ensure max_k is valid
    max_k = min(max_k, len(data) - 1)
    if max_k < 2:
        return 2, [], []
    
    wcss = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        wcss.append(kmeans.inertia_)
        
        # Only calculate silhouette score if we have enough samples
        if len(data) > k + 1:
            silhouette_scores.append(silhouette_score(data, labels))
    
    # Find best k (default to 3 if calculation fails)
    best_k = 3
    if silhouette_scores and len(silhouette_scores) > 1:
        best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    
    return best_k, wcss, silhouette_scores

# Function to run KMeans clustering
def run_kmeans(scaled_data, best_k=3):
    """ Runs KMeans clustering on dataset """
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters, kmeans

# Function to label clusters
def label_clusters_general(df, cluster_col='cluster', feature_columns=None):
    """ Assigns meaningful labels to customer segments based on key features """
    if feature_columns is None or len(feature_columns) == 0:
        # Default to first two numeric columns if none specified
        feature_columns = df.select_dtypes(include=['number']).columns[:2].tolist()
    
    # Ensure we have at least one feature column
    if not feature_columns or len(feature_columns) == 0:
        return df
    
    # Use the first two features at most
    feature_columns = feature_columns[:min(2, len(feature_columns))]
    
    # Group by cluster and calculate mean for each feature
    cluster_profiles = df.groupby(cluster_col)[feature_columns].mean().reset_index()
    
    # Calculate medians for each feature
    medians = {col: df[col].median() for col in feature_columns}
    
    # Generate segment names based on key features
    segment_mapping = {}
    for _, row in cluster_profiles.iterrows():
        cluster_id = row[cluster_col]
        feature_descriptions = []
        
        for feature in feature_columns:
            if row[feature] > medians[feature]:
                feature_descriptions.append(f"High {feature.replace('_', ' ')}")
            else:
                feature_descriptions.append(f"Low {feature.replace('_', ' ')}")
        
        segment_mapping[cluster_id] = " & ".join(feature_descriptions)
    
    # Apply segment names to original dataframe
    df['segment_name'] = df[cluster_col].map(segment_mapping)
    
    return df