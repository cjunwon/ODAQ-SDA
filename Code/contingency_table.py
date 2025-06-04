import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import pickle as pkl

with open('../Results/dense_rank_distance_matrix_experts_BSU1_BSU2_df.pkl', 'rb') as f:
    dense_rank_distance_matrix_experts_BSU1_BSU2_df = pkl.load(f)
with open('../Results/kmeans_rank_distance_matrix_experts_BSU1_BSU2_df.pkl', 'rb') as f:
    kmeans_rank_distance_matrix_experts_BSU1_BSU2_df = pkl.load(f)

# Cluster Assignments

# Clusters for 30 trials

# Perform hierarchical clustering on columns (trials/audio samples) for both distance matrices
linkage_matrix_dense_audio = linkage(dense_rank_distance_matrix_experts_BSU1_BSU2_df.T, method='ward')
linkage_matrix_kmeans_audio = linkage(kmeans_rank_distance_matrix_experts_BSU1_BSU2_df.T, method='ward')

# Extract 5 clusters from both hierarchical trees
num_clusters_trials = 5
dense_audio_clusters = fcluster(linkage_matrix_dense_audio, num_clusters_trials, criterion='maxclust')
kmeans_audio_clusters = fcluster(linkage_matrix_kmeans_audio, num_clusters_trials, criterion='maxclust')

# Create DataFrames mapping audio samples to their clusters
dense_audio_cluster_df = pd.DataFrame({
    'Audio Sample': dense_rank_distance_matrix_experts_BSU1_BSU2_df.columns,
    'Dense Cluster': dense_audio_clusters
})

kmeans_audio_cluster_df = pd.DataFrame({
    'Audio Sample': kmeans_rank_distance_matrix_experts_BSU1_BSU2_df.columns,
    'KMeans Cluster': kmeans_audio_clusters
})

# Merge to align the clusters from both methods
merged_audio_clusters = dense_audio_cluster_df.merge(kmeans_audio_cluster_df, on="Audio Sample")

# Build a contingency table comparing the two clustering results
contingency_table_trials = pd.crosstab(merged_audio_clusters['Dense Cluster'], merged_audio_clusters['KMeans Cluster'])

# Display the contingency table
print(contingency_table_trials)

# Initialize contingency table for trials/audio samples with lists
contingency_table_trials_specified = pd.DataFrame(
    [[[] for _ in range(5)] for _ in range(5)],  # Adjust for 5 clusters
    columns=[1, 2, 3, 4, 5], 
    index=[1, 2, 3, 4, 5]
)
contingency_table_trials_specified.index.name = 'Dense Cluster'
contingency_table_trials_specified.columns.name = 'KMeans Cluster'

# Define a function to retrieve audio sample lists
def get_audio_samples(dense_label, kmeans_label):
    return merged_audio_clusters[
        (merged_audio_clusters['Dense Cluster'] == dense_label) & 
        (merged_audio_clusters['KMeans Cluster'] == kmeans_label)
    ]['Audio Sample'].tolist()

# Populate the contingency table with actual audio sample lists
for dense in [1, 2, 3, 4, 5]:
    for kmeans in [1, 2, 3, 4, 5]:
        contingency_table_trials_specified.at[dense, kmeans] = get_audio_samples(dense, kmeans)

print(contingency_table_trials_specified)

# Export the contingency table to a csv file
contingency_table_trials_specified.to_csv('../Results/contingency_table_trials_specified.csv')

# Clusters for 42 Subjects

# Perform hierarchical clustering on rows (subjects) for both distance matrices
linkage_matrix_dense_subjects = linkage(dense_rank_distance_matrix_experts_BSU1_BSU2_df, method='ward')
linkage_matrix_kmeans_subjects = linkage(kmeans_rank_distance_matrix_experts_BSU1_BSU2_df, method='ward')

# Extract 3 clusters from both hierarchical trees
num_clusters_subjects = 3
dense_subject_clusters = fcluster(linkage_matrix_dense_subjects, num_clusters_subjects, criterion='maxclust')
kmeans_subject_clusters = fcluster(linkage_matrix_kmeans_subjects, num_clusters_subjects, criterion='maxclust')

# Create DataFrames mapping subjects to their clusters
dense_subject_cluster_df = pd.DataFrame({
    'Subject': dense_rank_distance_matrix_experts_BSU1_BSU2_df.index,
    'Dense Cluster': dense_subject_clusters
})

kmeans_subject_cluster_df = pd.DataFrame({
    'Subject': kmeans_rank_distance_matrix_experts_BSU1_BSU2_df.index,
    'KMeans Cluster': kmeans_subject_clusters
})

# Merge to align clusters from both methods
merged_subject_clusters = dense_subject_cluster_df.merge(kmeans_subject_cluster_df, on="Subject")

# Build a contingency table for subjects
contingency_table_subjects = pd.crosstab(merged_subject_clusters['Dense Cluster'], merged_subject_clusters['KMeans Cluster'])

# Display the contingency table
print(contingency_table_subjects)

contingency_table_subjects_specified = pd.DataFrame(
    [[[] for _ in range(3)] for _ in range(3)],
    columns=[1, 2, 3], 
    index=[1, 2, 3]
)
contingency_table_subjects_specified.index.name = 'Dense Cluster'
contingency_table_subjects_specified.columns.name = 'KMeans Cluster'

def get_subjects(dense_label, kmeans_label):
    return merged_subject_clusters[
        (merged_subject_clusters['Dense Cluster'] == dense_label) & 
        (merged_subject_clusters['KMeans Cluster'] == kmeans_label)
    ]['Subject'].tolist()
1
# Populate the contingency table with actual subject lists
for dense in [1, 2, 3]:
    for kmeans in [1, 2, 3]:
        contingency_table_subjects_specified.at[dense, kmeans] = get_subjects(dense, kmeans)

print(contingency_table_subjects_specified)

# Export the contingency table to a csv file
contingency_table_subjects_specified.to_csv('../Results/contingency_table_subjects_specified.csv')