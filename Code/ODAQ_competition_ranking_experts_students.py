import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import pickle as pkl

ODAQ_results = pd.read_csv('../Data/./ODAQ/ODAQ_listening_test/ODAQ_results.csv')
ODAQ_results_BSU1 = pd.read_csv('../Data/./ODAQ_v1_BSU/Cohort_B1_results.csv')
ODAQ_results_BSU2 = pd.read_csv('../Data/./ODAQ_v1_BSU/Cohort_B2_results.csv')

methods = ODAQ_results['method'].unique()
conditions = ODAQ_results['condition'].unique()
processes = ODAQ_results['process'].unique()
items = ODAQ_results['item'].unique()

print(methods)
print(conditions)
print(processes)
print(items)

# Dynamically create expert variables
unique_subjects = ODAQ_results['subject'].unique()
for i, subject in enumerate(unique_subjects, start=1):
    globals()[f"expert{i}"] = ODAQ_results[ODAQ_results['subject'] == subject]

# Dynamically create BSU1 variables
unique_subjects_BSU1 = ODAQ_results_BSU1['subject'].unique()
for i, subject in enumerate(unique_subjects_BSU1, start=1):
    globals()[f"BSU1_{i}"] = ODAQ_results_BSU1[ODAQ_results_BSU1['subject'] == subject]

# Dynamically create BSU2 variables
unique_subjects_BSU2 = ODAQ_results_BSU2['subject'].unique()
for i, subject in enumerate(unique_subjects_BSU2, start=1):
    globals()[f"BSU2_{i}"] = ODAQ_results_BSU2[ODAQ_results_BSU2['subject'] == subject]
    
print('Experts: ', unique_subjects)
print('BSU1: ', unique_subjects_BSU1)
print('BSU2: ', unique_subjects_BSU2)

# Initialize score lists dynamically for 26 experts
for i in range(1, 27):  # Assuming 26 experts
    globals()[f"expert{i}_scores"] = []

# Initialize score lists dynamically for BSU1
for i in range(1, 9):  # Assuming 26 experts
    globals()[f"BSU1_{i}_scores"] = []

# Initialize score lists dynamically for BSU2
for i in range(1, 9):  # Assuming 26 experts
    globals()[f"BSU2_{i}_scores"] = []

# Append scores systematically
for item in items:
    for i in range(1, 27):
        expert_df = globals()[f"expert{i}"]  # Access expert data frame
        scores = expert_df[expert_df['item'] == item]['score'].values
        globals()[f"expert{i}_scores"].append(scores)

    for i in range(1, 9):
        BSU1_df = globals()[f"BSU1_{i}"]
        scores = BSU1_df[BSU1_df['item'] == item]['score'].values
        globals()[f"BSU1_{i}_scores"].append(scores)

    for i in range(1, 9):
        BSU2_df = globals()[f"BSU2_{i}"]
        scores = BSU2_df[BSU2_df['item'] == item]['score'].values
        globals()[f"BSU2_{i}_scores"].append(scores)
        
# Competition Ranking Assignment

# Initialize score lists dynamically for 26 experts
for i in range(1, 27):  # Assuming 26 experts
    globals()[f"expert{i}_scores"] = []

# Initialize score lists dynamically for BSU1
for i in range(1, 9):  # Assuming 26 experts
    globals()[f"BSU1_{i}_scores"] = []

# Initialize score lists dynamically for BSU2
for i in range(1, 9):  # Assuming 26 experts
    globals()[f"BSU2_{i}_scores"] = []

# Append scores systematically
for item in items:
    for i in range(1, 27):
        expert_df = globals()[f"expert{i}"]  # Access expert data frame
        scores = expert_df[expert_df['item'] == item]['score'].values
        globals()[f"expert{i}_scores"].append(scores)

    for i in range(1, 9):
        BSU1_df = globals()[f"BSU1_{i}"]
        scores = BSU1_df[BSU1_df['item'] == item]['score'].values
        globals()[f"BSU1_{i}_scores"].append(scores)

    for i in range(1, 9):
        BSU2_df = globals()[f"BSU2_{i}"]
        scores = BSU2_df[BSU2_df['item'] == item]['score'].values
        globals()[f"BSU2_{i}_scores"].append(scores)

# Compute rankings systematically for 26 experts
for i in range(1, 27):  # Assuming 26 experts
    expert_scores = globals()[f"expert{i}_scores"]  # Get the score list
    globals()[f"expert{i}_rankings"] = np.array([competition_ranking(row) for row in expert_scores])

# Compute rankings systematically for BSU1
for i in range(1, 9):  # Assuming 8 BSU1 students
    BSU1_scores = globals()[f"BSU1_{i}_scores"]  # Get the score list
    globals()[f"BSU1_{i}_rankings"] = np.array([competition_ranking(row) for row in BSU1_scores])

# Compute rankings systematically for BSU2
for i in range(1, 9):  # Assuming 8 BSU2 students
    BSU2_scores = globals()[f"BSU2_{i}_scores"]  # Get the score list
    globals()[f"BSU2_{i}_rankings"] = np.array([competition_ranking(row) for row in BSU2_scores])

# Heatmap & Hierarchical Clustering

# Perfect ranking
perfect_ranking = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Define a distance function (Euclidean distance)
def compute_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)  # Euclidean distance

# Initialize a 26x30 matrix to store distances
distance_matrix_experts = np.zeros((26, 30))

# Compute distances systematically
for i in range(1, 27):  # 26 experts
    expert_rankings = globals()[f"expert{i}_rankings"]  # Get expert rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per expert
        distance_matrix_experts[i-1, j] = compute_distance(expert_rankings[j], perfect_ranking)

distance_matrix_experts_df = pd.DataFrame(distance_matrix_experts, columns=items)

# Increment index by 1 for distance_matrix_df

distance_matrix_experts_df.index += 1

# --------------------------------------------------------------------------------------------------------

# Initialize a 34x30 matrix to store distances for Experts + BSU1
distance_matrix_experts_BSU1 = np.zeros((34, 30))

# Compute distances systematically
for i in range(1, 27):  # 26 experts
    expert_rankings = globals()[f"expert{i}_rankings"]  # Get expert rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per expert
        distance_matrix_experts_BSU1[i-1, j] = compute_distance(expert_rankings[j], perfect_ranking)

for i in range(1, 9):  # 8 BSU1 students
    BSU1_rankings = globals()[f"BSU1_{i}_rankings"]  # Get BSU1 rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per BSU1 student
        distance_matrix_experts_BSU1[26+i-1, j] = compute_distance(BSU1_rankings[j], perfect_ranking)

distance_matrix_experts_BSU1_df = pd.DataFrame(distance_matrix_experts_BSU1, columns=items)

# Increment index by 1 for distance_matrix_experts_BSU1_df
distance_matrix_experts_BSU1_df.index += 1

# --------------------------------------------------------------------------------------------------------

# Initialize a 42x30 matrix to store distances for Experts + BSU1 + BSU2
distance_matrix_experts_BSU1_BSU2 = np.zeros((42, 30))

# Compute distances systematically
for i in range(1, 27):  # 26 experts
    expert_rankings = globals()[f"expert{i}_rankings"]  # Get expert rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per expert
        distance_matrix_experts_BSU1_BSU2[i-1, j] = compute_distance(expert_rankings[j], perfect_ranking)

for i in range(1, 9):  # 8 BSU1 students
    BSU1_rankings = globals()[f"BSU1_{i}_rankings"]  # Get BSU1 rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per BSU1 student
        distance_matrix_experts_BSU1_BSU2[26+i-1, j] = compute_distance(BSU1_rankings[j], perfect_ranking)

for i in range(1, 9):  # 8 BSU2 students
    BSU2_rankings = globals()[f"BSU2_{i}_rankings"]  # Get BSU2 rankings (30 vectors)

    for j in range(30):  # 30 ranking vectors per BSU2 student
        distance_matrix_experts_BSU1_BSU2[34+i-1, j] = compute_distance(BSU2_rankings[j], perfect_ranking)

distance_matrix_experts_BSU1_BSU2_df = pd.DataFrame(distance_matrix_experts_BSU1_BSU2, columns=items)

# Increment index by 1 for distance_matrix_experts_BSU1_BSU2_df
distance_matrix_experts_BSU1_BSU2_df.index += 1

# Create a heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(distance_matrix_experts_df, cmap="coolwarm", annot=False, linewidths=0.5)

plt.xlabel("Audio Sample")
plt.ylabel("Experts")
plt.title("Heatmap of Distance Between Expert Competition Rankings and Perfect Ranking")

for label in ax.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')
    elif text.isdigit() and 35 <= int(text) <= 42:
        label.set_color('orange')


plt.show()

# Perform hierarchical clustering (using Ward's method)
linkage_matrix_experts = linkage(distance_matrix_experts, method='ward')

# Create a clustermap (heatmap with hierarchical clustering)
clustermap_experts = sns.clustermap(
    distance_matrix_experts_df,
    cmap="coolwarm",
    method="ward",
    figsize=(12, 12),
    xticklabels=True,  # Display column labels (optional)
    yticklabels=True   # Display row labels (optional)
)

# Add axis labels
clustermap_experts.ax_heatmap.set_xlabel("Audio Samples", fontsize=12)
clustermap_experts.ax_heatmap.set_ylabel("Experts", fontsize=12)
clustermap_experts.ax_heatmap.set_title("Clustermap of Expert Competition Ranking Distances from Perfect Ranking", fontsize=14, pad=120)

# Show the plot
plt.show()

# PERFORMANCE-BASED CLUSTERING

# Extract clusters from the linkage matrix
num_clusters = 5  # Choose the number of clusters (you can adjust)
cluster_labels_experts = fcluster(linkage_matrix_experts, num_clusters, criterion='maxclust')

# Create a DataFrame mapping experts to their cluster
cluster_experts_df = pd.DataFrame({'Expert': [f"Expert {i}" for i in range(1, 27)], 
                           'Cluster': cluster_labels_experts})


# order cluster_df by Cluster
cluster_experts_df_ordered = cluster_experts_df.sort_values(by='Cluster')

# Create a heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(distance_matrix_experts_BSU1_df, cmap="coolwarm", annot=False, linewidths=0.5)

plt.xlabel("Audio Sample")
plt.ylabel("Experts, Cohort 1 Students")
plt.title("Heatmap of Distance Between Expert and Student (Cohort 1) Competition Rankings and Perfect Ranking")

for label in ax.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')
    elif text.isdigit() and 35 <= int(text) <= 42:
        label.set_color('orange')


plt.show()

# Perform hierarchical clustering (using Ward's method)
linkage_matrix_experts_BSU1 = linkage(distance_matrix_experts_BSU1_df, method='ward')

# Create a clustermap (heatmap with hierarchical clustering)
clustermap_experts_BSU1 = sns.clustermap(
    distance_matrix_experts_BSU1_df,
    cmap="coolwarm",
    method="ward",
    figsize=(12, 14),
    xticklabels=True,  # Display column labels (optional)
    yticklabels=True   # Display row labels (optional)
)

# Add axis labels
clustermap_experts_BSU1.ax_heatmap.set_xlabel("Audio Samples", fontsize=12)
clustermap_experts_BSU1.ax_heatmap.set_ylabel("Experts (Black), Cohort 1 Students (Green)", fontsize=12)
clustermap_experts_BSU1.ax_heatmap.set_title("Clustermap of Expert and Student (Cohort 1) Competition Ranking Distances from Perfect Ranking", fontsize=14, pad=120)

# Customize the row tick labels
for label in clustermap_experts_BSU1.ax_heatmap.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')

plt.show()

# Create a heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(distance_matrix_experts_BSU1_BSU2_df, cmap="coolwarm", annot=False, linewidths=0.5)

plt.xlabel("Audio Sample")
plt.ylabel("Experts, Cohort 1 Students, Cohort 2 Students")
plt.title("Heatmap of Distance Between Expert and Student (Cohort 1 & 2) Competition Rankings and Perfect Ranking")

for label in ax.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')
    elif text.isdigit() and 35 <= int(text) <= 42:
        label.set_color('orange')

plt.show()

# Perform hierarchical clustering (using Ward's method)
linkage_matrix_experts_BSU1_BSU2 = linkage(distance_matrix_experts_BSU1_BSU2_df, method='ward')

# Create a clustermap (heatmap with hierarchical clustering)
clustermap_experts_BSU1_BSU2 = sns.clustermap(
    distance_matrix_experts_BSU1_BSU2_df,
    cmap="coolwarm",
    method="ward",
    row_cluster=True,  # Ensure row clustering
    col_cluster=True,  # Ensure column clustering
    figsize=(12, 14),
    xticklabels=True,  # Display column labels (optional)
    yticklabels=True   # Display row labels (optional)
)

# Add axis labels
clustermap_experts_BSU1_BSU2.ax_heatmap.set_xlabel("Audio Samples", fontsize=12)
clustermap_experts_BSU1_BSU2.ax_heatmap.set_ylabel("Experts (Black), Cohort 1 Students (Green), Cohort 1 Students (Orange)", fontsize=12)
clustermap_experts_BSU1_BSU2.ax_heatmap.set_title("Clustermap of Expert and Student (Cohort 1 & 2) Competition Ranking Distances from Perfect Ranking", fontsize=14, pad=120)

# Customize the row tick labels
for label in clustermap_experts_BSU1_BSU2.ax_heatmap.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')
    elif text.isdigit() and 35 <= int(text) <= 42:
        label.set_color('orange')



row_order = clustermap_experts_BSU1_BSU2.dendrogram_row.reordered_ind
col_order = clustermap_experts_BSU1_BSU2.dendrogram_col.reordered_ind

print("Row order:", row_order)
print("Column order:", col_order)

# Show the plot
plt.show()



# export linkage_matrix_experts_BSU1_BSU2 to a pickle file

with open('../Results/dense_rank_linkage_matrix_experts_BSU1_BSU2.pkl', 'wb') as f:
    pkl.dump(linkage_matrix_experts_BSU1_BSU2, f)
    

# export distance_matrix_experts_BSU1_BSU2_df to a pickle file

with open('../Results/dense_rank_distance_matrix_experts_BSU1_BSU2_df.pkl', 'wb') as f:
    pkl.dump(distance_matrix_experts_BSU1_BSU2_df, f)

# Perform hierarchical clustering on the columns (audio samples)
linkage_matrix_audio = linkage(distance_matrix_experts_BSU1_BSU2_df.T, method='ward')

# Extract 5 clusters from the hierarchical tree
num_clusters = 5
audio_sample_clusters = fcluster(linkage_matrix_audio, num_clusters, criterion='maxclust')

# Create a dictionary mapping audio samples to their clusters
audio_sample_cluster_dict = dict(zip(distance_matrix_experts_BSU1_BSU2_df.columns, audio_sample_clusters))

# Convert the dictionary to a DataFrame
audio_sample_cluster_df = pd.DataFrame(list(audio_sample_cluster_dict.items()), columns=['Audio Sample', 'Cluster'])

# New dataframe containing number of samples per cluster
cluster_counts = audio_sample_cluster_df['Cluster'].value_counts()

print(cluster_counts)

# Define a distance function (Euclidean distance)
def compute_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)  # Euclidean distance

# Initialize a 26x30 matrix to store distances
distance_matrix_experts = np.zeros((26, 30))

# Compute distances systematically
for i in range(1, 27):  # 26 experts
    expert_rankings = globals()[f"expert{i}_rankings"]  # Get expert rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per expert
        distance_matrix_experts[i-1, j] = compute_distance(expert_rankings[j], perfect_ranking)

distance_matrix_experts_df = pd.DataFrame(distance_matrix_experts, columns=items)

# Increment index by 1 for distance_matrix_df

distance_matrix_experts_df.index += 1

# --------------------------------------------------------------------------------------------------------

# Initialize a 34x30 matrix to store distances for Experts + BSU1
distance_matrix_experts_BSU1 = np.zeros((34, 30))

# Compute distances systematically
for i in range(1, 27):  # 26 experts
    expert_rankings = globals()[f"expert{i}_rankings"]  # Get expert rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per expert
        distance_matrix_experts_BSU1[i-1, j] = compute_distance(expert_rankings[j], perfect_ranking)

for i in range(1, 9):  # 8 BSU1 students
    BSU1_rankings = globals()[f"BSU1_{i}_rankings"]  # Get BSU1 rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per BSU1 student
        distance_matrix_experts_BSU1[26+i-1, j] = compute_distance(BSU1_rankings[j], perfect_ranking)

distance_matrix_experts_BSU1_df = pd.DataFrame(distance_matrix_experts_BSU1, columns=items)

# Increment index by 1 for distance_matrix_experts_BSU1_df
distance_matrix_experts_BSU1_df.index += 1

# --------------------------------------------------------------------------------------------------------

# Initialize a 42x30 matrix to store distances for Experts + BSU1 + BSU2
distance_matrix_experts_BSU1_BSU2 = np.zeros((42, 30))

# Compute distances systematically
for i in range(1, 27):  # 26 experts
    expert_rankings = globals()[f"expert{i}_rankings"]  # Get expert rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per expert
        distance_matrix_experts_BSU1_BSU2[i-1, j] = compute_distance(expert_rankings[j], perfect_ranking)

for i in range(1, 9):  # 8 BSU1 students
    BSU1_rankings = globals()[f"BSU1_{i}_rankings"]  # Get BSU1 rankings (30 vectors)
    
    for j in range(30):  # 30 ranking vectors per BSU1 student
        distance_matrix_experts_BSU1_BSU2[26+i-1, j] = compute_distance(BSU1_rankings[j], perfect_ranking)

for i in range(1, 9):  # 8 BSU2 students
    BSU2_rankings = globals()[f"BSU2_{i}_rankings"]  # Get BSU2 rankings (30 vectors)

    for j in range(30):  # 30 ranking vectors per BSU2 student
        distance_matrix_experts_BSU1_BSU2[34+i-1, j] = compute_distance(BSU2_rankings[j], perfect_ranking)

distance_matrix_experts_BSU1_BSU2_df = pd.DataFrame(distance_matrix_experts_BSU1_BSU2, columns=items)

# Increment index by 1 for distance_matrix_experts_BSU1_BSU2_df
distance_matrix_experts_BSU1_BSU2_df.index += 1
distance_matrix_experts_df
# Create a heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(distance_matrix_experts_df, cmap="coolwarm", annot=False, linewidths=0.5)

plt.xlabel("Audio Sample")
plt.ylabel("Experts")
plt.title("Heatmap of Distance Between Expert Competition Rankings and Perfect Ranking")

for label in ax.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')
    elif text.isdigit() and 35 <= int(text) <= 42:
        label.set_color('orange')


plt.show()
# Perform hierarchical clustering (using Ward's method)
linkage_matrix_experts = linkage(distance_matrix_experts, method='ward')

# Create a clustermap (heatmap with hierarchical clustering)
clustermap_experts = sns.clustermap(
    distance_matrix_experts_df,
    cmap="coolwarm",
    method="ward",
    figsize=(12, 12),
    xticklabels=True,  # Display column labels (optional)
    yticklabels=True   # Display row labels (optional)
)

# Add axis labels
clustermap_experts.ax_heatmap.set_xlabel("Audio Samples", fontsize=12)
clustermap_experts.ax_heatmap.set_ylabel("Experts", fontsize=12)
clustermap_experts.ax_heatmap.set_title("Clustermap of Expert Competition Ranking Distances from Perfect Ranking", fontsize=14, pad=120)

# Show the plot
plt.show()
# PERFORMANCE-BASED CLUSTERING

# Extract clusters from the linkage matrix
num_clusters = 5  # Choose the number of clusters (you can adjust)
cluster_labels_experts = fcluster(linkage_matrix_experts, num_clusters, criterion='maxclust')

# Create a DataFrame mapping experts to their cluster
cluster_experts_df = pd.DataFrame({'Expert': [f"Expert {i}" for i in range(1, 27)], 
                           'Cluster': cluster_labels_experts})

cluster_experts_df
# order cluster_df by Cluster

cluster_experts_df_ordered = cluster_experts_df.sort_values(by='Cluster')

cluster_experts_df_ordered
# Example: Find which experts belong to Cluster 1
cluster_1_experts = cluster_experts_df[cluster_experts_df['Cluster'] == 1]
print(cluster_1_experts)
# Create a heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(distance_matrix_experts_BSU1_df, cmap="coolwarm", annot=False, linewidths=0.5)

plt.xlabel("Audio Sample")
plt.ylabel("Experts, Cohort 1 Students")
plt.title("Heatmap of Distance Between Expert and Student (Cohort 1) Competition Rankings and Perfect Ranking")

for label in ax.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')
    elif text.isdigit() and 35 <= int(text) <= 42:
        label.set_color('orange')


plt.show()
# Perform hierarchical clustering (using Ward's method)
linkage_matrix_experts_BSU1 = linkage(distance_matrix_experts_BSU1_df, method='ward')

# Create a clustermap (heatmap with hierarchical clustering)
clustermap_experts_BSU1 = sns.clustermap(
    distance_matrix_experts_BSU1_df,
    cmap="coolwarm",
    method="ward",
    figsize=(12, 14),
    xticklabels=True,  # Display column labels (optional)
    yticklabels=True   # Display row labels (optional)
)

# Add axis labels
clustermap_experts_BSU1.ax_heatmap.set_xlabel("Audio Samples", fontsize=12)
clustermap_experts_BSU1.ax_heatmap.set_ylabel("Experts (Black), Cohort 1 Students (Green)", fontsize=12)
clustermap_experts_BSU1.ax_heatmap.set_title("Clustermap of Expert and Student (Cohort 1) Competition Ranking Distances from Perfect Ranking", fontsize=14, pad=120)

# Customize the row tick labels
for label in clustermap_experts_BSU1.ax_heatmap.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')

plt.show()
# Create a heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(distance_matrix_experts_BSU1_BSU2_df, cmap="coolwarm", annot=False, linewidths=0.5)

plt.xlabel("Audio Sample")
plt.ylabel("Experts, Cohort 1 Students, Cohort 2 Students")
plt.title("Heatmap of Distance Between Expert and Student (Cohort 1 & 2) Competition Rankings and Perfect Ranking")

for label in ax.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')
    elif text.isdigit() and 35 <= int(text) <= 42:
        label.set_color('orange')

plt.show()
# Perform hierarchical clustering (using Ward's method)
linkage_matrix_experts_BSU1_BSU2 = linkage(distance_matrix_experts_BSU1_BSU2_df, method='ward')

# Create a clustermap (heatmap with hierarchical clustering)
clustermap_experts_BSU1_BSU2 = sns.clustermap(
    distance_matrix_experts_BSU1_BSU2_df,
    cmap="coolwarm",
    method="ward",
    row_cluster=True,  # Ensure row clustering
    col_cluster=True,  # Ensure column clustering
    figsize=(12, 14),
    xticklabels=True,  # Display column labels (optional)
    yticklabels=True   # Display row labels (optional)
)

# Add axis labels
clustermap_experts_BSU1_BSU2.ax_heatmap.set_xlabel("Audio Samples", fontsize=12)
clustermap_experts_BSU1_BSU2.ax_heatmap.set_ylabel("Experts (Black), Cohort 1 Students (Green), Cohort 1 Students (Orange)", fontsize=12)
clustermap_experts_BSU1_BSU2.ax_heatmap.set_title("Clustermap of Expert and Student (Cohort 1 & 2) Competition Ranking Distances from Perfect Ranking", fontsize=14, pad=120)

# Customize the row tick labels
for label in clustermap_experts_BSU1_BSU2.ax_heatmap.get_yticklabels():
    text = label.get_text()
    if text.isdigit() and 27 <= int(text) <= 34:
        label.set_color('green')
    elif text.isdigit() and 35 <= int(text) <= 42:
        label.set_color('orange')



row_order = clustermap_experts_BSU1_BSU2.dendrogram_row.reordered_ind
col_order = clustermap_experts_BSU1_BSU2.dendrogram_col.reordered_ind

print("Row order:", row_order)
print("Column order:", col_order)

# Show the plot
plt.show()
import pickle as pkl

# export linkage_matrix_experts_BSU1_BSU2 to a pickle file

with open('../Results/dense_rank_linkage_matrix_experts_BSU1_BSU2.pkl', 'wb') as f:
    pkl.dump(linkage_matrix_experts_BSU1_BSU2, f)
    

# export distance_matrix_experts_BSU1_BSU2_df to a pickle file

with open('../Results/dense_rank_distance_matrix_experts_BSU1_BSU2_df.pkl', 'wb') as f:
    pkl.dump(distance_matrix_experts_BSU1_BSU2_df, f)
# Perform hierarchical clustering on the columns (audio samples)
linkage_matrix_audio = linkage(distance_matrix_experts_BSU1_BSU2_df.T, method='ward')

# Extract 5 clusters from the hierarchical tree
num_clusters = 5
audio_sample_clusters = fcluster(linkage_matrix_audio, num_clusters, criterion='maxclust')

# Create a dictionary mapping audio samples to their clusters
audio_sample_cluster_dict = dict(zip(distance_matrix_experts_BSU1_BSU2_df.columns, audio_sample_clusters))

# Convert the dictionary to a DataFrame
audio_sample_cluster_df = pd.DataFrame(list(audio_sample_cluster_dict.items()), columns=['Audio Sample', 'Cluster'])

# New dataframe containing number of samples per cluster
cluster_counts = audio_sample_cluster_df['Cluster'].value_counts()

cluster_counts
# Spaghetti Plots
# reversed perfect ranking
reversed_perfect_ranking = np.array([8, 7, 6, 5, 4, 3, 2, 1])

# Create DataFrame for perfect ranking
perfect_df = pd.DataFrame({
    'Sample': ['Perfect Ranking'] * len(conditions),
    'Condition': conditions,
    'Ranking': reversed_perfect_ranking
})

reversed_expert1_rankings = expert1_rankings[:, ::-1]
reversed_expert2_rankings = expert2_rankings[:, ::-1]
reversed_BSU1_1_rankings = BSU1_1_rankings[:, ::-1]
reversed_BSU2_1_rankings = BSU2_1_rankings[:, ::-1]


# Spaghetti plot for Expert 1 rankings

# Reshape rankings data for Plotly
expert1_rankings_df = pd.DataFrame(reversed_expert1_rankings, columns=conditions)
expert1_rankings_df['Sample'] = items

# Melt dataframe for better visualization
expert1_rankings_df_melted = expert1_rankings_df.melt(id_vars=['Sample'], var_name='Condition', value_name='Ranking')

expert1_rankings_df_melted = pd.concat([perfect_df, expert1_rankings_df_melted])

# Create plot
fig = px.line(expert1_rankings_df_melted, x='Condition', y='Ranking', color='Sample', markers=True,
              title="Expert 1 (Competition) Rankings per Condition with Perfect Ranking Reference",
              labels={"Ranking": "Ranking (Lower is Better)", "Condition": "Conditions (Low to High Quality)"},
              template="plotly_white")

# Adjust figure dimensions
fig.update_layout(width=1000, height=800)

# Invert y-axis (lower ranks at top)
fig.update_yaxes(autorange="reversed")

# Modify the "Perfect Ranking" line to be more visible
fig.update_traces(
    selector=dict(name="Perfect Ranking"),
    line=dict(width=10, color='black'),
    marker=dict(size=14, color='black')
)


# Show figure
fig.show()

# Spaghetti plot for Expert 2 rankings

# Reshape rankings data for Plotly
expert2_rankings_df = pd.DataFrame(reversed_expert2_rankings, columns=conditions)
expert2_rankings_df['Sample'] = items

# Melt dataframe for better visualization
expert2_rankings_df_melted = expert2_rankings_df.melt(id_vars=['Sample'], var_name='Condition', value_name='Ranking')

expert2_rankings_df_melted = pd.concat([perfect_df, expert2_rankings_df_melted])

# Create plot
fig = px.line(expert2_rankings_df_melted, x='Condition', y='Ranking', color='Sample', markers=True,
              title="Expert 2 (Competition) Rankings per Condition with Perfect Ranking Reference",
              labels={"Ranking": "Ranking (Lower is Better)", "Condition": "Conditions (Low to High Quality)"},
              template="plotly_white")

# Adjust figure dimensions
fig.update_layout(width=1000, height=800)

# Invert y-axis (lower ranks at top)
fig.update_yaxes(autorange="reversed")

# Modify the "Perfect Ranking" line to be more visible
fig.update_traces(
    selector=dict(name="Perfect Ranking"),
    line=dict(width=10, color='black'),
    marker=dict(size=14, color='black')
)


# Show figure
fig.show()
# Spaghetti plot for BSU1_1 rankings

# Reshape rankings data for Plotly
BSU1_1_rankings_df = pd.DataFrame(reversed_BSU1_1_rankings, columns=conditions)
BSU1_1_rankings_df['Sample'] = items

# Melt dataframe for better visualization
BSU1_1_rankings_df_melted = BSU1_1_rankings_df.melt(id_vars=['Sample'], var_name='Condition', value_name='Ranking')

BSU1_1_rankings_df_melted = pd.concat([perfect_df, BSU1_1_rankings_df_melted])

# Create plot
fig = px.line(BSU1_1_rankings_df_melted, x='Condition', y='Ranking', color='Sample', markers=True,
              title="Cohort 1 Student 1 (Competition) Rankings per Condition with Perfect Ranking Reference",
              labels={"Ranking": "Ranking (Lower is Better)", "Condition": "Conditions (Low to High Quality)"},
              template="plotly_white")

# Adjust figure dimensions
fig.update_layout(width=1000, height=800)

# Invert y-axis (lower ranks at top)
fig.update_yaxes(autorange="reversed")

# Modify the "Perfect Ranking" line to be more visible
fig.update_traces(
    selector=dict(name="Perfect Ranking"),
    line=dict(width=10, color='black'),
    marker=dict(size=14, color='black')
)


# Show figure
fig.show()
# Spaghetti plot for BSU2_1 rankings

# Reshape rankings data for Plotly
BSU2_1_rankings_df = pd.DataFrame(reversed_BSU2_1_rankings, columns=conditions)

BSU2_1_rankings_df['Sample'] = items

# Melt dataframe for better visualization
BSU2_1_rankings_df_melted = BSU2_1_rankings_df.melt(id_vars=['Sample'], var_name='Condition', value_name='Ranking')

BSU2_1_rankings_df_melted = pd.concat([perfect_df, BSU2_1_rankings_df_melted])

# Create plot
fig = px.line(BSU2_1_rankings_df_melted, x='Condition', y='Ranking', color='Sample', markers=True,
              title="Cohort 2 Student 1 (Competition) Rankings per Condition with Perfect Ranking Reference",
              labels={"Ranking": "Ranking (Lower is Better)", "Condition": "Conditions (Low to High Quality)"},
              template="plotly_white")

# Adjust figure dimensions
fig.update_layout(width=1000, height=800)

# Invert y-axis (lower ranks at top)
fig.update_yaxes(autorange="reversed")

# Modify the "Perfect Ranking" line to be more visible
fig.update_traces(
    selector=dict(name="Perfect Ranking"),
    line=dict(width=10, color='black'),
    marker=dict(size=14, color='black')
)


# Show figure
fig.show()
# Spaghetti Plots Based on Contingency Table Results
print('Experts: ', unique_subjects)
print('BSU1: ', unique_subjects_BSU1)
print('BSU2: ', unique_subjects_BSU2)
good_listeners_index = [10, 28, 29, 30, 31, 32, 33, 34, 35, 38]
good_experts = ['Subject 10: USLA04']
good_students_BSU1 = ['D002', 'D003', 'D004', 'D008', 'D009', 'D010', 'D011']
good_students_BSU2 = ['D005', 'D015']
bad_listeners_index = [16, 23]
bad_experts = ['Subject 16: USLA10', 'Subject 23: USLA17']
easy_easiest_trials_competition_kmeans = ['TM_01b_trumpet', 'DE_ElephantsDream_LD0', 'LP_23_jazz', 'LP_AmateurOnPurpose', 'DE_female_speech_music_2_LD9', 'UN_AmateurOnPurpose', 'UN_CreatureFromTheBlackjackTable']
hard_hardest_trials_competition_kmeans = ['SH_AmateurOnPurpose', 'SH_CreatureFromTheBlackjackTable', 'DE_SitaSings_remix2_LD6', 'PE_27_castanets']
# Helper functions
def get_trial_indices(trial_names, items):
    """Return sorted indices of each trial in items."""
    return sorted([int(np.where(items == trial)[0][0]) for trial in trial_names])

def get_reversed_and_indexed(rankings, trial_indices):
    """Reverse the rankings and index by the given trial indices."""
    return rankings[:, ::-1][trial_indices]

def calculate_cumulative_difference(rankings, perfect_ranking):
    """Calculate the cumulative absolute difference row-wise."""
    return np.abs(rankings - perfect_ranking).sum(axis=1)

def compute_stats(diff_df):
    """Compute mean and sum for each column (excluding the 'Sample' column)."""
    stats = []
    for col in diff_df.columns:
        if col != 'Sample':
            stats.append({
                "Column": col,
                "Mean": diff_df[col].mean(),
                "Sum": diff_df[col].sum()
            })
    return pd.DataFrame(stats)

# Get trial indices for easy and hard trials
east_easiest_trial_indices = get_trial_indices(easy_easiest_trials_competition_kmeans, items)
hard_hardest_trial_indices = get_trial_indices(hard_hardest_trials_competition_kmeans, items)
print("Easiest trials:", east_easiest_trial_indices)
print("Hardest trials:", hard_hardest_trial_indices)

# Define ranking arrays in dictionaries for easier iteration.
# Good listeners (easy and hard trials)
good_listeners_easy = {
    'Expert 10': expert10_rankings,
    'BSU1_2': BSU1_2_rankings,
    'BSU1_3': BSU1_3_rankings,
    'BSU1_4': BSU1_4_rankings,
    'BSU1_5': BSU1_5_rankings,
    'BSU1_6': BSU1_6_rankings,
    'BSU1_7': BSU1_7_rankings,
    'BSU1_8': BSU1_8_rankings,
    'BSU2_1': BSU2_1_rankings,
    'BSU2_4': BSU2_4_rankings
}
good_listeners_hard = {
    'Expert 10': expert10_rankings,
    'BSU1_2': BSU1_2_rankings,
    'BSU1_3': BSU1_3_rankings,
    'BSU1_4': BSU1_4_rankings,
    'BSU1_5': BSU1_5_rankings,
    'BSU1_6': BSU1_6_rankings,
    'BSU1_7': BSU1_7_rankings,
    'BSU1_8': BSU1_8_rankings,
    'BSU2_1': BSU2_1_rankings,
    'BSU2_4': BSU2_4_rankings
}

# Bad listeners (easy and hard trials)
bad_listeners_easy = {
    'Expert 16': expert16_rankings,
    'Expert 23': expert23_rankings
}
bad_listeners_hard = {
    'Expert 16': expert16_rankings,
    'Expert 23': expert23_rankings
}

# Process rankings: reverse and index then compute cumulative differences
def process_rankings(ranking_dict, trial_indices, perfect_ranking):
    cum_diff = {}
    for key, array in ranking_dict.items():
        reversed_array = get_reversed_and_indexed(array, trial_indices)
        cum_diff[key] = calculate_cumulative_difference(reversed_array, perfect_ranking)
    return cum_diff

# Get cumulative differences for each group
cum_diff_good_easy = process_rankings(good_listeners_easy, east_easiest_trial_indices, reversed_perfect_ranking)
cum_diff_bad_easy = process_rankings(bad_listeners_easy, east_easiest_trial_indices, reversed_perfect_ranking)
cum_diff_good_hard = process_rankings(good_listeners_hard, hard_hardest_trial_indices, reversed_perfect_ranking)
cum_diff_bad_hard = process_rankings(bad_listeners_hard, hard_hardest_trial_indices, reversed_perfect_ranking)

# Create DataFrames for each case
def build_df(cum_diff_dict, trial_indices):
    df = pd.DataFrame({'Sample': items[trial_indices]})
    for key, diff in cum_diff_dict.items():
        df[key] = diff
    return df

df_good_easy = build_df(cum_diff_good_easy, east_easiest_trial_indices)
df_bad_easy = build_df(cum_diff_bad_easy, east_easiest_trial_indices)
df_good_hard = build_df(cum_diff_good_hard, hard_hardest_trial_indices)
df_bad_hard = build_df(cum_diff_bad_hard, hard_hardest_trial_indices)

# Compute stats (mean and sum) for each DataFrame
stats_good_easy = compute_stats(df_good_easy)
stats_bad_easy = compute_stats(df_bad_easy)
stats_good_hard = compute_stats(df_good_hard)
stats_bad_hard = compute_stats(df_bad_hard)

# Rankings (Good listeners) (Easy trials)
reversed_expert10_rankings_easy_easiest = expert10_rankings[:, ::-1][east_easiest_trial_indices]
reversed_BSU1_2_rankings_easy_easiest = BSU1_2_rankings[:, ::-1][east_easiest_trial_indices]
reversed_BSU1_3_rankings_easy_easiest = BSU1_3_rankings[:, ::-1][east_easiest_trial_indices]
reversed_BSU1_4_rankings_easy_easiest = BSU1_4_rankings[:, ::-1][east_easiest_trial_indices]
reversed_BSU1_5_rankings_easy_easiest = BSU1_5_rankings[:, ::-1][east_easiest_trial_indices]
reversed_BSU1_6_rankings_easy_easiest = BSU1_6_rankings[:, ::-1][east_easiest_trial_indices]
reversed_BSU1_7_rankings_easy_easiest = BSU1_7_rankings[:, ::-1][east_easiest_trial_indices]
reversed_BSU1_8_rankings_easy_easiest = BSU1_8_rankings[:, ::-1][east_easiest_trial_indices]
reversed_BSU2_1_rankings_easy_easiest = BSU2_1_rankings[:, ::-1][east_easiest_trial_indices]
reversed_BSU2_4_rankings_easy_easiest = BSU2_4_rankings[:, ::-1][east_easiest_trial_indices]

# Rankings (Bad listeners) (Easy trials)
reversed_expert16_rankings_easy_easiest = expert16_rankings[:, ::-1][east_easiest_trial_indices]
reversed_expert23_rankings_easy_easiest = expert23_rankings[:, ::-1][east_easiest_trial_indices]

# Rankings (Good listeners) (Hard trials)
reversed_expert10_rankings_hard_hardest = expert10_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_BSU1_2_rankings_hard_hardest = BSU1_2_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_BSU1_3_rankings_hard_hardest = BSU1_3_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_BSU1_4_rankings_hard_hardest = BSU1_4_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_BSU1_5_rankings_hard_hardest = BSU1_5_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_BSU1_6_rankings_hard_hardest = BSU1_6_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_BSU1_7_rankings_hard_hardest = BSU1_7_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_BSU1_8_rankings_hard_hardest = BSU1_8_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_BSU2_1_rankings_hard_hardest = BSU2_1_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_BSU2_4_rankings_hard_hardest = BSU2_4_rankings[:, ::-1][hard_hardest_trial_indices]

# Rankings (Bad listeners) (Hard trials)
reversed_expert16_rankings_hard_hardest = expert16_rankings[:, ::-1][hard_hardest_trial_indices]
reversed_expert23_rankings_hard_hardest = expert23_rankings[:, ::-1][hard_hardest_trial_indices]

