import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score
from matplotlib import cm
import mpld3
import streamlit.components.v1 as components
import seaborn as sns

def data_prepare():
    # File ID from the Google Drive share link
    # Load initial dataset
    file_id = "16EyRIcs_WUzolDfikMLcHWknm_SO75y9"
    url = f"https://drive.google.com/uc?id={file_id}"

    #Preprocessed dataset
    file_id2 = "1_CvWa-pJ1pCkStWjZ_PWmW441sKxPjVD"
    url2 = f"https://drive.google.com/uc?id={file_id2}"
 
    # Load all file
    df = pd.read_csv(url)
    df_cleaned = pd.read_csv(url2)
    return df, df_cleaned

def data_preprocessing(df):
    #Fill in missing value
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    # Detecting outlier using IQR
    def detect_outliers_iqr(df):
        outlier_df = pd.DataFrame(False, index=df.index, columns=df.columns)
        lower_bounds = {}
        upper_bounds = {}

        for column in df.select_dtypes(include=['number']).columns:
          Q1 = df[column].quantile(0.25)
          Q3 = df[column].quantile(0.75)
          IQR = Q3 - Q1
          lower_bound = Q1 - 1.5 * IQR
          upper_bound = Q3 + 1.5 * IQR

          lower_bounds[column] = lower_bound
          upper_bounds[column] = upper_bound

          outlier_df[column] = (df[column] < lower_bound) | (df[column] > upper_bound)

        return outlier_df, lower_bounds, upper_bounds

    #Outliers
    outlier_df, lower_bounds, upper_bounds = detect_outliers_iqr(df)

    #Remove outliers from the dataset
    df_cleaned = df[~outlier_df.any(axis=1)]

    #Convert necessary columns to categorical
    df_cleaned['wd'] = df_cleaned['wd'].astype('category')
    return df_cleaned

def feature_engineering(df_cleaned):
    #Combine 'year', 'month', 'day', 'hour' columns to datetime
    df_cleaned['datetime'] = pd.to_datetime(df_cleaned[['year','month','day','hour']])
    df_cleaned.insert(1,'datetime',df_cleaned.pop('datetime'))

    #Remove irrelevant features
    #'No' is row number
    #This dataset is specific for a single 'station'
    #'year', 'month', 'day', 'hour' replaced by datetime
    df_cleaned.drop(columns=['No', 'year', 'month', 'day', 'hour', 'station',], inplace=True)

    #Encoding
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_wd = encoder.fit_transform(df_cleaned[['wd']])

    #reset index to avoid misalignment
    df_cleaned.reset_index(drop=True, inplace=True)

    df_one_hot = pd.DataFrame(one_hot_wd, columns=encoder.get_feature_names_out(['wd']))
    df_encoded = pd.concat([df_cleaned.drop(columns='wd'), df_one_hot], axis=1)

    #Select numerical columns for normalization
    numerical_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns

    scaler = MinMaxScaler()
    df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

    # Apply PCA for algorithm
    # Reduce dimensions while keeping 80% of variance
    pca_train = PCA(n_components=0.8)
    df_pca_train = pca_train.fit_transform(df_encoded.drop(['datetime'], axis=1))
    df_pca_df = pd.DataFrame(df_pca_train, columns=[f"PC{i+1}" for i in range(df_pca_train.shape[1])])

    # Apply PCA for visualization
    pca_plot = PCA(n_components=2)
    df_pca_plot = pca_plot.fit_transform(df_pca_train)
    df_plot = pd.DataFrame(df_pca_plot, columns=["PC1", "PC2"])

    # convert to numpy
    df_numpy = df_pca_df.to_numpy()
    df_plot = df_plot.to_numpy()

    return df_plot, df_numpy

def gmm_model(df_numpy):
    # Fit GMM model
    gmm = GaussianMixture(n_components=7, covariance_type='full', random_state=42)
    gmm.fit(df_numpy)
    labels = gmm.predict(df_numpy)
    return labels

def mean_shift_model(df_numpy):
    # Estimate bandwidth using a suitable method (e.g., 'scott' or 'silverman')
    bandwidth = estimate_bandwidth(df_numpy, quantile=0.05)

    # Initialize MeanShift with the estimated bandwidth
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=1)

    # Fit MeanShift to the data
    ms.fit(df_numpy)

    # Get cluster labels
    labels = ms.labels_

    # Get cluster centers
    cluster_centers = ms.cluster_centers_

    return labels, cluster_centers

def spectral_clustering_model(df_numpy):
    # Perform Spectral Clustering
    sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)

    # Fit the model and predict the cluster labels
    labels = sc.fit_predict(df_numpy)

    return labels

def optics_model(df_numpy):
    optics = OPTICS(min_samples=3, xi=0.2, min_cluster_size=0.05)

    # Fit the model to your data
    labels = optics.fit_predict(df_numpy)
    return labels

def birch_model(df_numpy):
    birch = Birch(threshold=0.1, branching_factor=10,n_clusters=6)
    # Get the cluster labels from the best result
    labels = birch.fit_predict(df_numpy)
    return labels

def agglo_model(df_numpy):
    agglo = AgglomerativeClustering(n_clusters=6,linkage='ward')
    # Get the cluster labels from the best result
    labels = agglo.fit_predict(df_numpy)
    return labels

def kmeans_model(df_numpy):
    # Maximin initialization function
    def initialize_maximin(X, k):
        centroids = [X[np.random.choice(range(X.shape[0]))]]
        for _ in range(1, k):
            dist_sq = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2)**2, axis=1)
            next_centroid = X[np.argmax(dist_sq)]
            centroids.append(next_centroid)
        return np.array(centroids)

    # Perform KMeans clustering with the best parameter value
    kmeans = KMeans(n_clusters=10, init=initialize_maximin(df_numpy,10), n_init=1, max_iter=100, algorithm='elkan',random_state=42)
    labels = kmeans.fit_predict(df_numpy)

    return labels, kmeans.cluster_centers_

def fuzzy_c_means_model(df_numpy):
    # Fuzzy C-Means clustering
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        df_numpy.T,           # make sure df_numpy is shape (features, samples) after transpose
        c=10,             # number of clusters
        m=1.3,             # fuzziness parameter
        error=0.005,     # stopping criterion
        maxiter=1000  # maximum number of iterations
    )

    # get hard cluster labels with the highest membership value
    cluster_membership = np.argmax(u, axis=0)
    return cluster_membership, cntr

 
st.title("🌫️ Clustering of Air Quality in Beijing🌍")

#Dataset Preparation
st.subheader("📊 Dataset Preparation")
st.subheader("🔍 Choose Your Dataset")

# --- Initialize model scores and data option tracking in session state ---
if 'models_scores' not in st.session_state:
    st.session_state.models_scores = [
        ('Gaussian Mixture Models', None),
        ('Mean Shift', None),
        ('Spectral Clustering', None),
        ('OPTICS', None),
        ('BIRCH', None),
        ('Agglomerative Clustering', None),
        ('Enhanced K-Means', None),
        ('Fuzzy C-Means', None),
    ]

if 'previous_data_option' not in st.session_state:
    st.session_state.previous_data_option = None

#data_option = st.radio(
#    "Would you like to use the sample dataset or upload own dataset?",
#    ("Use Sample Data", "Upload Own Data")
#)
data_option = "Upload Own Data"

if st.session_state.previous_data_option is not None and data_option != st.session_state.previous_data_option:
    for i in range(len(st.session_state.models_scores)):
        st.session_state.models_scores[i] = (st.session_state.models_scores[i][0], None)
st.session_state.previous_data_option = data_option

# Handle dataset selection
#if data_option == "Use Sample Data":
#    df, df_cleaned = data_prepare()
#elif data_option == "Upload Own Data":
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_cleaned = data_preprocessing(df)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Display Datasets
st.subheader("📊 Initial Dataset")
st.dataframe(df)

#Data Exploration
st.subheader("🔎 Data Exploration")

st.markdown("Use the options below to filter and explore the data.")

# Numeric column filtering
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if numeric_columns:
    selected_num_col = st.selectbox("Select a numeric column to filter", numeric_columns)
    min_val = df[selected_num_col].min()
    max_val = df[selected_num_col].max()
    user_range = st.slider(f"Filter {selected_num_col} range", min_val, max_val, (min_val, max_val))
    df_filtered = df[(df[selected_num_col] >= user_range[0]) & (df[selected_num_col] <= user_range[1])]
else:
    df_filtered = df

# Categorical filtering (if any)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
if categorical_columns:
    selected_cat_col = st.selectbox("Select a categorical column to filter", categorical_columns)
    unique_values = df[selected_cat_col].unique()
    selected_values = st.multiselect(f"Filter {selected_cat_col}", unique_values, default=list(unique_values))
    df_filtered = df_filtered[df_filtered[selected_cat_col].isin(selected_values)]

st.write("🎯 Filtered Dataset")
st.dataframe(df_filtered)

# Visualize numeric column distribution
if numeric_columns:
    col_to_plot = st.selectbox("Select numeric column to visualize", numeric_columns)
    fig, ax = plt.subplots()
    ax.hist(df_filtered[col_to_plot], bins=20, color='skyblue', edgecolor='black')
    ax.set_title(f"Distribution of {col_to_plot}")
    ax.set_xlabel(col_to_plot)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    #def plot_distribution(column):
    #plt.figure(figsize=(10, 6))

    # Plot histogram with kde
    #sns.histplot(df_cleaned[column], bins=30, kde=True)

    # Plot mean and median lines
    #plt.axvline(df_cleaned[column].mean(), color='red', linestyle='--', linewidth=2)
    #plt.axvline(df_cleaned[column].median(), color='green', linestyle='-', linewidth=2)

    #Labels and title
    #plt.title(f'Distribution of {column}')
    #plt.xlabel(column)
    #plt.ylabel('Frequency')
    #plt.show()

    # Visualize the clusters
    #fig, ax = plt.subplots()
    #scatter = ax.scatter(df_plot[:, 0], df_plot[:, 1], c=labels, cmap='viridis')
    #plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
    #ax.set_title("Mean Shift Clustering")
    #ax.set_xlabel("PCA 1")
    #ax.set_ylabel("PCA 2")

    # Display in Streamlit
    #st.pyplot(fig)

# Display cleaned dataset
st.subheader("📊 Cleaned Dataset")
st.success("🎉 Dataset has been successfully cleaned.")
st.dataframe(df_cleaned)

#Feature engineering
df_plot, df_numpy = feature_engineering(df_cleaned)

#PCA Visualization
# Plot PCA result
st.subheader("📊 PCA Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_plot[:, 0], df_plot[:, 1], c='blue', alpha=0.5)
ax.set_title('PCA - 2D Visualization')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
st.pyplot(fig)

#Interactive Charts
st.subheader("📊 Interactive Charts")

# Sidebar to select algorithm
algorithm = st.selectbox("Choose Clustering Algorithm", ["Gaussian Mixture Model", "Mean Shift", "Spectral Clustering","OPTICS","BIRCH","Agglomerative Clustering","Enhanced K-Means","Fuzzy C-Means"])

# Run clustering based on selected algorithm
if algorithm == "Gaussian Mixture Model":
    labels = gmm_model(df_numpy)
    st.session_state.models_scores[0] = ('Gaussian Mixture Models', silhouette_score(df_numpy, labels))

    # Plot using matplotlib
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_plot[:, 0], df_plot[:, 1], c=labels, cmap='viridis')
    ax.set_title("Gaussian Mixture Model Clustering")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # Display in Streamlit
    st.pyplot(fig)

elif algorithm == "Mean Shift":
    labels, cluster_centers = mean_shift_model(df_numpy)
    st.session_state.models_scores[1] = ('Mean Shift', silhouette_score(df_numpy, labels))

    # Visualize the clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_plot[:, 0], df_plot[:, 1], c=labels, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
    ax.set_title("Mean Shift Clustering")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # Display in Streamlit
    st.pyplot(fig)

elif algorithm == "Spectral Clustering":
    labels = spectral_clustering_model(df_numpy)
    st.session_state.models_scores[2] = ('Spectral Clustering', silhouette_score(df_numpy, labels))

    # Plot using matplotlib
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_plot[:, 0], df_plot[:, 1], c=labels, cmap='viridis')
    ax.set_title("Spectral Clustering")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # Display in Streamlit
    st.pyplot(fig)

elif algorithm == "OPTICS":
    labels = optics_model(df_numpy)
    st.session_state.models_scores[3] = ('OPTICS', silhouette_score(df_numpy, labels))

    # Plot using matplotlib
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_plot[:, 0], df_plot[:, 1], c=labels, cmap='viridis')
    ax.set_title("OPTICS Clustering")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # Display in Streamlit
    st.pyplot(fig)

elif algorithm == "BIRCH":
    labels = birch_model(df_numpy)
    st.session_state.models_scores[4] = ('BIRCH', silhouette_score(df_numpy, labels))

    # Plot using matplotlib
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_plot[:, 0], df_plot[:, 1], c=labels, cmap='viridis')
    ax.set_title("BIRCH Clustering")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # Display in Streamlit
    st.pyplot(fig)

elif algorithm == "Agglomerative Clustering":
    labels = agglo_model(df_numpy)
    st.session_state.models_scores[5] = ('Agglomerative Clustering', silhouette_score(df_numpy, labels))

    # Plot using matplotlib
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_plot[:, 0], df_plot[:, 1], c=labels, cmap='viridis')
    ax.set_title("Agglomerative Clustering")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")

    # Display in Streamlit
    st.pyplot(fig)

elif algorithm == "Enhanced K-Means":
    labels, cluster_centers = kmeans_model(df_numpy)
    st.session_state.models_scores[6] = ('Enhanced K-Means', silhouette_score(df_numpy, labels))

    # Visualize the results
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_plot[:, 0], df_plot[:, 1], c=labels, cmap='viridis', marker='o', s=100)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                s=200, c='red', marker='x', label='Cluster Centers')  # cluster centers

    for i in range(cluster_centers.shape[0]):
        ax.text(cluster_centers[i, 0], cluster_centers[i, 1], f'Cluster {i+1}', color='red', fontsize=12)

    ax.set_title("KMeans Clustering with Maximin Initialization")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

elif algorithm == "Fuzzy C-Means":
    cluster_membership, cntr = fuzzy_c_means_model(df_numpy)
    st.session_state.models_scores[7] = ('Fuzzy C-Means', silhouette_score(df_numpy, cluster_membership))

    # Plot the points
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_plot[:, 0], df_plot[:, 1], c=cluster_membership, cmap='viridis', marker='o', s=100)

    # plot cluster centers
    ax.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='x', s=200, label='Cluster Centers')

    # add cluster labels
    for i in range(cntr.shape[0]):
        ax.text(cntr[i, 0], cntr[i, 1], f'Cluster {i+1}', color='red', fontsize=12)

    ax.set_title("Fuzzy C-Means Clustering")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()
    st.pyplot(fig)


# Comparative Analysis
st.subheader("📊 Comparative Analysis")

# Prepare data for plotting
model_names = [model[0] for model in st.session_state.models_scores]
scores = [model[1] if model[1] is not None else 0 for model in st.session_state.models_scores]

# Create a colormap with unique colors for each bar
colors = cm.viridis(np.linspace(0, 1, len(model_names)))
 
# Create the matplotlib figure
fig, ax = plt.subplots()
bars = ax.bar(model_names, scores, color=colors)

#Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

# Customize appearance
ax.set_ylabel('Silhouette Score')
ax.set_title('Comparison of Silhouette Scores for each Clustering Algorithm')
plt.xticks(rotation=45, ha='right')

# Display in Streamlit
st.pyplot(fig)
