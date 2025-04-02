import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import os
import zipfile
import tensorflow_datasets as tfds
from sklearn import datasets
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set up Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = "vijultyagi"
os.environ['KAGGLE_KEY'] = "7c7ad291191bdd995337b23b0eaf64ca"

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Dataset Generator", page_icon="📊", layout="wide")

# Header with a nice background
st.markdown("""
    <style>
        .title {
            font-size: 50px;
            font-weight: 600;
            text-align: center;
            color: #2a3d66;
            margin-top: 50px;
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            padding: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title for the app
st.markdown('<div class="title">🚀 Dataset Generator</div>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("🔧 Configuration")
st.sidebar.header("Adjust the parameters below to generate a dataset")

# User input options for different datasets
dataset_type = st.sidebar.selectbox("Select Dataset Type", ["Classification", "Regression", "Time-Series", "From URL", "📂 Advanced Dataset Loader"])

# Display UI for "📂 Advanced Dataset Loader" Option
if dataset_type == "📂 Advanced Dataset Loader":
    st.title("📂 Advanced Dataset Loader")
    st.sidebar.subheader("🔍 Search & Load Datasets")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # ------------------------ Scikit-learn Datasets ------------------------
    st.sidebar.subheader("📊 Scikit-learn Datasets")
    sklearn_datasets = {
        "Iris": "iris",
        "Wine": "wine",
        "Breast Cancer": "breast_cancer",
        "Diabetes": "diabetes",
        "Digits": "digits"
    }

    selected_sklearn = st.sidebar.selectbox("Select Scikit-learn Dataset:", list(sklearn_datasets.keys()))
    if st.sidebar.button("📊 Load Scikit-learn Dataset"):
        data = getattr(datasets, f"load_{sklearn_datasets[selected_sklearn]}")()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.write(f"### {selected_sklearn} Dataset (Scikit-learn)")
        st.dataframe(df.head())

    # ------------------------ TensorFlow Datasets ------------------------
    st.sidebar.subheader("📊 TensorFlow Datasets")
    tfds_datasets = tfds.list_builders()
    selected_tfds = st.sidebar.selectbox("Select TensorFlow Dataset:", tfds_datasets)
    if st.sidebar.button("📊 Load TensorFlow Dataset"):
        try:
            ds = tfds.load(selected_tfds, split='train')
            df = tfds.as_dataframe(ds)
            st.write(f"### {selected_tfds} Dataset (TensorFlow Datasets)")
            st.dataframe(df.head())
        except Exception as e:
            st.sidebar.error(f"⚠️ TensorFlow Datasets Error: {e}")

    # ------------------------ OpenML Datasets ------------------------
    st.sidebar.subheader("📊 OpenML Datasets")
    openml_dataset_id = st.sidebar.text_input("Enter OpenML Dataset ID:")
    if st.sidebar.button("📊 Load OpenML Dataset") and openml_dataset_id.isdigit():
        try:
            df = fetch_openml(data_id=int(openml_dataset_id), as_frame=True).data
            st.write(f"### OpenML Dataset (ID: {openml_dataset_id})")
            st.dataframe(df.head())
        except Exception as e:
            st.sidebar.error(f"⚠️ OpenML Error: {e}")
    else:
        if openml_dataset_id and not openml_dataset_id.isdigit():
            st.sidebar.error("⚠️ Please enter a valid numeric OpenML Dataset ID.")

    # ------------------------ Kaggle Dataset Search ------------------------
    st.sidebar.subheader("📂 Kaggle Datasets")
    search_query = st.sidebar.text_input("Enter Kaggle dataset keyword:")
    if st.sidebar.button("🔍 Search Kaggle"):
        if search_query:
            datasets_kaggle = api.dataset_list(search_query)
            if datasets_kaggle:
                dataset_ref = st.sidebar.selectbox("Select a Kaggle dataset:", datasets_kaggle, format_func=lambda x: x.ref)
                if st.sidebar.button("⬇️ Download Dataset"):
                    path = f"datasets/{dataset_ref.split('/')[-1]}"
                    api.dataset_download_files(dataset_ref, path="datasets", unzip=True)
                    st.success(f"✅ Dataset downloaded to: {path}")
            else:
                st.sidebar.warning("⚠️ No datasets found. Try a different keyword.")
    
    # ------------------------ Seaborn Datasets ------------------------
    st.sidebar.subheader("📊 Seaborn Datasets")
    seaborn_datasets = sns.get_dataset_names()
    selected_seaborn = st.sidebar.selectbox("Select Seaborn Dataset:", seaborn_datasets)
    if st.sidebar.button("📊 Load Seaborn Dataset"):
        df = sns.load_dataset(selected_seaborn)
        st.write(f"### {selected_seaborn} Dataset (Seaborn)")
        st.dataframe(df.head())
    
    # ------------------------ Show Downloaded Kaggle Datasets ------------------------
    if os.path.exists("datasets"):
        st.sidebar.subheader("📂 View Downloaded Datasets")
        files = os.listdir("datasets")
        selected_file = st.sidebar.selectbox("Select a file:", ["Select a file"] + files)
        if selected_file != "Select a file" and selected_file.endswith(".csv"):
            df = pd.read_csv(os.path.join("datasets", selected_file))
            st.write(f"### {selected_file} Dataset (Downloaded from Kaggle)")
            st.dataframe(df.head())

else:
    # Dataset Generation Options for other categories (Classification, Regression, etc.)
    st.sidebar.header("Adjust the parameters below to generate a dataset")
    num_rows = st.sidebar.slider("Number of Rows", 100, 1000, 500)
    num_features = st.sidebar.slider("Number of Features", 2, 20, 10)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
    outlier_percentage = st.sidebar.slider("Outlier Percentage", 0.0, 0.2, 0.05)
    scale_features = st.sidebar.selectbox("Feature Scaling", ["None", "Standardization", "Normalization"])
    missing_data_percentage = st.sidebar.slider("Missing Data Percentage", 0.0, 0.2, 0.05)
    
    # Select the dataset type
    if dataset_type == "Classification":
        target = np.random.choice([0, 1], size=num_rows)
        features = np.random.randn(num_rows, num_features)
        if noise_level > 0:
            features += np.random.normal(0, noise_level, size=features.shape)  # Adding noise
        if outlier_percentage > 0:
            num_outliers = int(outlier_percentage * num_rows)
            outlier_indices = np.random.choice(num_rows, num_outliers, replace=False)
            features[outlier_indices] = np.random.uniform(-10, 10, (num_outliers, num_features))
        if scale_features != "None":
            if scale_features == "Standardization":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
        df = pd.DataFrame(features, columns=[f"Feature_{i+1}" for i in range(num_features)])
        df["Target"] = target
        st.subheader("📊 Generated Classification Dataset")
        st.dataframe(df.head())

    elif dataset_type == "Regression":
        target = np.random.randn(num_rows)
        features = np.random.randn(num_rows, num_features)
        if noise_level > 0:
            features += np.random.normal(0, noise_level, size=features.shape)  # Adding noise
        if outlier_percentage > 0:
            num_outliers = int(outlier_percentage * num_rows)
            outlier_indices = np.random.choice(num_rows, num_outliers, replace=False)
            features[outlier_indices] = np.random.uniform(-10, 10, (num_outliers, num_features))
        if scale_features != "None":
            if scale_features == "Standardization":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
        df = pd.DataFrame(features, columns=[f"Feature_{i+1}" for i in range(num_features)])
        df["Target"] = target
        st.subheader("📊 Generated Regression Dataset")
        st.dataframe(df.head())

    elif dataset_type == "Time-Series":
        time_index = pd.date_range(start="2020-01-01", periods=num_rows, freq="D")
        features = np.random.randn(num_rows, num_features)
        if noise_level > 0:
            features += np.random.normal(0, noise_level, size=features.shape)  # Adding noise
        if outlier_percentage > 0:
            num_outliers = int(outlier_percentage * num_rows)
            outlier_indices = np.random.choice(num_rows, num_outliers, replace=False)
            features[outlier_indices] = np.random.uniform(-10, 10, (num_outliers, num_features))
        if scale_features != "None":
            if scale_features == "Standardization":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
        df = pd.DataFrame(features, columns=[f"Feature_{i+1}" for i in range(num_features)], index=time_index)
        st.subheader("📊 Generated Time-Series Dataset")
        st.dataframe(df.head())

    elif dataset_type == "From URL":
        url = st.sidebar.text_input("Enter URL for dataset:", "")
        if st.sidebar.button("📥 Load Dataset"):
            try:
                df = pd.read_csv(url)
                st.write("### Dataset from URL")
                st.dataframe(df.head())
            except Exception as e:
                st.sidebar.error(f"⚠️ URL Loading Error: {e}")

    # Export buttons
    st.sidebar.subheader("Download Dataset")
    if not df.empty:
        st.sidebar.download_button(
            label="Download as CSV",
            data=df.to_csv(index=False),
            file_name="generated_dataset.csv",
            mime="text/csv"
        )
        # Excel Export with ExcelWriter
        import io
        with io.BytesIO() as excel_buffer:
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Sheet1")
            st.sidebar.download_button(
                label="Download as Excel",
                data=excel_buffer.getvalue(),
                file_name="generated_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
