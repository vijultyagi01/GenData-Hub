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

# ✅ Set Kaggle API Credentials
os.environ['KAGGLE_USERNAME'] = "vijultyagi"
os.environ['KAGGLE_KEY'] = "1e76833f680568077457316c7a5ea42e"

# ✅ Ensure TFDS manual directory exists
tfds_dir = os.path.join(os.path.expanduser("~"), "tensorflow_datasets", "downloads", "manual")
os.makedirs(tfds_dir, exist_ok=True)

# ✅ Streamlit Page Config
st.set_page_config(page_title="Dataset Generator", page_icon="📊", layout="wide")
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
st.markdown('<div class="title">🚀 Dataset Generator</div>', unsafe_allow_html=True)

# ✅ Sidebar: Dataset Type Selection
st.sidebar.title("🔧 Configuration")
dataset_type = st.sidebar.selectbox("Select Dataset Type", [
    "Classification", "Regression", "Time-Series", "From URL", "📂 Advanced Dataset Loader"
])

# ✅ Utilities
def add_data_variations(features, noise_level, outlier_percentage):
    if noise_level > 0:
        features += np.random.normal(0, noise_level, size=features.shape)
    if outlier_percentage > 0:
        n_outliers = int(outlier_percentage * features.shape[0])
        indices = np.random.choice(features.shape[0], n_outliers, replace=False)
        features[indices] = np.random.uniform(-10, 10, (n_outliers, features.shape[1]))
    return features

def scale_data(features, method):
    if method == "Standardization":
        return StandardScaler().fit_transform(features)
    elif method == "Normalization":
        return MinMaxScaler().fit_transform(features)
    return features

# ✅ Session States for Kaggle Search Results
if 'kaggle_results' not in st.session_state:
    st.session_state.kaggle_results = []
if 'selected_kaggle_dataset' not in st.session_state:
    st.session_state.selected_kaggle_dataset = ""

# ✅ Advanced Loader Mode
if dataset_type == "📂 Advanced Dataset Loader":
    st.title("📂 Advanced Dataset Loader")
    st.sidebar.header("🔍 Search & Load Datasets")

    api = KaggleApi()
    api.authenticate()

    # 📊 Scikit-learn Datasets
    st.sidebar.subheader("📊 Scikit-learn")
    skl_datasets = {
        "Iris": "iris", "Wine": "wine", "Breast Cancer": "breast_cancer",
        "Diabetes": "diabetes", "Digits": "digits"
    }
    skl_choice = st.sidebar.selectbox("Scikit-learn Dataset", list(skl_datasets.keys()))
    if st.sidebar.button("Load Scikit-learn Dataset"):
        data = getattr(datasets, f"load_{skl_datasets[skl_choice]}")()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.subheader(f"{skl_choice} Dataset (Scikit-learn)")
        st.dataframe(df.head())

    # 📊 TensorFlow Datasets
    st.sidebar.subheader("📊 TensorFlow Datasets")
    tfds_list = tfds.list_builders()
    tfds_choice = st.sidebar.selectbox("TensorFlow Dataset", tfds_list)
    if st.sidebar.button("Load TFDS Dataset"):
        try:
            ds = tfds.load(tfds_choice, split="train")
            df = tfds.as_dataframe(ds)
            st.subheader(f"{tfds_choice} Dataset (TFDS)")
            st.dataframe(df.head())
        except Exception as e:
            st.sidebar.error("❌ TFDS Load Error:")
            st.error(f"""
                🔔 Some TFDS datasets (like Ravens Matrices) require manual download.
                \nGo to: https://console.cloud.google.com/storage/browser/ravens-matrices
                \nPlace the downloaded files into:
                \n`{tfds_dir}`
                \n\nError: {e}
            """)

    # 📊 OpenML Dataset
    st.sidebar.subheader("📊 OpenML Dataset")
    openml_id = st.sidebar.text_input("OpenML Dataset ID")
    if st.sidebar.button("Load OpenML Dataset") and openml_id.isdigit():
        try:
            df = fetch_openml(data_id=int(openml_id), as_frame=True).data
            st.subheader(f"OpenML Dataset (ID: {openml_id})")
            st.dataframe(df.head())
        except Exception as e:
            st.sidebar.error(f"OpenML Error: {e}")

    # 📂 Kaggle Dataset Search
    st.sidebar.subheader("📂 Kaggle Datasets")
    search_query = st.sidebar.text_input("🔍 Kaggle Search Keyword")
    if st.sidebar.button("🔍 Search on Kaggle"):
        try:
            results = api.dataset_list(search=search_query)
            st.session_state.kaggle_results = [d.ref for d in results]
            st.success(f"Found {len(results)} datasets.")
        except Exception as e:
            st.sidebar.error(f"🚨 Kaggle Search Error: {e}")

    if st.session_state.kaggle_results:
        st.session_state.selected_kaggle_dataset = st.sidebar.selectbox("Choose Kaggle Dataset", st.session_state.kaggle_results)
        if st.sidebar.button("⬇️ Download Selected Dataset"):
            try:
                dataset_ref = st.session_state.selected_kaggle_dataset
                dataset_folder = dataset_ref.split("/")[-1]
                path = os.path.join("datasets", dataset_folder)
                os.makedirs(path, exist_ok=True)
                api.dataset_download_files(dataset_ref, path=path, unzip=True)
                st.success(f"✅ Downloaded and extracted to `{path}`")

                csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
                if csv_files:
                    df = pd.read_csv(os.path.join(path, csv_files[0]))
                    st.subheader(f"📄 {csv_files[0]} (from Kaggle)")
                    st.dataframe(df.head())
                else:
                    st.warning("Downloaded dataset doesn't contain CSV files.")
            except Exception as e:
                st.sidebar.error(f"❌ Kaggle Download Error: {e}")

    # 📊 Seaborn Datasets
    st.sidebar.subheader("📊 Seaborn Datasets")
    seaborn_list = sns.get_dataset_names()
    seaborn_choice = st.sidebar.selectbox("Seaborn Dataset", seaborn_list)
    if st.sidebar.button("Load Seaborn Dataset"):
        df = sns.load_dataset(seaborn_choice)
        st.subheader(f"{seaborn_choice} Dataset (Seaborn)")
        st.dataframe(df.head())

    # 📁 Local CSV Viewer
    if os.path.exists("datasets"):
        st.sidebar.subheader("📂 Local CSV Viewer")
        all_dirs = [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]
        file_choice = st.sidebar.selectbox("Choose CSV Folder", ["-- Select --"] + all_dirs)
        if file_choice != "-- Select --":
            csv_files = [f for f in os.listdir(os.path.join("datasets", file_choice)) if f.endswith(".csv")]
            if csv_files:
                df = pd.read_csv(os.path.join("datasets", file_choice, csv_files[0]))
                st.subheader(f"{csv_files[0]} (from Local Kaggle)")
                st.dataframe(df.head())

# ✅ Dataset Generator
else:
    st.sidebar.header("🧪 Generate Dataset")
    rows = st.sidebar.slider("Rows", 100, 1000, 500)
    cols = st.sidebar.slider("Features", 2, 20, 10)
    noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
    outliers = st.sidebar.slider("Outlier %", 0.0, 0.2, 0.05)
    scaling = st.sidebar.selectbox("Scaling", ["None", "Standardization", "Normalization"])
    missing_pct = st.sidebar.slider("Missing Data %", 0.0, 0.2, 0.05)

    features = np.random.randn(rows, cols)
    features = add_data_variations(features, noise, outliers)
    features = scale_data(features, scaling)

    if dataset_type == "Classification":
        target = np.random.choice([0, 1], rows)
        df = pd.DataFrame(features, columns=[f"Feature_{i+1}" for i in range(cols)])
        df["Target"] = target
        st.subheader("📊 Classification Dataset")

    elif dataset_type == "Regression":
        target = np.random.randn(rows)
        df = pd.DataFrame(features, columns=[f"Feature_{i+1}" for i in range(cols)])
        df["Target"] = target
        st.subheader("📊 Regression Dataset")

    elif dataset_type == "Time-Series":
        index = pd.date_range("2020-01-01", periods=rows, freq="D")
        df = pd.DataFrame(features, columns=[f"Feature_{i+1}" for i in range(cols)], index=index)
        st.subheader("📊 Time-Series Dataset")

    elif dataset_type == "From URL":
        url = st.sidebar.text_input("Dataset URL")
        if st.sidebar.button("Load Dataset"):
            try:
                df = pd.read_csv(url)
                st.subheader("📥 Dataset from URL")
            except Exception as e:
                st.sidebar.error(f"URL Load Error: {e}")

    # Add Missing Data
    if missing_pct > 0 and 'df' in locals():
        mask = np.random.rand(*df.shape) < missing_pct
        df = df.mask(mask)

    if 'df' in locals():
        st.dataframe(df.head())

        # ✅ Download Buttons
        st.sidebar.subheader("⬇️ Download Dataset")
        st.sidebar.download_button("Download CSV", df.to_csv(index=False), "generated_dataset.csv", "text/csv")

        import io
        with io.BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False)
            st.sidebar.download_button("Download Excel", buffer.getvalue(), "generated_dataset.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
