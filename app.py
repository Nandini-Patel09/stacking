
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Stacking Model",
    layout="wide",
    page_icon="🧠"
)

# --------------------------------------------------
# ADVANCED DARK UI
# --------------------------------------------------
st.markdown("""
<style>

/* Main App */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #020617
    );
    color: #f8fafc;
}

/* Title */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    padding: 10px;
    color: #f8fafc;
    letter-spacing: 1px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #334155;
}

/* Cards */
.metric-card {
    background: rgba(30,41,59,0.8);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #334155;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

/* Buttons */
.stButton>button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #6d28d9
    );
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
    border: 1px solid #334155;
}

/* Selectboxes */
div[data-baseweb="select"] > div {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
}

/* Slider */
.stSlider > div {
    color: white;
}

/* Headers */
h1, h2, h3 {
    color: #f8fafc;
}

/* Success */
.stSuccess {
    border-radius: 12px;
}

/* Info */
.stInfo {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOGGER
# --------------------------------------------------
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[{timestamp}] {message}")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# --------------------------------------------------
# FOLDERS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="main-title">
        🧠 Stacking Model
    </div>
    <div class="subtitle">
        Intelligent Classification & Regression System
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("## ⚙️ Stacking Controls")

task_type = st.sidebar.selectbox(
    "🧠 Task Type",
    ["Classification", "Regression"]
)

test_size = st.sidebar.slider(
    "📊 Test Size (%)",
    10,
    40,
    25
)

cv_folds = st.sidebar.slider(
    "🔄 CV Folds",
    3,
    10,
    5
)

random_state = st.sidebar.number_input(
    "🎲 Random State",
    value=42
)

use_grid = st.sidebar.checkbox(
    "🚀 Enable GridSearchCV"
)

# --------------------------------------------------
# STEP 1 : DATA INGESTION
# --------------------------------------------------
st.header("📥 Step 1 : Data Ingestion")

option = st.radio(
    "Choose Data Source",
    ["Download Dataset", "Upload CSV"]
)

df = st.session_state.df

if option == "Download Dataset":

    if st.button("⬇️ Download Iris Dataset"):

        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

        response = requests.get(url)

        raw_path = os.path.join(
            RAW_DIR,
            "iris.csv"
        )

        with open(raw_path, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(raw_path)

        st.session_state.df = df

        st.success("Dataset downloaded successfully")

        log("Dataset downloaded")

# Upload CSV
if option == "Upload CSV":

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:

        raw_path = os.path.join(
            RAW_DIR,
            uploaded_file.name
        )

        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = pd.read_csv(raw_path)

        st.session_state.df = df

        st.success("Dataset uploaded successfully")

        log("CSV uploaded")

# --------------------------------------------------
# STEP 2 : EDA
# --------------------------------------------------
if df is not None:

    st.header("📊 Step 2 : Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h2>{df.shape[0]}</h2>
                <p>Rows</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h2>{df.shape[1]}</h2>
                <p>Columns</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h2>{df.isnull().sum().sum()}</h2>
                <p>Missing Values</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🧾 Dataset Preview")
    st.dataframe(df.head())

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:

        st.markdown("### 🔥 Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax
        )

        st.pyplot(fig)

# --------------------------------------------------
# STEP 3 : DATA CLEANING
# --------------------------------------------------
if df is not None:

    st.header("🧹 Step 3 : Data Cleaning")

    strategy = st.selectbox(
        "Choose Missing Value Strategy",
        ["Mean", "Median", "Drop Rows"]
    )

    df_clean = df.copy()

    if strategy == "Drop Rows":

        df_clean.dropna(inplace=True)

    else:

        for col in df_clean.select_dtypes(include=np.number):

            if strategy == "Mean":

                df_clean[col] = df_clean[col].fillna(
                    df_clean[col].mean()
                )

            else:

                df_clean[col] = df_clean[col].fillna(
                    df_clean[col].median()
                )

    st.session_state.df_clean = df_clean

    st.success("Data cleaned successfully")

# --------------------------------------------------
# STEP 4 : SAVE CLEANED DATA
# --------------------------------------------------
st.header("💾 Step 4 : Save Cleaned Dataset")

if st.button("Save Cleaned Dataset"):

    if st.session_state.df_clean is None:

        st.error("No cleaned data found")

    else:

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = f"cleaned_{ts}.csv"

        path = os.path.join(
            CLEAN_DIR,
            filename
        )

        st.session_state.df_clean.to_csv(
            path,
            index=False
        )

        st.success("Cleaned dataset saved")

# --------------------------------------------------
# STEP 5 : LOAD CLEANED DATASET
# --------------------------------------------------
st.header("📂 Step 5 : Load Cleaned Dataset")

files = os.listdir(CLEAN_DIR)

df_model = None

if files:

    selected = st.selectbox(
        "Select Dataset",
        files
    )

    df_model = pd.read_csv(
        os.path.join(CLEAN_DIR, selected)
    )

    st.dataframe(df_model.head())
# --------------------------------------------------
# STEP 6 : TRAIN STACKING MODEL
# --------------------------------------------------
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.impute import SimpleImputer

if df_model is not None:

    st.header("🧠 Step 6 : Train Stacking Model")

    target = st.selectbox(
        "🎯 Select Target Column",
        df_model.columns
    )

    X = df_model.drop(columns=[target])

    X = X.select_dtypes(include=np.number)

    y = df_model[target]

    # -----------------------------------------
    # Encoding
    # -----------------------------------------
    if task_type == "Classification":

        if y.dtype == "object":

            encoder = LabelEncoder()

            y = encoder.fit_transform(y)

    # -----------------------------------------
    # Missing Values
    # -----------------------------------------
    missing_count = X.isnull().sum().sum()

    if missing_count > 0:

        st.warning(
            f"Dataset contains {missing_count} missing values. Auto-imputation applied."
        )

    imputer = SimpleImputer(strategy="mean")

    X = imputer.fit_transform(X)

    # -----------------------------------------
    # Scaling
    # -----------------------------------------
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------
    # Split
    # -----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size/100,
        random_state=random_state
    )

    # -----------------------------------------
    # Architecture Card
    # -----------------------------------------
    st.info("""
    🧠 Stacking Architecture

    Base Learners:
    • Random Forest
    • Decision Tree
    • KNN
    • AdaBoost

    ↓

    Meta Learner

    • Logistic Regression (Classification)
    • Linear Regression (Regression)

    ↓

    Final Prediction
    """)

    # -----------------------------------------
    # Models
    # -----------------------------------------
    if task_type == "Classification":

        base_models = [

            ("rf",
             RandomForestClassifier(
                 random_state=random_state
             )),

            ("dt",
             DecisionTreeClassifier(
                 random_state=random_state
             )),

            ("knn",
             KNeighborsClassifier()),

            ("ada",
             AdaBoostClassifier(
                 random_state=random_state
             ))

        ]

        model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=cv_folds,
            n_jobs=-1
        )

        param_grid = {
            "final_estimator__C": [0.1, 1, 10]
        }

    else:

        base_models = [

            ("rf",
             RandomForestRegressor(
                 random_state=random_state
             )),

            ("dt",
             DecisionTreeRegressor(
                 random_state=random_state
             )),

            ("knn",
             KNeighborsRegressor()),

            ("ada",
             AdaBoostRegressor(
                 random_state=random_state
             ))

        ]

        model = StackingRegressor(
            estimators=base_models,
            final_estimator=LinearRegression(),
            n_jobs=-1
        )

        param_grid = {}

    # -----------------------------------------
    # Train
    # -----------------------------------------
    if st.button("🚀 Train Stacking Model"):

        if use_grid and task_type == "Classification":

            st.info("Running GridSearchCV...")

            grid = GridSearchCV(
                model,
                param_grid,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1
            )

            grid.fit(
                X_train,
                y_train
            )

            model = grid.best_estimator_

            st.success(
                "GridSearchCV Completed"
            )

            st.subheader(
                "🏆 Best Parameters"
            )

            st.json(
                grid.best_params_
            )

        else:

            model.fit(
                X_train,
                y_train
            )

        # -----------------------------------------
        # Prediction
        # -----------------------------------------
        y_pred = model.predict(
            X_test
        )

        st.markdown("---")
        st.header(
            "📈 Model Evaluation"
        )

        # -----------------------------------------
        # Classification
        # -----------------------------------------
        if task_type == "Classification":

            acc = accuracy_score(
                y_test,
                y_pred
            )

            st.success(
                f"🎯 Accuracy : {acc:.4f}"
            )

            cm = confusion_matrix(
                y_test,
                y_pred
            )

            st.subheader(
                "🧩 Confusion Matrix"
            )

            fig, ax = plt.subplots(
                figsize=(6,4)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="mako",
                ax=ax
            )

            st.pyplot(fig)

            st.subheader(
                "📄 Classification Report"
            )

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            st.dataframe(
                pd.DataFrame(report).transpose()
            )

            stack_score = acc

        # -----------------------------------------
        # Regression
        # -----------------------------------------
        else:

            mse = mean_squared_error(
                y_test,
                y_pred
            )

            rmse = np.sqrt(
                mse
            )

            r2 = r2_score(
                y_test,
                y_pred
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📉 MSE",
                    f"{mse:.4f}"
                )

            with col2:
                st.metric(
                    "📏 RMSE",
                    f"{rmse:.4f}"
                )

            with col3:
                st.metric(
                    "🎯 R² Score",
                    f"{r2:.4f}"
                )

            results = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred
            })

            st.subheader(
                "📊 Actual vs Predicted"
            )

            st.dataframe(
                results.head(15)
            )

            fig, ax = plt.subplots(
                figsize=(7,5)
            )

            ax.scatter(
                y_test,
                y_pred
            )

            ax.set_xlabel(
                "Actual"
            )

            ax.set_ylabel(
                "Predicted"
            )

            st.pyplot(fig)

            stack_score = r2

        # -----------------------------------------
        # Performance Arena
        # -----------------------------------------
        st.markdown("---")
        st.header(
            "🔥 Performance Arena"
        )

        comparison = []

        for name, mdl in base_models:

            mdl.fit(
                X_train,
                y_train
            )

            pred = mdl.predict(
                X_test
            )

            if task_type == "Classification":

                score = accuracy_score(
                    y_test,
                    pred
                )

            else:

                score = r2_score(
                    y_test,
                    pred
                )

            comparison.append(
                [name.upper(), score]
            )

        comparison.append(
            [
                "STACKING ENSEMBLE",
                stack_score
            ]
        )

        comparison_df = pd.DataFrame(
            comparison,
            columns=[
                "Model",
                "Score"
            ]
        )

        st.dataframe(
            comparison_df
        )

        fig, ax = plt.subplots(
            figsize=(10,5)
        )

        sns.barplot(
            data=comparison_df,
            x="Score",
            y="Model",
            ax=ax
        )

        st.pyplot(fig)

        # -----------------------------------------
        # Save Model
        # -----------------------------------------
        st.markdown("---")
        st.header(
            "💾 Save Trained Model"
        )

        model_name = (
            f"stacking_{task_type.lower()}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )

        model_path = os.path.join(
            MODEL_DIR,
            model_name
        )

        model_data = {
            "model": model,
            "scaler": scaler,
            "imputer": imputer,
            "features": list(
                df_model.drop(columns=[target])
                .select_dtypes(include=np.number)
                .columns
            )
        }

        with open(
            model_path,
            "wb"
        ) as f:

            pickle.dump(
                model_data,
                f
            )

        st.success(
            f"Model saved successfully in models/{model_name}"
        )

        log(
            f"Model saved : {model_path}"
        )