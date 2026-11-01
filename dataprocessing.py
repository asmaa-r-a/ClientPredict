from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TARGET = "Churn"

# الأعمدة الرقمية والوصفية الأساسية
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlyCharge", "TotalServices"]
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

def _resolve_data_path() -> Path:
    if DATA_DIR.exists():
        for candidate in DATA_DIR.iterdir():
            if candidate.is_file() and candidate.name.lower() == "telco_churn.csv":
                return candidate
    return DATA_DIR / "telco_Churn.csv"

DATA_PATH = _resolve_data_path()

def load_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"], errors="ignore")
    
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
    if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
        
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    # --- إضافة الميزات الذكية الجديدة لرفع الدقة (Feature Engineering) ---
    # 1. متوسط تكلفة الاستهلاك الفعلي نسبة لمدة الاشتراك
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    
    # 2. حساب إجمالي عدد الخدمات الإضافية التي يشترك بها العميل (كلما زادت الخدمات، قل احتمال مغادرته)
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["TotalServices"] = 0
    for col in service_cols:
        if col in df.columns:
            df["TotalServices"] += df[col].apply(lambda x: 1 if str(x).lower() in ["yes", "1"] else 0)

    return df

def split_xy(df):
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return X, y

def build_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])