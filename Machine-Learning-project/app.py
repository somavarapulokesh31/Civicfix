"""
============================================================
CIVICFIX - MAIN FLASK APPLICATION
============================================================
NYC 311 Civic Complaint System
Run with: python app.py
============================================================
"""

import os
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ============static_folder=os.path.join(BASE_DIR, "static")
# ================================================
# PATHS
# ============================================================
BASE_DIR      = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project"
DATASET_PATH  = os.path.join(BASE_DIR, "dataset", "erm2-nwe9.csv")
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
DB_PATH       = os.path.join(BASE_DIR, "civicfix.db")
UPLOAD_DIR    = os.path.join(BASE_DIR, "uploads", "civic_issue_images")
EDA_DIR       = os.path.join(BASE_DIR, "outputs", "EDA_analysis")
CORR_DIR      = os.path.join(BASE_DIR, "outputs", "correlation")
PREPROC_DIR   = os.path.join(BASE_DIR, "outputs", "preprocessing")
MODEL_RES     = os.path.join(BASE_DIR, "outputs", "model_results")
DATASET_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "datasets")
os.makedirs(DATASET_UPLOAD_DIR, exist_ok=True)

from functools import wraps

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

# ============================================================
# AUTH HELPERS
# ============================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def manager_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login"))
        if session.get("role") != "manager":
            flash("Access restricted to managers.", "danger")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated

# ============================================================
# APP SETUP
# ============================================================
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = "civicfix_secret_key_2024_nyc311"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB — handles large CSV datasets

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# DATABASE HELPERS
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'user'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id      TEXT    UNIQUE NOT NULL,
            user_id        INTEGER,
            complaint_type TEXT    NOT NULL,
            description    TEXT,
            borough        TEXT,
            address        TEXT,
            priority       TEXT    DEFAULT 'Normal',
            image_path     TEXT,
            status         TEXT    DEFAULT 'Open',
            department     TEXT    DEFAULT '',
            assigned_to    TEXT    DEFAULT '',
            completed_at   TEXT    DEFAULT NULL,
            submitted_at   TEXT    NOT NULL
        )
    """)

    # Migrate existing DB: add columns if missing
    existing_cols = [row[1] for row in cur.execute("PRAGMA table_info(complaints)").fetchall()]
    for col, definition in [
        ("department",   "TEXT DEFAULT ''"),
        ("assigned_to",  "TEXT DEFAULT ''"),
        ("completed_at", "TEXT DEFAULT NULL"),
    ]:
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE complaints ADD COLUMN {col} {definition}")

    # Seed demo users
    demo_users = [
        ("manager", "manager@civicfix.nyc", generate_password_hash("manager123"), "manager"),
        ("user",    "user@civicfix.nyc",    generate_password_hash("user123"),    "user"),
    ]
    for uname, email, pw, role in demo_users:
        cur.execute(
            "INSERT OR IGNORE INTO users (username, email, password_hash, role) VALUES (?,?,?,?)",
            (uname, email, pw, role),
        )

    conn.commit()
    conn.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# DATASET STATS HELPER
# ============================================================
_dataset_cache = {}

def load_dataset_stats(path=None):
    global _dataset_cache
    if path is None:
        path = DATASET_PATH
    if path in _dataset_cache:
        return _dataset_cache[path]
    try:
        df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        df.drop(columns=unnamed, inplace=True)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        total_rows = len(df)
        total_cols = len(df.columns)
        columns    = list(df.columns)

        # Safe numeric counts — handle any CSV, not just CivicFix
        unique_complaint_types = int(df["complaint_type"].nunique()) if "complaint_type" in df.columns else 0
        unique_boroughs        = int(df["borough"].nunique())        if "borough"        in df.columns else 0
        unique_agencies        = int(df["agency"].nunique())         if "agency"         in df.columns else 0

        # Status distribution — safe for any CSV
        status_dist = {}
        if "status" in df.columns:
            try:
                status_dist = df["status"].value_counts().to_dict()
            except Exception:
                status_dist = {}

        # Top 10 complaint types — safe
        top10_complaint = []
        if "complaint_type" in df.columns:
            try:
                top10_complaint = (
                    df["complaint_type"].value_counts().head(10)
                    .reset_index()
                    .rename(columns={"complaint_type": "type", "count": "count"})
                    .to_dict("records")
                )
            except Exception:
                top10_complaint = []

        # Borough distribution — safe
        borough_dist = []
        if "borough" in df.columns:
            try:
                borough_dist = (
                    df["borough"].value_counts().reset_index()
                    .rename(columns={"borough": "borough", "count": "count"})
                    .to_dict("records")
                )
            except Exception:
                borough_dist = []

        # Missing value summary
        missing_counts = df.isnull().sum()
        missing_info = [
            {"column": col,
             "missing": int(missing_counts[col]),
             "pct": round(missing_counts[col] / total_rows * 100, 1) if total_rows > 0 else 0}
            for col in df.columns if missing_counts[col] > 0
        ]
        missing_info.sort(key=lambda x: x["missing"], reverse=True)

        # Numerical & categorical column summary
        num_cols  = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols  = [c for c in df.columns if c not in num_cols]

        # Column type info for generic CSV display
        col_info = []
        for col in df.columns:
            dtype   = str(df[col].dtype)
            n_miss  = int(df[col].isnull().sum())
            n_uniq  = int(df[col].nunique())
            col_info.append({
                "name":    col,
                "dtype":   dtype,
                "missing": n_miss,
                "unique":  n_uniq,
                "pct_missing": round(n_miss / total_rows * 100, 1) if total_rows > 0 else 0,
            })

        result = {
            "total_rows":             total_rows,
            "total_cols":             total_cols,
            "columns":                columns,
            "col_info":               col_info,
            "num_cols":               num_cols,
            "cat_cols":               cat_cols,
            "unique_complaint_types": unique_complaint_types,
            "unique_boroughs":        unique_boroughs,
            "unique_agencies":        unique_agencies,
            "status_dist":            status_dist,
            "top10_complaint":        top10_complaint,
            "borough_dist":           borough_dist,
            "missing_info":           missing_info,
            "filename":               os.path.basename(path),
        }
        _dataset_cache[path] = result
        return result

    except Exception as e:
        return {"error": str(e), "filename": os.path.basename(path) if path else ""}


# ============================================================
# PREDICTION HELPER  (rule-based, no model file required)
# ============================================================
NYPD_TYPES = {
    "Noise - Residential", "Noise - Commercial", "Noise - Street/Sidewalk",
    "Noise - Vehicle", "Noise - Helicopter", "Noise - Park",
    "Blocked Driveway", "Illegal Parking", "Derelict Vehicle",
    "Disorderly Youth", "Panhandling", "Graffiti",
}

HIGH_VOLUME_TYPES = {
    "HEAT/HOT WATER", "Heating", "PLUMBING", "PAINT/PLASTER",
    "Rodent", "Unsanitary Condition", "Dirty Conditions",
}

CLOSED_BOROUGHS = {"BROOKLYN", "QUEENS"}

def predict_status(complaint_type: str, borough: str, agency: str, priority: str):
    ct = (complaint_type or "").strip()
    b  = (borough or "").strip().upper()
    ag = (agency or "").strip().upper()
    pr = (priority or "Normal").strip()

    # Rule 1 – NYPD agency → In Progress
    if ag == "NYPD" or any(n.upper() in ct.upper() for n in NYPD_TYPES):
        confidence = 72
        explanation = (
            "NYPD-related complaints are actively investigated and typically "
            "remain 'In Progress' while officers follow up on the incident."
        )
        return "In Progress", confidence, explanation

    # Rule 2 – Noise complaints → Open
    if "noise" in ct.lower():
        confidence = 68
        explanation = (
            "Noise complaints are high in volume and often sit in an 'Open' "
            "queue until a field inspector can verify and close the case."
        )
        return "Open", confidence, explanation

    # Rule 3 – High-priority complaints → Open (escalated queue)
    if pr == "Urgent":
        confidence = 65
        explanation = (
            "Urgent complaints are immediately flagged in the system and "
            "placed in an 'Open' high-priority queue awaiting dispatch."
        )
        return "Open", confidence, explanation

    # Rule 4 – Brooklyn/Queens + high-volume types → Closed
    if b in CLOSED_BOROUGHS and any(hv.upper() in ct.upper() for hv in HIGH_VOLUME_TYPES):
        confidence = 78
        explanation = (
            f"{borough} has a high operational capacity for {complaint_type} complaints. "
            "Historical data shows these are resolved and closed quickly."
        )
        return "Closed", confidence, explanation

    # Rule 5 – Infrastructure / maintenance agencies → Closed
    if ag in {"DEP", "DSNY", "DOT", "DDC", "HPD", "DPR"}:
        confidence = 70
        explanation = (
            f"Complaints handled by {agency} (infrastructure/maintenance agencies) "
            "have a historically high closure rate based on the dataset."
        )
        return "Closed", confidence, explanation

    # Default
    confidence = 55
    explanation = (
        "Based on average resolution patterns across all complaint types and "
        "boroughs, this complaint is most likely to be resolved and marked 'Closed'."
    )
    return "Closed", confidence, explanation


# ============================================================
# DEPARTMENT AUTO-ASSIGNMENT
# ============================================================
DEPT_MAP = {
    # NYPD
    "Illegal Parking"           : "NYPD — Traffic Division",
    "Noise - Residential"       : "NYPD — Community Affairs",
    "Noise - Street/Sidewalk"   : "NYPD — Community Affairs",
    "Noise - Vehicle"           : "NYPD — Traffic Division",
    "Blocked Driveway"          : "NYPD — Traffic Division",
    "Derelict Vehicle"          : "NYPD — Traffic Division",
    "Graffiti"                  : "NYPD — Community Affairs",
    # Sanitation
    "UNSANITARY CONDITION"      : "Dept. of Sanitation (DSNY)",
    "Dirty Condition"           : "Dept. of Sanitation (DSNY)",
    # Transportation
    "Street Condition"          : "Dept. of Transportation (DOT)",
    "Street Light Condition"    : "Dept. of Transportation (DOT)",
    "Pothole"                   : "Dept. of Transportation (DOT)",
    "Sidewalk Condition"        : "Dept. of Transportation (DOT)",
    # Water / Environment
    "Water Maintenance"         : "Dept. of Environmental Protection (DEP)",
    # Housing
    "HEAT/HOT WATER"            : "Dept. of Housing Preservation (HPD)",
    "PLUMBING"                  : "Dept. of Housing Preservation (HPD)",
    "PAINT/PLASTER"             : "Dept. of Housing Preservation (HPD)",
    # Default
    "Other"                     : "Municipal Services",
}

def auto_assign_department(complaint_type: str) -> str:
    return DEPT_MAP.get(complaint_type, "Municipal Services")


# ============================================================
# ROUTES
# ============================================================

# ---------- HOME ----------
@app.route("/")
def home():
    # DB complaint counts
    conn = get_db()
    db_total  = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    db_closed = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Closed'").fetchone()[0]
    db_inprog = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='In Progress'").fetchone()[0]
    db_open   = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Open'").fetchone()[0]
    recent    = conn.execute(
        "SELECT * FROM complaints ORDER BY submitted_at DESC LIMIT 5"
    ).fetchall()
    conn.close()

    # Real dataset stats from erm2-nwe9.csv
    ds = load_dataset_stats()
    ds_total   = ds.get("total_rows", 0)
    ds_cols    = ds.get("total_cols", 0)
    ds_types   = ds.get("unique_complaint_types", 0)
    ds_boroughs= ds.get("unique_boroughs", 0)
    ds_agencies= ds.get("unique_agencies", 0)
    ds_closed  = ds.get("status_dist", {}).get("Closed", 0)
    ds_inprog  = ds.get("status_dist", {}).get("In Progress", 0)
    ds_open    = ds.get("status_dist", {}).get("Open", 0)

    return render_template(
        "home.html",
        # DB counts (for recent complaints section)
        total=db_total, closed=db_closed, inprog=db_inprog, open_c=db_open,
        recent=recent,
        # Dataset real values
        ds_total=ds_total, ds_cols=ds_cols,
        ds_types=ds_types, ds_boroughs=ds_boroughs, ds_agencies=ds_agencies,
        ds_closed=ds_closed, ds_inprog=ds_inprog, ds_open=ds_open,
    )


# ---------- ABOUT ----------
@app.route("/about")
def about():
    return render_template("about.html")


# ---------- CONTACT ----------
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        flash("Thank you for your message! We'll get back to you shortly.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")


# ---------- DATASET ----------
@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    # Handle dataset upload
    if request.method == "POST":
        if "dataset_file" in request.files:
            f = request.files["dataset_file"]
            if f and f.filename.endswith(".csv"):
                fname = secure_filename(f.filename)
                save_path = os.path.join(DATASET_UPLOAD_DIR, fname)
                f.save(save_path)
                flash(f"Dataset '{fname}' uploaded successfully!", "success")
                return redirect(url_for("dataset", selected=fname))
            else:
                flash("Only CSV files are supported.", "danger")

    # Available datasets: built-in + uploaded
    builtin = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]
    uploaded = [f for f in os.listdir(DATASET_UPLOAD_DIR) if f.endswith(".csv")]
    all_datasets = sorted(set(builtin + uploaded))

    # Which dataset is selected
    selected = request.args.get("selected", "erm2-nwe9.csv")
    if selected in builtin:
        active_path = os.path.join(DATASET_DIR, selected)
    elif selected in uploaded:
        active_path = os.path.join(DATASET_UPLOAD_DIR, selected)
    else:
        active_path = DATASET_PATH
        selected = "erm2-nwe9.csv"

    stats = load_dataset_stats(active_path)
    return render_template("dataset.html", stats=stats,
                           all_datasets=all_datasets, selected=selected)


# ---------- VISUALIZATION ----------
@app.route("/visualization")
def visualization():
    def get_imgs(folder):
        if not os.path.isdir(folder):
            return []
        return sorted(f for f in os.listdir(folder)
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))

    def label(filename):
        return (filename.replace("_", " ")
                        .replace(".png", "")
                        .replace(".jpg", "")
                        .title())

    eda_imgs   = get_imgs(EDA_DIR)
    corr_imgs  = get_imgs(CORR_DIR)
    prepr_imgs = get_imgs(PREPROC_DIR)
    model_imgs = get_imgs(MODEL_RES)

    sections = [
        {
            "id":    "eda",
            "group": "Dataset Exploration",
            "icon":  "fa-chart-bar",
            "title": "Exploratory Data Analysis (EDA)",
            "desc":  "Distribution charts, trends and patterns from the CivicFix dataset",
            "color": "#2563eb",
            "about": (
                "Exploratory Data Analysis (EDA) is the first step in any ML project. "
                "It helps us understand the shape, distribution and quality of the data before modelling. "
                "Here we analyse the NYC 311 dataset (50,000 records) to uncover patterns in complaint types, "
                "borough distributions, agency performance, submission channels and monthly trends. "
                "EDA reveals class imbalances, data gaps and feature relationships that guide preprocessing decisions."
            ),
            "keypoints": ["50,000 NYC 311 records", "6 boroughs", "149 complaint types",
                          "14 agencies", "Monthly trend analysis", "Status distribution"],
            "imgs": [(f, label(f), "eda_image") for f in eda_imgs],
        },
        {
            "id":    "corr",
            "group": "Dataset Exploration",
            "icon":  "fa-diagram-project",
            "title": "Correlation & Outlier Analysis",
            "desc":  "Correlation heatmap, boxplots and IQR-based outlier detection on numerical features",
            "color": "#7c3aed",
            "about": (
                "Correlation analysis measures the linear relationship between numerical features using "
                "Pearson's correlation coefficient (−1 to +1). A heatmap highlights which features "
                "are strongly related (e.g. latitude vs y_coordinate) helping avoid multicollinearity. "
                "Boxplots visualise the interquartile range (IQR) and identify outliers — data points "
                "that fall more than 1.5×IQR below Q1 or above Q3. "
                "Understanding outliers helps decide whether to cap, remove or keep extreme values."
            ),
            "keypoints": ["Pearson correlation", "Heatmap visualisation", "IQR outlier detection",
                          "Q1/Q3 boxplots", "Multicollinearity check", "Numerical features only"],
            "imgs": [(f, label(f), "corr_image") for f in corr_imgs],
        },
        {
            "id":    "preproc",
            "group": "Data Preparation",
            "icon":  "fa-gear",
            "title": "Preprocessing & Feature Engineering",
            "desc":  "Embedding encoding charts — mean target encoding for high-cardinality categorical columns",
            "color": "#f97316",
            "about": (
                "Preprocessing transforms raw data into a format suitable for ML models. "
                "These charts show the results of mean target encoding (embedding encoding): "
                "each category (e.g. agency, complaint_type, borough) is replaced by the historical "
                "average status value for that category. This encodes the relationship between a category "
                "and the target without creating thousands of one-hot columns. "
                "For example, NYPD has a high mean status (mostly closed/in-progress) while "
                "lesser-known agencies show different resolution patterns."
            ),
            "keypoints": ["Mean target encoding", "High-cardinality handling", "Borough closed rates",
                          "Agency performance", "Complaint type resolution", "Feature compression"],
            "imgs": [(f, label(f), "preproc_image") for f in prepr_imgs],
        },
    ]

    # Show each model as its own expandable section with its related charts.
    model_sections = [
        ("logistic_regression", "Classification & Model Evaluation", "Logistic Regression", "fa-chart-line", "#2563eb", ["logistic_regression_confusion_matrix.png", "logistic_regression_metrics.png", "logistic_regression_feature_importance.png"], "Status classification with a linear decision boundary."),
        ("decision_tree", "Classification & Model Evaluation", "Decision Tree", "fa-sitemap", "#16a34a", ["decision_tree_confusion_matrix.png", "decision_tree_metrics.png", "decision_tree_feature_importance.png"], "Status classification using interpretable feature splits."),
        ("random_forest", "Classification & Model Evaluation", "Random Forest", "fa-tree", "#15803d", ["random_forest_confusion_matrix.png", "random_forest_metrics.png", "random_forest_feature_importance.png"], "Ensemble classification using bagged decision trees."),
        ("gradient_boosting", "Classification & Model Evaluation", "Gradient Boosting", "fa-arrow-trend-up", "#7c3aed", ["gradient_boosting_confusion_matrix.png", "gradient_boosting_metrics.png", "gradient_boosting_feature_importance.png"], "Sequential boosting for complaint status classification."),
        ("adaboost", "Classification & Model Evaluation", "AdaBoost", "fa-bolt", "#f97316", ["adaboost_confusion_matrix.png", "adaboost_metrics.png", "adaboost_feature_importance.png"], "Adaptive boosting that gives more weight to misclassified records."),
        ("cv_all", "Classification & Model Evaluation", "Cross-Validation Model Comparison", "fa-rotate", "#0891b2", ["model_comparison.png", "classifier_accuracy_comparison.png", "classifier_f1_comparison.png"], "Classifier score comparisons from the saved evaluation results."),
        ("kmeans", "Clustering & Dimension Reduction", "K-Means Clustering", "fa-circle-nodes", "#0891b2", ["kmeans_elbow_silhouette.png", "kmeans_cluster_visualization.png", "kmeans_cluster_sizes.png"], "Cluster selection, projected complaint groups, and cluster sizes."),
        ("hierarchical", "Clustering & Dimension Reduction", "Hierarchical Clustering", "fa-diagram-project", "#be185d", ["hierarchical_dendrogram.png", "hierarchical_linkage_comparison.png", "hierarchical_cluster_visualization.png"], "Ward-linkage hierarchy, linkage scores, and cluster projection."),
        ("isolation_forest", "Clustering & Dimension Reduction", "Anomaly Detection (Isolation Forest)", "fa-triangle-exclamation", "#ef4444", ["anomaly_detection_pca.png", "anomaly_score_distribution.png", "anomaly_contamination_sensitivity.png"], "Anomaly projection, score distribution, and contamination sensitivity."),
        ("dbscan_raw", "Clustering & Dimension Reduction", "DBSCAN (Raw Geographic Data)", "fa-map-location-dot", "#d97706", ["dbscan_raw_geographic.png", "dbscan_raw_eps_sensitivity.png", "dbscan_raw_cluster_sizes.png"], "Geographic hotspots, epsilon sensitivity, and cluster sizes."),
        ("dbscan_preproc", "Clustering & Dimension Reduction", "DBSCAN (Pre-Processed Data)", "fa-filter", "#ca8a04", ["dbscan_preprocessed_clusters.png", "dbscan_preprocessed_eps_sensitivity.png", "dbscan_preprocessed_cluster_sizes.png"], "Feature-space clusters, epsilon sensitivity, and cluster sizes."),
        ("pca", "Clustering & Dimension Reduction", "Principal Component Analysis (PCA)", "fa-compress-arrows-alt", "#0284c7", ["pca_scree_plot.png", "pca_2d_scatter.png", "pca_feature_loadings.png"], "Explained variance, two-component projection, and feature loadings."),
        ("tsne", "Clustering & Dimension Reduction", "t-SNE", "fa-braille", "#7c3aed", ["tsne_perplexity_comparison.png", "tsne_by_agency.png", "tsne_by_status.png"], "Embeddings compared by perplexity, agency, and complaint status."),
    ]
    for section_id, group, title, icon, color, filenames, description in model_sections:
        available = [f for f in filenames if f in model_imgs]
        sections.append({
            "id": section_id,
            "group": group,
            "icon": icon,
            "title": title,
            "desc": description,
            "color": color,
            "about": description,
            "keypoints": [],
            "imgs": [(f, label(f), "model_image") for f in available],
        })
    total = sum(len(s["imgs"]) for s in sections)
    return render_template("visualization.html", sections=sections, total=total)

# ---------- SERVE IMAGES ----------
@app.route("/eda_image/<filename>")
def eda_image(filename):
    from flask import send_from_directory
    return send_from_directory(EDA_DIR, filename)

@app.route("/corr_image/<filename>")
def corr_image(filename):
    from flask import send_from_directory
    return send_from_directory(CORR_DIR, filename)

@app.route("/preproc_image/<filename>")
def preproc_image(filename):
    from flask import send_from_directory
    return send_from_directory(PREPROC_DIR, filename)

@app.route("/model_image/<filename>")
def model_image(filename):
    from flask import send_from_directory
    return send_from_directory(MODEL_RES, filename)


# ---------- ISSUES ----------
@app.route("/issues")
def issues():
    status_filter = request.args.get("status", "all")
    conn = get_db()
    if status_filter == "all":
        rows = conn.execute(
            "SELECT * FROM complaints ORDER BY submitted_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM complaints WHERE status=? ORDER BY submitted_at DESC",
            (status_filter,),
        ).fetchall()
    conn.close()
    return render_template(
        "issues.html", complaints=rows,
        status_filter=status_filter,
        role=session.get("role", "user"),
    )


# ---------- REPORT ----------
@app.route("/report", methods=["GET", "POST"])
def report():
    if request.method == "POST":
        complaint_type = request.form.get("complaint_type", "")
        description    = request.form.get("description", "")
        borough        = request.form.get("borough", "")
        address        = request.form.get("address", "")
        priority       = request.form.get("priority", "Normal")
        image_path     = None

        if "image" in request.files:
            file = request.files["image"]
            if file and file.filename and allowed_file(file.filename):
                filename   = secure_filename(file.filename)
                unique_name = f"{uuid.uuid4().hex}_{filename}"
                file.save(os.path.join(UPLOAD_DIR, unique_name))
                image_path = unique_name

        ticket_id    = f"NYC-{uuid.uuid4().hex[:8].upper()}"
        submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_id      = session.get("user_id")

        # AUTO-ASSIGN department based on complaint type
        department = auto_assign_department(complaint_type)

        conn = get_db()
        conn.execute(
            """INSERT INTO complaints
               (ticket_id, user_id, complaint_type, description, borough,
                address, priority, image_path, status, department, submitted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (ticket_id, user_id, complaint_type, description, borough,
             address, priority, image_path, "Open", department, submitted_at),
        )
        conn.commit()
        conn.close()

        flash(f"Complaint submitted! Ticket: {ticket_id} → Auto-assigned to: {department}", "success")
        return redirect(url_for("report"))

    return render_template("report_issue.html")


# ---------- PREDICT ----------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        complaint_type = request.form.get("complaint_type", "")
        borough        = request.form.get("borough", "")
        agency         = request.form.get("agency", "")
        priority       = request.form.get("priority", "Normal")

        predicted_status, confidence, explanation = predict_status(
            complaint_type, borough, agency, priority
        )
        result = {
            "status":      predicted_status,
            "confidence":  confidence,
            "explanation": explanation,
            "input": {
                "complaint_type": complaint_type,
                "borough":        borough,
                "agency":         agency,
                "priority":       priority,
            },
        }

    return render_template("predict.html", result=result)


# ---------- DASHBOARD ----------
@app.route("/dashboard")
def dashboard():
    conn  = get_db()
    total  = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
    closed = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Closed'").fetchone()[0]
    open_c = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Open'").fetchone()[0]
    inprog = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='In Progress'").fetchone()[0]
    recent = conn.execute(
        "SELECT * FROM complaints ORDER BY submitted_at DESC LIMIT 10"
    ).fetchall()
    # Top complaint types from DB
    top_types_db = conn.execute(
        "SELECT complaint_type, COUNT(*) as cnt FROM complaints "
        "GROUP BY complaint_type ORDER BY cnt DESC LIMIT 8"
    ).fetchall()
    # Borough breakdown from DB
    borough_db = conn.execute(
        "SELECT borough, COUNT(*) as cnt FROM complaints "
        "WHERE borough IS NOT NULL AND borough != '' "
        "GROUP BY borough ORDER BY cnt DESC"
    ).fetchall()
    conn.close()

    # Full dataset stats
    ds = load_dataset_stats()
    ds_total   = ds.get("total_rows", 0)
    ds_cols    = ds.get("total_cols", 0)
    ds_closed  = ds.get("status_dist", {}).get("Closed", 0)
    ds_open    = ds.get("status_dist", {}).get("Open", 0)
    ds_inprog  = ds.get("status_dist", {}).get("In Progress", 0)
    ds_missing = len(ds.get("missing_info", []))
    top10      = ds.get("top10_complaint", [])
    borough_dist = ds.get("borough_dist", [])
    agency_dist  = ds.get("agency_dist",  [])

    # Chart data — doughnut + bar
    ds_status_labels = ['Closed', 'In Progress', 'Open']
    ds_status_values = [int(ds_closed), int(ds_inprog), int(ds_open)]
    db_status_labels = ['Open', 'In Progress', 'Closed']
    db_status_values = [int(open_c), int(inprog), int(closed)]

    # Top complaint labels/values for bar chart (from dataset)
    top_labels = [r.get("type", r.get("complaint_type", "?")) for r in top10]
    top_values = [r.get("count", 0) for r in top10]

    # Borough chart (from dataset)
    bor_labels = [r.get("borough", "?") for r in borough_dist[:6]]
    bor_values = [r.get("count", 0) for r in borough_dist[:6]]

    return render_template(
        "dashboard.html",
        total=total, closed=closed, open_c=open_c, inprog=inprog,
        recent=recent,
        ds_total=ds_total, ds_cols=ds_cols,
        ds_closed=ds_closed, ds_open=ds_open, ds_inprog=ds_inprog,
        ds_missing=ds_missing,
        ds_status_labels=ds_status_labels, ds_status_values=ds_status_values,
        db_status_labels=db_status_labels, db_status_values=db_status_values,
        top_labels=top_labels, top_values=top_values,
        bor_labels=bor_labels, bor_values=bor_values,
        agency_dist=agency_dist,
    )


# ---------- REPORTS ----------
@app.route("/reports")
def reports():
    model_images = []
    model_csvs   = []
    model_results = []
    report_models = []
    if os.path.isdir(MODEL_RES):
        model_images = sorted(
            f for f in os.listdir(MODEL_RES)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        model_csvs = sorted(
            f for f in os.listdir(MODEL_RES)
            if f.lower().endswith(".csv")
        )
        # Parse all_model_results.csv if it exists
        csv_path = os.path.join(MODEL_RES, "all_model_results.csv")
        if os.path.isfile(csv_path):
            try:
                df_results = pd.read_csv(csv_path)
                model_results = df_results.to_dict("records")
            except Exception:
                pass
    metric_by_name = {str(row.get("Model", "")): row for row in model_results}
    report_catalog = [
        ("Logistic Regression", "M2", "Classification", ["logistic_regression_confusion_matrix.png"], ""),
        ("Decision Tree", "M3", "Classification", ["decision_tree_confusion_matrix.png"], ""),
        ("Random Forest", "M3", "Classification", ["random_forest_confusion_matrix.png"], ""),
        ("Gradient Boosting", "M3", "Classification", ["gradient_boosting_confusion_matrix.png"], ""),
        ("AdaBoost", "M3", "Classification", ["adaboost_confusion_matrix.png"], ""),
        ("Cross-Validation (All Classifiers)", "M3", "Evaluation", ["model_comparison.png"], "5-fold classifier evaluation; see the comparison chart for model results."),
        ("K-Means Clustering", "M4", "Clustering", ["kmeans_elbow_silhouette.png", "kmeans_cluster_visualization.png", "kmeans_cluster_sizes.png"], "Best K=5; silhouette score 0.478."),
        ("Hierarchical Clustering", "M4", "Clustering", ["hierarchical_dendrogram.png", "hierarchical_linkage_comparison.png", "hierarchical_cluster_visualization.png"], "Ward linkage, K=5; silhouette score 0.4832."),
        ("Anomaly Detection (Isolation Forest)", "M4", "Anomaly Detection", ["anomaly_detection_pca.png", "anomaly_score_distribution.png", "anomaly_contamination_sensitivity.png"], "5% contamination setting; anomaly counts and scores are shown in the charts."),
        ("DBSCAN (Raw Dataset)", "M4", "Clustering", ["dbscan_raw_geographic.png", "dbscan_raw_eps_sensitivity.png"], "Geographic clustering; eps=0.02; 4 clusters in the saved sample."),
        ("DBSCAN (Pre-Processed)", "M4", "Clustering", ["dbscan_preprocessed_clusters.png", "dbscan_preprocessed_eps_sensitivity.png"], "Feature-space clustering; eps=0.8; cluster and noise counts are shown in the charts."),
        ("PCA", "M4", "Dimensionality Reduction", ["pca_scree_plot.png", "pca_2d_scatter.png", "pca_feature_loadings.png"], "Two-component projection explains 93.9% of variance."),
        ("t-SNE", "M4", "Dimensionality Reduction", ["tsne_perplexity_comparison.png", "tsne_by_agency.png"], "Embedding comparison uses perplexity values 10, 30, and 50."),
    ]
    available_images = set(model_images)
    for name, module, model_type, charts, evaluation in report_catalog:
        report_models.append({
            "name": name,
            "module": module,
            "type": model_type,
            "metrics": metric_by_name.get(name, {}),
            "metric_label": "Held-out test metrics",
            "evaluation": evaluation,
            "charts": [filename for filename in charts if filename in available_images],
        })
    report_ranked_models = sorted(report_models, key=lambda model: (
        0 if model["name"] in metric_by_name else 1,
        -float(model["metrics"].get("Accuracy", -1) or -1),
    ))
    return render_template(
        "reports.html",
        model_images=model_images,
        model_csvs=model_csvs,
        model_results=model_results,
        report_models=report_models,
        report_ranked_models=report_ranked_models,
    )


# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role_req = request.form.get("role", "user")

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            if user["role"] != role_req:
                flash(f"This account is registered as '{user['role']}', not '{role_req}'.", "danger")
                return redirect(url_for("login"))
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


# ---------- REGISTER ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        role     = request.form.get("role", "user")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("register"))

        pw_hash = generate_password_hash(password)
        conn    = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, role) VALUES (?,?,?,?)",
                (username, email, pw_hash, role),
            )
            conn.commit()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
        finally:
            conn.close()

    return render_template("register.html")


# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ---------- ADMIN: UPDATE STATUS ----------
@app.route("/admin/update_status", methods=["POST"])
def update_status():
    if session.get("role") != "manager":
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    data       = request.get_json()
    ticket_id  = data.get("ticket_id")
    new_status = data.get("status")

    if new_status not in ("Open", "In Progress", "Closed"):
        return jsonify({"success": False, "error": "Invalid status"}), 400

    # Set completed_at timestamp when Closed, clear it otherwise
    completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if new_status == "Closed" else None

    conn = get_db()
    conn.execute(
        "UPDATE complaints SET status=?, completed_at=? WHERE ticket_id=?",
        (new_status, completed_at, ticket_id),
    )
    conn.commit()
    conn.close()

    return jsonify({
        "success"     : True,
        "ticket_id"   : ticket_id,
        "status"      : new_status,
        "completed_at": completed_at
    })


# ---------- PREPROCESSING ----------
@app.route("/preprocessing")
def preprocessing():
    # Show preprocessing steps info + datasets available
    datasets = []
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    if os.path.isdir(dataset_dir):
        datasets = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".csv"))

    steps = [
        {"num": "01", "name": "Load & Understand Dataset", "desc": "Load erm2-nwe9.csv (50,000 NYC 311 records), inspect shape, dtypes, missing values", "script": "load_understand_dataset.py", "output": "Console report", "status": "done"},
        {"num": "02", "name": "Handle Missing Values", "desc": "Drop columns >60% missing, fill numerical with mean, categorical with mode", "script": "clean_del_mean_model_missing_imputer_M2.py", "output": "clean_missing_values.csv", "status": "done"},
        {"num": "03", "name": "Label Encoding", "desc": "Encode high-cardinality columns: agency, complaint_type, descriptor, location_type", "script": "clean_label_encode_M2.py", "output": "clean_label_encoded.csv", "status": "done"},
        {"num": "04", "name": "One-Hot Encoding", "desc": "Binary encode low-cardinality: borough, open_data_channel_type, address_type", "script": "clean_one_hot_encode_M2.py", "output": "clean_one_hot_encoded.csv", "status": "done"},
        {"num": "05", "name": "Ordinal Encoding", "desc": "Encode status with natural order: Open=0 < In Progress=1 < Closed=2", "script": "clean_ordinal_encode_M2.py", "output": "clean_ordinal_encoded.csv", "status": "done"},
        {"num": "06", "name": "Target Encoding", "desc": "Replace categories with mean of target variable per category", "script": "clean_target_encode_M2.py", "output": "clean_target_encoded.csv", "status": "done"},
        {"num": "07", "name": "Embedding Encoding", "desc": "Mean target encoding for complaint_type, agency, borough — captures resolution history", "script": "clean_embedding_encode_M2.py", "output": "clean_embedding_encoded.csv", "status": "done"},
        {"num": "08", "name": "Feature Scaling", "desc": "StandardScaler applied to numerical columns: council_district, latitude, longitude", "script": "clean_minmax_stand_norma_M2.py", "output": "clean_scaled.csv", "status": "done"},
        {"num": "09", "name": "Final Preprocessing Pipeline", "desc": "Combined pipeline: select 8 features, encode, scale, save ML-ready dataset", "script": "final_preprocess_M2.py", "output": "final_preprocess_M2.csv", "status": "done"},
    ]

    # Check which output files actually exist
    for step in steps:
        out = step["output"]
        if out.endswith(".csv"):
            path = os.path.join(BASE_DIR, "dataset", out)
            step["exists"] = os.path.isfile(path)
            if step["exists"]:
                size = os.path.getsize(path)
                step["size"] = f"{size // 1024:,} KB"
            else:
                step["size"] = ""
        else:
            step["exists"] = True
            step["size"] = ""

    return render_template("preprocessing.html", steps=steps, datasets=datasets)


# ---------- ML MODELS ----------
@app.route("/models")
def models_page():
    import json

    models_info = [
        {"name": "Logistic Regression", "slug": "logistic_regression", "module": "M2",
         "type": "Classification", "desc": "Predicts complaint status probability using sigmoid/softmax function. Multinomial for 3-class prediction.",
         "features": "complaint_type, agency, borough, location_type, channel, council_district, lat, lng",
         "target": "status (Open=0, In Progress=1, Closed=2)"},
        {"name": "Decision Tree", "slug": "decision_tree", "module": "M3",
         "type": "Classification", "desc": "Splits data on feature thresholds to reach leaf-node class predictions. max_depth=10.",
         "features": "All 8 features", "target": "status (3 classes)"},
        {"name": "Random Forest", "slug": "random_forest", "module": "M3",
         "type": "Classification", "desc": "Ensemble of 100 decision trees with bagging. Reduces overfitting vs single tree.",
         "features": "All 8 features", "target": "status (3 classes)"},
        {"name": "Gradient Boosting", "slug": "gradient_boosting", "module": "M3",
         "type": "Classification", "desc": "Sequential trees each correcting residuals of the previous. learning_rate=0.1.",
         "features": "All 8 features", "target": "status (3 classes)"},
        {"name": "AdaBoost", "slug": "adaboost", "module": "M3",
         "type": "Classification", "desc": "Adaptive boosting of weak learners (depth=1 trees). Reweights misclassified samples.",
         "features": "All 8 features", "target": "status (3 classes)"},
        {"name": "Cross-Validation (All Classifiers)", "slug": "cv_all", "module": "M3",
         "type": "Evaluation", "desc": "5-Fold Stratified Cross-Validation on all classifiers. Accuracy ± std deviation shown.",
         "features": "All 8 features", "target": "status (3 classes)"},
        {"name": "K-Means Clustering", "slug": "kmeans", "module": "M4",
         "type": "Clustering", "desc": "Groups complaints into K clusters by similarity. Optimal K found via Elbow + Silhouette methods.",
         "features": "lat, lng, complaint_type, agency, borough, council_district", "target": "None (unsupervised)"},
        {"name": "Hierarchical Clustering", "slug": "hierarchical", "module": "M4",
         "type": "Clustering", "desc": "Agglomerative (bottom-up) and divisive (top-down bisection) clustering. Dendrogram analysis.",
         "features": "lat, lng, complaint_type, agency, borough, council_district", "target": "None (unsupervised)"},
        {"name": "Anomaly Detection (Isolation Forest)", "slug": "isolation_forest", "module": "M4",
         "type": "Anomaly Detection", "desc": "Detects unusual complaints that don't follow normal patterns. 5% contamination rate.",
         "features": "All 8 features", "target": "None (unsupervised)"},
        {"name": "DBSCAN (Raw Dataset)", "slug": "dbscan_raw", "module": "M4",
         "type": "Clustering", "desc": "Density-based geographic clustering on lat/lng. Finds NYC complaint hotspot regions.",
         "features": "latitude, longitude", "target": "None (unsupervised)"},
        {"name": "DBSCAN (Pre-Processed)", "slug": "dbscan_preproc", "module": "M4",
         "type": "Clustering", "desc": "DBSCAN on encoded + scaled dataset. Finds density-based clusters in feature space.",
         "features": "All encoded features", "target": "None (unsupervised)"},
        {"name": "PCA", "slug": "pca", "module": "M4",
         "type": "Dimensionality Reduction", "desc": "Reduces 8 features to 2-3 principal components. Scree plot + 2D scatter + feature loadings.",
         "features": "All 8 features", "target": "None (unsupervised)"},
        {"name": "t-SNE", "slug": "tsne", "module": "M4",
         "type": "Dimensionality Reduction", "desc": "Non-linear 2D visualisation of complaint clusters. Compares perplexity=10,30,50.",
         "features": "All 8 features", "target": "None (unsupervised)"},
    ]

    # Load results CSV if it exists
    results = []
    csv_path = os.path.join(BASE_DIR, "outputs", "model_results", "all_model_results.csv")
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
            results = df.to_dict("records")
        except Exception:
            pass

    # Get model result images
    model_res_dir = os.path.join(BASE_DIR, "outputs", "model_results")
    model_imgs = []
    if os.path.isdir(model_res_dir):
        model_imgs = sorted(f for f in os.listdir(model_res_dir) if f.endswith(".png"))

    # Check which pkl files exist
    models_dir = os.path.join(BASE_DIR, "models")
    for m in models_info:
        pkl = os.path.join(models_dir, f"{m['slug']}.pkl")
        m["trained"] = os.path.isfile(pkl)

    # Map model slug → its result images
    MODEL_CHART_MAP = {
        "logistic_regression": ["logistic_regression_confusion_matrix.png", "logistic_regression_metrics.png", "logistic_regression_feature_importance.png"],
        "decision_tree"       : ["decision_tree_confusion_matrix.png", "decision_tree_metrics.png", "decision_tree_feature_importance.png"],
        "random_forest"       : ["random_forest_confusion_matrix.png", "random_forest_metrics.png", "random_forest_feature_importance.png"],
        "gradient_boosting"   : ["gradient_boosting_confusion_matrix.png", "gradient_boosting_metrics.png", "gradient_boosting_feature_importance.png"],
        "adaboost"            : ["adaboost_confusion_matrix.png", "adaboost_metrics.png", "adaboost_feature_importance.png"],
        "cv_all"              : ["model_comparison.png", "classifier_accuracy_comparison.png", "classifier_f1_comparison.png"],
        "kmeans"              : ["kmeans_elbow_silhouette.png",
                                 "kmeans_cluster_visualization.png",
                                 "kmeans_cluster_sizes.png"],
        "hierarchical"        : ["hierarchical_dendrogram.png",
                                 "hierarchical_linkage_comparison.png",
                                 "hierarchical_cluster_visualization.png"],
        "isolation_forest"    : ["anomaly_detection_pca.png",
                                 "anomaly_score_distribution.png",
                                 "anomaly_contamination_sensitivity.png"],
        "dbscan_raw"          : ["dbscan_raw_geographic.png",
                                 "dbscan_raw_eps_sensitivity.png",
                                 "dbscan_raw_cluster_sizes.png"],
        "dbscan_preproc"      : ["dbscan_preprocessed_clusters.png",
                                 "dbscan_preprocessed_eps_sensitivity.png",
                                 "dbscan_preprocessed_cluster_sizes.png"],
        "pca"                 : ["pca_scree_plot.png",
                                 "pca_2d_scatter.png",
                                 "pca_feature_loadings.png"],
        "tsne"                : ["tsne_perplexity_comparison.png",
                                 "tsne_by_agency.png",
                                 "tsne_by_status.png"],
    }
    MODEL_EVALUATION = {
        "cv_all": "5-fold stratified cross-validation; fold scores and variation are shown in the charts.",
        "kmeans": "5 clusters; best silhouette score 0.478 (15,000-row sample).",
        "hierarchical": "Ward linkage with 5 clusters; silhouette score 0.4832.",
        "isolation_forest": "5% contamination setting; anomaly counts and score distribution are shown in the charts.",
        "dbscan_raw": "4 geographic clusters and 3 noise points in the 15,000-row sample (eps=0.02).",
        "dbscan_preproc": "1 cluster and 1 noise point in the 10,000-row sample (eps=0.8).",
        "pca": "Two-component projection explains 93.9% of variance; component loadings and scree plot are included.",
        "tsne": "Embedding comparisons use perplexity values 10, 30, and 50; see charts for results.",
    }
    results_by_model = {str(row.get("Model", "")): row for row in results}
    proxy_path = os.path.join(model_res_dir, "model_proxy_results.csv")
    if os.path.isfile(proxy_path):
        try:
            proxy_df = pd.read_csv(proxy_path)
            proxy_by_model = {str(row.get("Model", "")): row for row in proxy_df.to_dict("records")}
            results_by_model.update(proxy_by_model)
        except Exception:
            proxy_by_model = {}
    else:
        proxy_by_model = {}
    # Only include charts that actually exist on disk
    for m in models_info:
        slug   = m["slug"]
        charts = MODEL_CHART_MAP.get(slug, [])
        m["metrics"] = results_by_model.get(m["name"], {})
        m["metric_label"] = "Status-prediction proxy (held-out sample)" if slug in proxy_by_model else "Held-out test metrics"
        m["evaluation"] = MODEL_EVALUATION.get(slug, "")
        m["charts"] = [
            c for c in charts
            if os.path.isfile(os.path.join(model_res_dir, c))
        ]

    best_model = max(results, key=lambda x: x.get("Accuracy", 0)) if results else None

    return render_template("models.html", models_info=models_info, results=results,
                           model_imgs=model_imgs, best_model=best_model)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    init_db()
    print("=" * 55)
    print("  CivicFix - NYC 311 Civic Complaint System")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
