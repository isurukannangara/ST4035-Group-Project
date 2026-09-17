import streamlit as st
import pandas as pd
import plotly.express as px

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Missing Data & Neural Networks",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM LIGHT DASHBOARD STYLE
# ============================================================

st.markdown("""
<style>

    .stApp {
        background-color: #F4F6FB;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #64748B;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 25px;
        margin-bottom: 15px;
        padding-left: 10px;
        border-left: 4px solid #2563EB;
    }

    .sub-section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1E293B;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #2563EB;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
        text-align: center;
    }

    .metric-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        color: #64748B;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #1D4ED8;
    }

    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid var(--accent, #2563EB);
        border-radius: 10px;
        padding: 22px;
        min-height: 150px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }

    .info-card h3 {
        color: #1E293B;
        margin-top: 0;
        font-size: 19px;
    }

    .info-card p {
        color: #475569;
        line-height: 1.6;
        font-size: 14.5px;
    }

    .finding-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    .finding-card b {
        color: #1E293B;
    }

    .finding-card span {
        color: #475569;
        font-size: 14.5px;
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #0F172A !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #0F172A !important;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD RESULTS
# ============================================================

RESULTS_FILE = "results/metrics.csv"
RAW_DATA_FILE = "data/raw/credit_card_default.csv"

df = pd.read_csv(RESULTS_FILE)
raw_df = pd.read_csv(RAW_DATA_FILE)

# Metrics that may be present in metrics.csv, in order of priority.
# The app only shows the ones that actually exist, so it works whether
# metrics.csv has 2 columns or the full set of 5.
METRIC_PRIORITY = ["auc_roc", "mcc", "macro_f1", "auc_pr", "brier_score"]
METRIC_LABELS = {
    "auc_roc": "AUC-ROC",
    "mcc": "MCC",
    "macro_f1": "Macro-F1",
    "auc_pr": "AUC-PR",
    "brier_score": "Brier Score"
}
LOWER_IS_BETTER = {"brier_score"}

AVAILABLE_METRICS = [m for m in METRIC_PRIORITY if m in df.columns]

METHOD_LABELS = {
    "mean_mode": "Mean/Mode",
    "knn": "KNN",
    "mice": "MICE",
    "missforest": "MissForest"
}
METHOD_COLORS = {
    "Mean/Mode": "#2563EB",
    "KNN": "#059669",
    "MICE": "#D97706",
    "MissForest": "#7C3AED"
}

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div style="font-size:22px; font-weight:700; color:#0F172A; '
    'margin-bottom:20px;">📊 Project Dashboard</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigate to",
    [
        "Project Overview",
        "Dataset & EDA",
        "Methodology",
        "Results Dashboard",
        "Overall Results"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **ST4035 – Data Science Project**

    *Missing Data Handling in Neural Networks*

    Dataset: UCI Default of Credit Card Clients
    """
)

# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.markdown(
        '<div class="main-title">Missing Data Handling in Neural Networks</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'This project studies how different missing-data mechanisms and imputation '
        'methods affect neural network performance using the UCI Default of Credit Card Clients dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Study Overview</div>', unsafe_allow_html=True)

    st.write(
        "The original dataset has no missing values. Missing values were introduced "
        "artificially at 10%, 20%, and 30% using MCAR, MAR, and MNAR. Four imputation "
        "methods were then applied, and their effect on neural network performance was "
        "compared with the clean-data baseline."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    overview_cards = [
        (c1, "Observations", "30,000"),
        (c2, "Features", "23"),
        (c3, "Missing Mechanisms", "3"),
        (c4, "Imputation Methods", "4"),
    ]
    for col, label, value in overview_cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Missing Data Mechanisms</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    mechanism_cards = [
        (c1, "#2563EB", "MCAR", "Missing Completely At Random",
         "Missingness was generated randomly without depending on the observed data values."),
        (c2, "#059669", "MAR", "Missing At Random",
         "Missingness was generated based on other observed variables."),
        (c3, "#D97706", "MNAR", "Missing Not At Random",
         "Missingness was generated based on the values of the variable itself."),
    ]
    for col, accent, title, subtitle_text, body in mechanism_cards:
        with col:
            st.markdown(f"""
            <div class="info-card" style="--accent: {accent};">
                <h3>{title}</h3>
                <p><b>{subtitle_text}</b></p>
                <p>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Imputation Methods</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    methods = [
        (c1, "#2563EB", "Mean/Mode", "Mean for numerical variables and mode for categorical variables."),
        (c2, "#059669", "KNN", "k=5 nearest-neighbour imputation using the selected features."),
        (c3, "#D97706", "MICE", "Chained-equations imputation using a LightGBM backend (miceforest)."),
        (c4, "#7C3AED", "MissForest", "Iterative Random Forest imputation via scikit-learn's IterativeImputer."),
    ]
    for col, accent, name, description in methods:
        with col:
            st.markdown(f"""
            <div class="info-card" style="--accent: {accent};">
                <h3>{name}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Study Design</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    study_design = [
        (
            c1,
            "Missingness Levels",
            "10%, 20%, and 30%"
        ),
        (
            c2,
            "Evaluation Metrics",
            "AUC-ROC, AUC-PR, Macro-F1, MCC, and Brier Score"
        ),
        (
            c3,
            "Prediction Model",
            "Feed-forward Neural Network"
        )
    ]

    for col, title, value in study_design:
        with col:
            st.markdown(f"""
            <div class="info-card" style="--accent: #2563EB; min-height: 120px;">
                <h3>{title}</h3>
                <p>{value}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        ST4035 Data Science Project
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATASET & EDA
# ============================================================

elif page == "Dataset & EDA":

    st.markdown(
        '<div class="main-title">Dataset & Exploratory Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Overview of the UCI Default of Credit Card Clients dataset and key findings from the exploratory analysis'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Dataset Summary</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    summary_cards = [
        (c1, "Observations", "30,000"),
        (c2, "Predictor Variables", "23"),
        (c3, "Target Variable", "1"),
        (c4, "Original Missing Values", "0"),
    ]

    for col, label, value in summary_cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.write(
        "The dataset contains information about credit card clients, including "
        "credit limits, demographic characteristics, repayment status, bill "
        "amounts and payment amounts. The original dataset contains no missing values."
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Class Distribution</div>',
        unsafe_allow_html=True
    )

    target = "default payment next month"

    class_counts = raw_df[target].value_counts().sort_index()

    non_default_pct = class_counts.get(0, 0) / len(raw_df) * 100
    default_pct = class_counts.get(1, 0) / len(raw_df) * 100

    c1, c2 = st.columns([1, 2])

    with c1:

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:14px;">
            <div class="metric-label">Non-Default</div>
            <div class="metric-value">{non_default_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Default</div>
            <div class="metric-value" style="color:#D97706;">
                {default_pct:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        class_df = pd.DataFrame({
            "Class": ["Non-Default (0)", "Default (1)"],
            "Share": [non_default_pct, default_pct]
        })

        fig_class = px.bar(
            class_df,
            x="Class",
            y="Share",
            text="Share",
            color="Class",
            color_discrete_map={
                "Non-Default (0)": "#2563EB",
                "Default (1)": "#D97706"
            }
        )

        fig_class.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            textfont_color="#0F172A"
        )

        fig_class.update_layout(
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            font=dict(color="#0F172A"),
            yaxis_title="Share of Observations (%)",
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(
            fig_class,
            use_container_width=True
        )

    st.markdown(
        f"""
        <div class="finding-card">
            <b>Interpretation:</b>
            <span>
            The target is imbalanced, with approximately {non_default_pct:.1f}% 
            non-default observations and {default_pct:.1f}% default observations.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # TARGET CORRELATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Target Correlation</div>',
        unsafe_allow_html=True
    )

    numeric_df = raw_df.select_dtypes(include="number")

    correlations = (
        numeric_df.corr()[target]
        .drop(target)
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .head(7)
        .sort_values()
    )

    corr_df = correlations.reset_index()
    corr_df.columns = ["Feature", "Correlation"]

    fig_corr = px.bar(
        corr_df,
        x="Correlation",
        y="Feature",
        orientation="h",
        text="Correlation"
    )

    fig_corr.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig_corr.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0F172A"),
        xaxis_title="Pearson Correlation",
        yaxis_title="Feature",
        margin=dict(t=20, b=20)
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    st.markdown(
        """
        <div class="finding-card">
            <b>Interpretation:</b>
            <span>
            PAY_0, PAY_2 and PAY_3 have the strongest correlations with the
            default outcome among the variables examined. These are associations
            with the target and do not imply causation.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # FEATURE DISTRIBUTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Feature Distributions</div>',
        unsafe_allow_html=True
    )

    distribution_features = [
        "LIMIT_BAL",
        "BILL_AMT1",
        "PAY_AMT1"
    ]

    dist_df = raw_df[distribution_features].melt(
        var_name="Feature",
        value_name="Value"
    )

    fig_dist = px.histogram(
        dist_df,
        x="Value",
        facet_col="Feature",
        facet_col_wrap=3,
        nbins=40,
        labels={
            "Value": "Value",
            "Feature": "Feature"
        }
    )

    fig_dist.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0F172A"),
        showlegend=False,
        margin=dict(t=40, b=20)
    )

    fig_dist.for_each_annotation(
        lambda a: a.update(text=a.text.replace("Feature=", ""))
    )

    st.plotly_chart(
        fig_dist,
        use_container_width=True
    )

    st.markdown(
        """
        <div class="finding-card">
            <b>Interpretation:</b>
            <span>
            The monetary variables show right-skewed distributions, with many
            observations concentrated at lower values and fewer observations at
            higher values.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CATEGORICAL VARIABLES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Categorical Variables</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    # Education
    with c1:

        education_counts = (
            raw_df["EDUCATION"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        education_counts.columns = ["Education", "Count"]

        fig_edu = px.bar(
            education_counts,
            x="Education",
            y="Count",
            text="Count"
        )

        fig_edu.update_traces(
            textposition="outside"
        )

        fig_edu.update_layout(
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0F172A"),
            title="Education Codes",
            showlegend=False,
            margin=dict(t=50, b=20)
        )

        st.plotly_chart(
            fig_edu,
            use_container_width=True
        )

    # Marriage
    with c2:

        marriage_counts = (
            raw_df["MARRIAGE"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        marriage_counts.columns = ["Marriage", "Count"]

        fig_marriage = px.bar(
            marriage_counts,
            x="Marriage",
            y="Count",
            text="Count"
        )

        fig_marriage.update_traces(
            textposition="outside"
        )

        fig_marriage.update_layout(
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0F172A"),
            title="Marriage Codes",
            showlegend=False,
            margin=dict(t=50, b=20)
        )

        st.plotly_chart(
            fig_marriage,
            use_container_width=True
        )

    st.markdown(
        """
        <div class="finding-card">
            <b>Interpretation:</b>
            <span>
            The EDA identified some undocumented category codes in the dataset,
            including 0, 5 and 6 in EDUCATION and 0 in MARRIAGE.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KEY EDA FINDINGS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Key EDA Findings</div>',
        unsafe_allow_html=True
    )

    findings = [
        (
            "Original missingness",
            "The original dataset contains no missing values."
        ),
        (
            "Bill amount variables",
            "BILL_AMT1 to BILL_AMT6 are highly intercorrelated."
        )
    ]

    for title, detail in findings:
        st.markdown(
            f"""
            <div class="finding-card">
                <b>{title}:</b>
                <span>{detail}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # FEATURES SELECTED FOR MISSINGNESS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Features Selected for Missingness</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Missingness was introduced only into these six features. "
        "The target variable was not modified."
    )

    missing_features = [
        "LIMIT_BAL",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "BILL_AMT1",
        "PAY_AMT1"
    ]

    feature_cols = st.columns(6)

    for col, feature in zip(feature_cols, missing_features):

        with col:

            st.markdown(
                f"""
                <div class="metric-card" style="padding:12px;">
                    <div class="metric-label" style="font-size:12px;">
                        {feature}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        """
        <div class="footer">
            ST4035 Data Science Project
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology":

    st.title("Methodology")

    st.write(
        "This section describes how missing data was generated, "
        "how it was imputed, and how the neural network was trained "
        "and evaluated."
    )

    st.markdown("---")

    # ========================================================
    # 1. MISSING DATA GENERATION
    # ========================================================

    st.subheader("1. Missing Data Generation")

    st.write(
        "Missing values were artificially introduced into six selected "
        "features using three missing-data mechanisms."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### MCAR")
            st.markdown("**Missing Completely At Random**")
            st.write(
                "Values were selected randomly without depending "
                "on the observed data values."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### MAR")
            st.markdown("**Missing At Random**")
            st.write(
                "Missingness was generated using other observed "
                "variables. For example, missingness in LIMIT_BAL "
                "was related to AGE and PAY_0."
            )

    with col3:
        with st.container(border=True):
            st.markdown("### MNAR")
            st.markdown("**Missing Not At Random**")
            st.write(
                "Missingness was related to the value of the variable "
                "itself, with higher values having a higher probability "
                "of being missing."
            )

    st.info(
        "Each mechanism was applied at 10%, 20%, and 30% missingness, "
        "producing 9 missing-data datasets."
    )

    # ========================================================
    # FEATURES SELECTED
    # ========================================================

    st.subheader("Features Selected for Missingness")

    st.write(
        "Missingness was introduced only into these six features. "
        "The target variable was not modified."
    )

    features = [
        "LIMIT_BAL",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "BILL_AMT1",
        "PAY_AMT1"
    ]

    feature_cols = st.columns(6)

    for col, feature in zip(feature_cols, features):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        font-weight: 600;
                        color: #315d8c;
                        padding: 8px 0;
                        letter-spacing: 0.5px;
                    ">
                        {feature}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ========================================================
    # 2. IMPUTATION
    # ========================================================

    st.subheader("2. Imputation")

    st.write(
        "Four imputation methods were applied separately to each "
        "missing-data dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.markdown("### Mean/Mode")
            st.write(
                "Mean was used for numerical and ordinal features, "
                "while the most frequent value was used for nominal "
                "categorical features."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### KNN")
            st.write(
                "Missing values were estimated using the 5 nearest "
                "neighbours."
            )

    with col3:
        with st.container(border=True):
            st.markdown("### MICE")
            st.write(
                "Missing values were estimated iteratively using "
                "the selected features."
            )

    with col4:
        with st.container(border=True):
            st.markdown("### MissForest")
            st.write(
                "Random forest models were used to estimate the "
                "missing values."
            )

    st.info(
        "Applying the four methods to all 9 missing-data datasets "
        "produced 36 imputed datasets."
    )

    # ========================================================
    # 3. NEURAL NETWORK
    # ========================================================

    st.subheader("3. Neural Network & Training Setup")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### Architecture")

            st.write(
                "Input → Dense(64, ReLU) → Dropout(0.3) → "
                "Dense(32, ReLU) → Dropout(0.3) → "
                "Dense(1, Sigmoid)"
            )

            st.write(
                "The same neural network architecture was used for "
                "the clean baseline and all 36 imputed datasets."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### Training Configuration")

            st.write(
                "**Optimizer:** Adam (learning rate = 0.001)"
            )

            st.write(
                "**Loss:** Binary cross-entropy"
            )

            st.write(
                "**Batch size:** 64"
            )

            st.write(
                "**Maximum epochs:** 50"
            )

            st.write(
                "**Early stopping:** patience = 5"
            )

    st.markdown("---")

    # ========================================================
    # 4. TRAIN / TEST SPLIT
    # ========================================================

    st.subheader("4. Data Splitting & Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### Train / Test Split")

            st.write(
                "The data was divided into 80% training data and "
                "20% test data using stratified sampling."
            )

            st.write(
                "The same split was used across the experiments "
                "to make the comparisons consistent."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### Data Leakage Prevention")

            st.write(
                "Imputation methods were fitted using the training "
                "data and then applied to the dataset."
            )

            st.write(
                "This prevents information from the test data being "
                "used when estimating missing values."
            )

    # ========================================================
    # 5. PERFORMANCE METRICS
    # ========================================================

    st.subheader("5. Performance Metrics")

    st.write(
        "The neural network predictions were evaluated using "
        "five performance measures."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    metrics = [
        ("AUC-ROC", "Measures the ability to distinguish between default and non-default cases."),
        ("AUC-PR", "Measures performance using precision and recall."),
        ("Macro-F1", "Balances precision and recall across both classes."),
        ("MCC", "Measures the quality of binary classification predictions."),
        ("Brier Score", "Measures the accuracy of predicted probabilities.")
    ]

    for col, (name, description) in zip(
        [col1, col2, col3, col4, col5],
        metrics
    ):
        with col:
            with st.container(border=True):
                st.markdown(f"### {name}")
                st.write(description)

    st.info(
        "The clean dataset was used as the baseline, and the results "
        "from the 36 imputed datasets were compared with this baseline."
    )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    st.caption("ST4035 Data Science Project")
# ============================================================
# RESULTS DASHBOARD
# ============================================================

elif page == "Results Dashboard":

    st.markdown('<div class="main-title">Results Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        'Explore neural network performance under different missing-data conditions'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Experiment Selection</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        mechanism = st.selectbox("Missing Data Mechanism", ["MCAR", "MAR", "MNAR"])

    with col2:
        level = st.selectbox("Missingness Level", [10, 20, 30])

    with col3:
        metric = st.selectbox(
            "Evaluation Metric",
            [METRIC_LABELS[m] for m in AVAILABLE_METRICS]
        )

    metric_column = [m for m in AVAILABLE_METRICS if METRIC_LABELS[m] == metric][0]
    lower_is_better = metric_column in LOWER_IS_BETTER

    filtered = df[
        (df["mechanism"].str.upper() == mechanism) &
        (df["level"] == level / 100)
    ].copy()

    filtered = filtered[filtered["imputation_method"] != "none"]

    baseline = df[df["mechanism"] == "baseline"].iloc[0]
    baseline_value = baseline[metric_column]

    st.markdown("---")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown(f'<div class="section-title">{metric} Comparison</div>', unsafe_allow_html=True)

        chart_data = filtered.copy()
        chart_data["imputation_method"] = chart_data["imputation_method"].replace(METHOD_LABELS)

        chart = px.bar(
            chart_data,
            x="imputation_method",
            y=metric_column,
            text=metric_column,
            color="imputation_method",
            color_discrete_map=METHOD_COLORS,
            labels={"imputation_method": "Imputation Method", metric_column: metric}
        )
        chart.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        chart.update_layout(
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            font=dict(color="#0F172A"),
            margin=dict(t=20, b=20)
        )
        if not lower_is_better:
            chart.update_layout(yaxis_range=[
                max(0, filtered[metric_column].min() - 0.05),
                min(1, filtered[metric_column].max() + 0.05)
            ])
        chart.update_traces(textfont_color="#0F172A")
        st.plotly_chart(chart, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Baseline</div>', unsafe_allow_html=True)
        st.metric(f"Clean Data {metric}", f"{baseline_value:.4f}")
        st.write(
            "The baseline represents the neural network trained on the original "
            "dataset without artificially introduced missing values."
        )
        if lower_is_better:
            st.caption("Lower is better for this metric.")

    st.markdown('<div class="section-title">Detailed Results</div>', unsafe_allow_html=True)

    table_cols = ["imputation_method"] + AVAILABLE_METRICS
    display_df = filtered[table_cols].copy()
    display_df["imputation_method"] = display_df["imputation_method"].replace(METHOD_LABELS)
    display_df.columns = ["Imputation Method"] + [METRIC_LABELS[m] for m in AVAILABLE_METRICS]

    for m in AVAILABLE_METRICS:
        display_df[METRIC_LABELS[m]] = display_df[METRIC_LABELS[m]].round(4)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================
# OVERALL RESULTS
# ============================================================

elif page == "Overall Results":

    st.markdown('<div class="main-title">Overall Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        'Performance trends across missing-data mechanisms and missingness levels'
        '</div>',
        unsafe_allow_html=True
    )

    results_df = df[df["mechanism"] != "baseline"].copy()
    results_df["Missingness"] = (results_df["level"] * 100).astype(int)

    trend_metric = st.selectbox(
        "Metric to plot",
        [METRIC_LABELS[m] for m in AVAILABLE_METRICS],
        key="trend_metric"
    )
    trend_col = [m for m in AVAILABLE_METRICS if METRIC_LABELS[m] == trend_metric][0]

    st.markdown(f'<div class="section-title">{trend_metric} Performance</div>', unsafe_allow_html=True)

    method_colors_raw = {
        "mean_mode": "#2563EB",
        "knn": "#059669",
        "mice": "#D97706",
        "missforest": "#7C3AED"
    }

    fig_trend = px.line(
        results_df,
        x="Missingness",
        y=trend_col,
        color="imputation_method",
        color_discrete_map=method_colors_raw,
        facet_col="mechanism",
        markers=True,
        labels={
            trend_col: trend_metric,
            "Missingness": "Missingness (%)",
            "imputation_method": "Imputation Method"
        }
    )
    fig_trend.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0F172A")
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown('<div class="section-title">Best Method by Condition</div>', unsafe_allow_html=True)
    st.write(f"Best-performing imputation method per mechanism × level, ranked by {trend_metric}.")

    rank_rows = []
    for mech in ["mcar", "mar", "mnar"]:
        for level_pct in [10, 20, 30]:
            subset = results_df[
                (results_df["mechanism"] == mech) &
                (results_df["Missingness"] == level_pct)
            ]
            if subset.empty:
                continue
            if trend_col in LOWER_IS_BETTER:
                best_row = subset.loc[subset[trend_col].idxmin()]
            else:
                best_row = subset.loc[subset[trend_col].idxmax()]
            rank_rows.append({
                "Mechanism": mech.upper(),
                "Missingness": f"{level_pct}%",
                "Best Method": METHOD_LABELS.get(best_row["imputation_method"], best_row["imputation_method"]),
                trend_metric: round(best_row[trend_col], 4)
            })

    st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="footer">
        ST4035 Data Science Project
    </div>
    """, unsafe_allow_html=True)