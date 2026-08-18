from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="HR Pulse | Attrition Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hr_attrition_pipeline.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

CV_RESULTS_PATH = REPORTS / "cv_results.csv"
TEST_RESULTS_PATH = REPORTS / "test_results.csv"


# ============================================================
# PREMIUM UI
# ============================================================

st.html(
    """
    <style>

    /* ======================================================
       GLOBAL
    ====================================================== */

    :root {
        --navy: #0f172a;
        --navy-2: #172033;
        --blue: #4f46e5;
        --blue-2: #6366f1;
        --purple: #7c3aed;
        --green: #059669;
        --red: #dc2626;
        --amber: #d97706;
        --text: #172033;
        --muted: #64748b;
        --border: #e5e7eb;
        --surface: #ffffff;
        --background: #f5f7fb;
    }

    /* Application background */

    .stApp {
        background:
            linear-gradient(
                180deg,
                #f8fafc 0%,
                #f5f7fb 100%
            );
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }

    /* Hide unnecessary Streamlit chrome */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ======================================================
       TOP BRAND
    ====================================================== */

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 1.4rem;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;

        background:
            linear-gradient(
                135deg,
                #4f46e5,
                #7c3aed
            );

        color: white;
        font-size: 20px;
        font-weight: 800;

        box-shadow:
            0 10px 25px rgba(79,70,229,.22);
    }

    .brand-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--navy);
        line-height: 1.1;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: .76rem;
        margin-top: 3px;
    }


    /* ======================================================
       HERO
    ====================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        border-radius: 24px;
        padding: 2.2rem 2.4rem;

        background:
            linear-gradient(
                120deg,
                #111827 0%,
                #1e1b4b 55%,
                #312e81 100%
            );

        color: white;

        box-shadow:
            0 18px 45px rgba(15,23,42,.16);

        margin-bottom: 1.3rem;
    }

    .hero::after {
        content: "";
        position: absolute;

        width: 280px;
        height: 280px;

        right: -80px;
        top: -120px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(129,140,248,.34),
                transparent 68%
            );
    }

    .hero-eyebrow {
        color: #c7d2fe;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 850;
        letter-spacing: -.035em;
        margin-bottom: .55rem;
    }

    .hero-description {
        color: #cbd5e1;
        font-size: .93rem;
        max-width: 760px;
        line-height: 1.65;
    }

    .ready {
        display: inline-flex;
        align-items: center;
        gap: 7px;

        margin-top: 1.15rem;

        padding: 7px 12px;

        border-radius: 999px;

        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.14);

        color: #d1fae5;
        font-size: .72rem;
        font-weight: 750;
    }

    .ready-dot {
        width: 7px;
        height: 7px;
        background: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 12px rgba(52,211,153,.8);
    }


    /* ======================================================
       KPI CARDS
    ====================================================== */

    .metric-card {
        background: white;
        border: 1px solid var(--border);

        border-radius: 18px;

        padding: 1.15rem 1.25rem;

        min-height: 118px;

        box-shadow:
            0 7px 22px rgba(15,23,42,.045);
    }

    .metric-label {
        color: #64748b;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--navy);
        font-size: 1.42rem;
        font-weight: 820;
        margin-top: .45rem;
    }

    .metric-help {
        color: #94a3b8;
        font-size: .73rem;
        margin-top: .25rem;
    }


    /* ======================================================
       SECTION
    ====================================================== */

    .section {
        margin-top: 1.8rem;
        margin-bottom: .9rem;
    }

    .section-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .section-number {
        width: 30px;
        height: 30px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 9px;

        background: #eef2ff;
        color: #4f46e5;

        font-size: .72rem;
        font-weight: 850;
    }

    .section-title {
        color: var(--navy);
        font-size: 1.15rem;
        font-weight: 800;
    }

    .section-description {
        color: #64748b;
        font-size: .78rem;
        margin: .35rem 0 0 40px;
    }


    /* ======================================================
       FORM CARDS
    ====================================================== */

    div[data-testid="stForm"] {
        background: white;

        border: 1px solid var(--border);

        border-radius: 22px;

        padding: 1.35rem 1.45rem;

        box-shadow:
            0 10px 35px rgba(15,23,42,.045);
    }

    div[data-testid="stForm"] label {
        color: #334155 !important;
        font-size: .78rem !important;
        font-weight: 650 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] > div {
        border-color: #dbe1ea !important;
        border-radius: 10px !important;
        background: #fbfcfe !important;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-testid="stNumberInput"] > div:hover {
        border-color: #a5b4fc !important;
    }


    /* ======================================================
       SUBMIT BUTTON
    ====================================================== */

    div[data-testid="stFormSubmitButton"] button {
        height: 50px;

        border: none !important;
        border-radius: 12px !important;

        background:
            linear-gradient(
                100deg,
                #4f46e5,
                #7c3aed
            ) !important;

        color: white !important;

        font-size: .9rem !important;
        font-weight: 800 !important;

        box-shadow:
            0 10px 25px rgba(79,70,229,.20);

        transition:
            transform .15s ease,
            box-shadow .15s ease;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 15px 30px rgba(79,70,229,.26);
    }


    /* ======================================================
       RESULT PANEL
    ====================================================== */

    .result-panel {
        margin-top: 1.5rem;

        background: white;

        border: 1px solid var(--border);

        border-radius: 22px;

        overflow: hidden;

        box-shadow:
            0 15px 40px rgba(15,23,42,.06);
    }

    .result-top {
        padding: 1.7rem 1.8rem;

        background:
            linear-gradient(
                135deg,
                #eef2ff,
                #f5f3ff
            );

        border-bottom: 1px solid #e5e7eb;
    }

    .result-label {
        color: #6366f1;

        font-size: .7rem;
        font-weight: 850;

        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .result-main {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;

        gap: 20px;

        margin-top: .35rem;
    }

    .probability {
        color: var(--navy);

        font-size: 3.5rem;
        font-weight: 900;

        line-height: 1;

        letter-spacing: -.04em;
    }

    .probability-label {
        color: #64748b;

        font-size: .78rem;

        margin-top: .45rem;
    }

    .prediction {
        color: var(--navy);

        font-size: 1.3rem;

        font-weight: 820;

        text-align: right;
    }

    .prediction-sub {
        color: #64748b;

        font-size: .73rem;

        text-align: right;

        margin-top: .2rem;
    }

    .result-body {
        padding: 1.35rem 1.8rem;
    }


    /* ======================================================
       RISK BADGE
    ====================================================== */

    .risk {
        display: inline-flex;

        align-items: center;

        gap: 8px;

        padding: 8px 13px;

        border-radius: 999px;

        font-size: .72rem;

        font-weight: 800;

        margin-top: .7rem;
    }

    .risk-low {
        color: #047857;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
    }

    .risk-medium {
        color: #b45309;
        background: #fffbeb;
        border: 1px solid #fde68a;
    }

    .risk-high {
        color: #b91c1c;
        background: #fef2f2;
        border: 1px solid #fecaca;
    }


    /* ======================================================
       SUMMARY BOX
    ====================================================== */

    .summary {
        margin-top: 1rem;

        padding: 1rem 1.1rem;

        border-radius: 14px;

        background: #f8fafc;

        border: 1px solid #e2e8f0;

        color: #475569;

        font-size: .78rem;

        line-height: 1.6;
    }

    .summary-title {
        color: #1e293b;

        font-weight: 800;

        margin-bottom: .25rem;
    }


    /* ======================================================
       DISCLAIMER
    ====================================================== */

    .disclaimer {
        margin-top: 1rem;

        padding: .9rem 1rem;

        border-radius: 12px;

        background: #fff;

        border: 1px dashed #cbd5e1;

        color: #64748b;

        font-size: .7rem;

        line-height: 1.55;
    }


    /* ======================================================
       FIGURE PRESENTATION
    ====================================================== */

    .figure-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: .75rem;
        box-shadow: 0 8px 25px rgba(15,23,42,.04);
        overflow: hidden;
    }

    .figure-note {
        color: #64748b;
        font-size: .68rem;
        line-height: 1.45;
        margin: .35rem .35rem .1rem;
    }

    /* ENTERPRISE INTELLIGENCE */
    .intel-card{background:rgba(255,255,255,.96);border:1px solid #e2e8f0;border-radius:20px;padding:1.15rem 1.2rem;box-shadow:0 10px 30px rgba(15,23,42,.045);height:100%;}
    .intel-kicker{color:#64748b;font-size:.68rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase;}
    .intel-value{color:#0f172a;font-size:1.55rem;font-weight:850;letter-spacing:-.025em;margin-top:.35rem;}
    .intel-caption{color:#64748b;font-size:.72rem;line-height:1.45;margin-top:.28rem;}
    .intel-accent{width:34px;height:4px;border-radius:999px;background:linear-gradient(90deg,#4f46e5,#7c3aed);margin-bottom:.75rem;}
    .insight-card{background:linear-gradient(135deg,#eef2ff,#f5f3ff);border:1px solid #c7d2fe;border-radius:20px;padding:1.1rem 1.2rem;min-height:122px;}
    .insight-title{color:#3730a3;font-size:.78rem;font-weight:850;margin-bottom:.35rem;}
    .insight-text{color:#475569;font-size:.74rem;line-height:1.55;}
    .chart-heading{color:#172033;font-size:.94rem;font-weight:850;margin-bottom:.2rem;}
    .chart-subheading{color:#64748b;font-size:.7rem;line-height:1.45;margin-bottom:.65rem;}
    .driver-shell{background:white;border:1px solid #e2e8f0;border-radius:22px;padding:1.2rem;box-shadow:0 10px 30px rgba(15,23,42,.045);}
    .driver-badge{display:inline-flex;align-items:center;padding:5px 9px;border-radius:999px;background:#eef2ff;color:#4f46e5;font-size:.65rem;font-weight:850;letter-spacing:.06em;text-transform:uppercase;}


    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    </style>
    """
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(
            f"Model artifact not found: {MODEL_PATH}"
        )
        st.stop()

    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# FIGURE HELPER
# ============================================================

def show_figure(path, width=1050):
    """Display report figures at a controlled size so large source images
    do not dominate the Streamlit page."""
    if path.exists():
        st.html('<div class="figure-card">')
        st.image(str(path), width=width)
        st.html('</div>')


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="
            padding:18px 4px 10px 4px;
        ">

            <div style="
                font-size:1.2rem;
                font-weight:800;
                margin-bottom:4px;
            ">
                HR Pulse
            </div>

            <div style="
                color:#94a3b8;
                font-size:.75rem;
                margin-bottom:28px;
            ">
                Attrition Intelligence
            </div>

            <div style="
                padding:12px;
                border-radius:12px;
                background:#1e293b;
                border:1px solid #334155;
                margin-bottom:12px;
            ">
                <div style="
                    color:#94a3b8;
                    font-size:.65rem;
                    text-transform:uppercase;
                    letter-spacing:.08em;
                ">
                    Model
                </div>

                <div style="
                    margin-top:5px;
                    font-weight:750;
                ">
                    Logistic Regression
                </div>
            </div>

            <div style="
                padding:12px;
                border-radius:12px;
                background:#1e293b;
                border:1px solid #334155;
                margin-bottom:12px;
            ">
                <div style="
                    color:#94a3b8;
                    font-size:.65rem;
                    text-transform:uppercase;
                    letter-spacing:.08em;
                ">
                    Validation
                </div>

                <div style="
                    margin-top:5px;
                    font-weight:750;
                ">
                    Stratified 5-Fold CV
                </div>
            </div>

            <div style="
                padding:12px;
                border-radius:12px;
                background:#1e293b;
                border:1px solid #334155;
            ">
                <div style="
                    color:#94a3b8;
                    font-size:.65rem;
                    text-transform:uppercase;
                    letter-spacing:.08em;
                ">
                    Test ROC-AUC
                </div>

                <div style="
                    margin-top:5px;
                    font-weight:750;
                ">
                    0.8102
                </div>
            </div>

        </div>
        """
    )


# ============================================================
# BRAND
# ============================================================

st.html(
    """
    <div class="brand">

        <div class="brand-icon">
            ◈
        </div>

        <div>
            <div class="brand-name">
                HR Pulse
            </div>

            <div class="brand-subtitle">
                Workforce intelligence platform
            </div>
        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-eyebrow">
            HR ANALYTICS · PREDICTIVE INTELLIGENCE
        </div>

        <div class="hero-title">
            Employee Attrition Intelligence
        </div>

        <div class="hero-description">
            Assess employee attrition probability using a
            stratified and cross-validated machine-learning
            pipeline with automated categorical encoding
            and numerical feature preprocessing.
        </div>

        <div class="ready">
            <span class="ready-dot"></span>
            MODEL READY
        </div>

    </div>
    """
)


# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                MODEL
            </div>

            <div class="metric-value">
                Logistic Regression
            </div>

            <div class="metric-help">
                Selected through CV
            </div>

        </div>
        """
    )


with c2:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                CV ROC-AUC
            </div>

            <div class="metric-value">
                0.8379
            </div>

            <div class="metric-help">
                Stratified 5-fold
            </div>

        </div>
        """
    )


with c3:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                TEST ROC-AUC
            </div>

            <div class="metric-value">
                0.8102
            </div>

            <div class="metric-help">
                Held-out evaluation
            </div>

        </div>
        """
    )


with c4:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-label">
                TEST ACCURACY
            </div>

            <div class="metric-value">
                86.39%
            </div>

            <div class="metric-help">
                Final evaluation set
            </div>

        </div>
        """
    )


# ============================================================
# SECTION HEADER
# ============================================================

st.html(
    """
    <div class="section">

        <div class="section-row">

            <div class="section-number">
                01
            </div>

            <div class="section-title">
                Employee Assessment
            </div>

        </div>

        <div class="section-description">
            Build an employee profile to generate
            a model-based attrition assessment.
        </div>

    </div>
    """
)


# ============================================================
# FORM
# ============================================================

with st.form("employee_assessment"):

    # ========================================================
    # PERSONAL
    # ========================================================

    st.html(
        """
        <div style="
            font-size:1rem;
            font-weight:800;
            color:#1e293b;
            margin-bottom:1rem;
        ">
            Personal profile
        </div>
        """
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=35,
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
        )

    with a2:

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"],
        )

        distance_from_home = st.number_input(
            "Distance From Home",
            min_value=1,
            max_value=100,
            value=10,
        )

    with a3:

        education = st.number_input(
            "Education Level",
            min_value=1,
            max_value=5,
            value=3,
        )

        education_field = st.selectbox(
            "Education Field",
            [
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other",
            ],
        )


    st.html(
        """
        <div style="
            height:1px;
            background:#edf0f4;
            margin:1.2rem 0;
        "></div>
        """
    )


    # ========================================================
    # JOB
    # ========================================================

    st.html(
        """
        <div style="
            font-size:1rem;
            font-weight:800;
            color:#1e293b;
            margin-bottom:1rem;
        ">
            Job profile
        </div>
        """
    )

    b1, b2, b3 = st.columns(3)

    with b1:

        department = st.selectbox(
            "Department",
            [
                "Sales",
                "Research & Development",
                "Human Resources",
            ],
        )

        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources",
            ],
        )

        job_level = st.number_input(
            "Job Level",
            1,
            5,
            2,
        )

    with b2:

        business_travel = st.selectbox(
            "Business Travel",
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel",
            ],
        )

        overtime = st.selectbox(
            "OverTime",
            ["Yes", "No"],
        )

        job_involvement = st.number_input(
            "Job Involvement",
            1,
            4,
            3,
        )

    with b3:

        job_satisfaction = st.number_input(
            "Job Satisfaction",
            1,
            4,
            3,
        )

        environment_satisfaction = st.number_input(
            "Environment Satisfaction",
            1,
            4,
            3,
        )

        relationship_satisfaction = st.number_input(
            "Relationship Satisfaction",
            1,
            4,
            3,
        )


    st.html(
        """
        <div style="
            height:1px;
            background:#edf0f4;
            margin:1.2rem 0;
        "></div>
        """
    )


    # ========================================================
    # COMPENSATION
    # ========================================================

    st.html(
        """
        <div style="
            font-size:1rem;
            font-weight:800;
            color:#1e293b;
            margin-bottom:1rem;
        ">
            Compensation
        </div>
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        daily_rate = st.number_input(
            "Daily Rate",
            100,
            1600,
            800,
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            30,
            100,
            65,
        )

    with c2:

        monthly_income = st.number_input(
            "Monthly Income",
            1000,
            20000,
            5000,
            step=100,
        )

        monthly_rate = st.number_input(
            "Monthly Rate",
            2000,
            27000,
            14000,
            step=100,
        )

    with c3:

        percent_salary_hike = st.number_input(
            "Percent Salary Hike",
            10,
            30,
            15,
        )

        stock_option_level = st.number_input(
            "Stock Option Level",
            0,
            3,
            1,
        )


    st.html(
        """
        <div style="
            height:1px;
            background:#edf0f4;
            margin:1.2rem 0;
        "></div>
        """
    )


    # ========================================================
    # CAREER
    # ========================================================

    st.html(
        """
        <div style="
            font-size:1rem;
            font-weight:800;
            color:#1e293b;
            margin-bottom:1rem;
        ">
            Career history
        </div>
        """
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        total_working_years = st.number_input(
            "Total Working Years",
            0,
            50,
            10,
        )

        num_companies_worked = st.number_input(
            "Number of Companies Worked",
            0,
            20,
            2,
        )

    with d2:

        years_at_company = st.number_input(
            "Years At Company",
            0,
            40,
            5,
        )

        years_in_current_role = st.number_input(
            "Years In Current Role",
            0,
            20,
            3,
        )

    with d3:

        years_since_last_promotion = st.number_input(
            "Years Since Last Promotion",
            0,
            20,
            1,
        )

        years_with_curr_manager = st.number_input(
            "Years With Current Manager",
            0,
            20,
            3,
        )


    st.html(
        """
        <div style="
            height:1px;
            background:#edf0f4;
            margin:1.2rem 0;
        "></div>
        """
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.html(
        """
        <div style="
            font-size:1rem;
            font-weight:800;
            color:#1e293b;
            margin-bottom:1rem;
        ">
            Performance & wellbeing
        </div>
        """
    )

    e1, e2, e3 = st.columns(3)

    with e1:

        performance_rating = st.number_input(
            "Performance Rating",
            1,
            5,
            3,
        )

    with e2:

        training_times_last_year = st.number_input(
            "Training Times Last Year",
            0,
            20,
            3,
        )

    with e3:

        work_life_balance = st.number_input(
            "Work-Life Balance",
            1,
            4,
            3,
        )


    st.write("")

    submitted = st.form_submit_button(
        "Assess Employee Attrition Risk  →",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "DailyRate": daily_rate,
                "DistanceFromHome": distance_from_home,
                "Education": education,
                "EnvironmentSatisfaction": environment_satisfaction,
                "HourlyRate": hourly_rate,
                "JobInvolvement": job_involvement,
                "JobLevel": job_level,
                "JobSatisfaction": job_satisfaction,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies_worked,
                "PercentSalaryHike": percent_salary_hike,
                "PerformanceRating": performance_rating,
                "RelationshipSatisfaction": relationship_satisfaction,
                "StockOptionLevel": stock_option_level,
                "TotalWorkingYears": total_working_years,
                "TrainingTimesLastYear": training_times_last_year,
                "WorkLifeBalance": work_life_balance,
                "YearsAtCompany": years_at_company,
                "YearsInCurrentRole": years_in_current_role,
                "YearsSinceLastPromotion": years_since_last_promotion,
                "YearsWithCurrManager": years_with_curr_manager,
                "BusinessTravel": business_travel,
                "Department": department,
                "EducationField": education_field,
                "Gender": gender,
                "JobRole": job_role,
                "MaritalStatus": marital_status,
                "OverTime": overtime,
            }
        ]
    )


    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = model.predict(
        input_data
    )[0]

    probability_pct = probability * 100


    # ========================================================
    # RISK
    # ========================================================

    if probability < 0.30:

        risk = "Low Risk"
        risk_class = "risk-low"

    elif probability < 0.60:

        risk = "Moderate Risk"
        risk_class = "risk-medium"

    else:

        risk = "High Risk"
        risk_class = "risk-high"


    prediction_text = (
        "Likely to Leave"
        if prediction == 1
        else "Likely to Stay"
    )


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.html(
        """
        <div class="section">

            <div class="section-row">

                <div class="section-number">
                    02
                </div>

                <div class="section-title">
                    Attrition Intelligence
                </div>

            </div>

            <div class="section-description">
                Model-generated assessment for the submitted
                employee profile.
            </div>

        </div>
        """
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.html(
        f"""
        <div class="result-panel">

            <div class="result-top">

                <div class="result-label">
                    ATTRITION ASSESSMENT
                </div>

                <div class="result-main">

                    <div>

                        <div class="probability">
                            {probability_pct:.2f}%
                        </div>

                        <div class="probability-label">
                            Estimated attrition probability
                        </div>

                        <div class="risk {risk_class}">
                            {risk}
                        </div>

                    </div>

                    <div>

                        <div class="prediction">
                            {prediction_text}
                        </div>

                        <div class="prediction-sub">
                            Logistic Regression prediction
                        </div>

                    </div>

                </div>

            </div>

            <div class="result-body">

                <div style="
                    color:#64748b;
                    font-size:.7rem;
                    font-weight:800;
                    text-transform:uppercase;
                    letter-spacing:.08em;
                    margin-bottom:8px;
                ">
                    Probability profile
                </div>

                <div style="
                    height:12px;
                    border-radius:999px;
                    background:#e2e8f0;
                    overflow:hidden;
                ">

                    <div style="
                        width:{probability_pct}%;
                        height:100%;
                        border-radius:999px;
                        background:
                            linear-gradient(
                                90deg,
                                #4f46e5,
                                #7c3aed
                            );
                    "></div>

                </div>

                <div class="summary">

                    <div class="summary-title">
                        Assessment summary
                    </div>

                    The submitted employee profile produces
                    an estimated attrition probability of
                    <strong>{probability_pct:.2f}%</strong>.
                    The current model classification is
                    <strong>{prediction_text}</strong>.

                </div>

                <div class="disclaimer">

                    <strong>Responsible-use notice:</strong>
                    This prototype is intended for educational
                    and analytical demonstration. Predictions
                    are statistical estimates and should not be
                    used as the sole basis for employment,
                    promotion, disciplinary, termination, or
                    other consequential workforce decisions.

                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # RESULT METRICS
    # ========================================================

    st.write("")

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Attrition Probability",
            f"{probability_pct:.2f}%",
        )

    with r2:

        st.metric(
            "Model Decision",
            prediction_text,
        )

    with r3:

        st.metric(
            "Risk Band",
            risk,
        )

    # ============================================================
# MODEL INTELLIGENCE — ENTERPRISE ANALYTICS
# ============================================================

st.html("""
<div class="section">
    <div class="section-row"><div class="section-number">04</div><div class="section-title">Model Intelligence</div></div>
    <div class="section-description">A compact evaluation workspace for model quality, discrimination, classification behaviour, and validation stability.</div>
</div>
""")

# Executive snapshot
cv_auc_value, test_auc_value, test_accuracy_value = "0.8379", "0.8102", "86.39%"

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.html("""<div class="intel-card"><div class="intel-accent"></div><div class="intel-kicker">Selected model</div><div class="intel-value" style="font-size:1.25rem;">Logistic Regression</div><div class="intel-caption">Final serialized production pipeline</div></div>""")
with m2:
    st.html(f"""<div class="intel-card"><div class="intel-accent"></div><div class="intel-kicker">CV ROC-AUC</div><div class="intel-value">{cv_auc_value}</div><div class="intel-caption">Stratified 5-fold validation</div></div>""")
with m3:
    st.html(f"""<div class="intel-card"><div class="intel-accent"></div><div class="intel-kicker">Held-out ROC-AUC</div><div class="intel-value">{test_auc_value}</div><div class="intel-caption">Final evaluation set</div></div>""")
with m4:
    st.html(f"""<div class="intel-card"><div class="intel-accent"></div><div class="intel-kicker">Test accuracy</div><div class="intel-value">{test_accuracy_value}</div><div class="intel-caption">Held-out classification performance</div></div>""")

st.write("")
st.html("""<div class="intel-card" style="height:auto;margin-bottom:1rem;"><div class="chart-heading">Model benchmark</div><div class="chart-subheading">Cross-validated metrics used to compare candidate classifiers before final model selection.</div></div>""")

if CV_RESULTS_PATH.exists():
    cv_df = pd.read_csv(CV_RESULTS_PATH)
    performance_df = cv_df.pivot(index="Model", columns="Metric", values="Mean")
    desired_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    available_metrics = [m for m in desired_metrics if m in performance_df.columns]
    display_df = performance_df[available_metrics].copy()
    display_df.columns = [m.replace("_", " ").title() for m in display_df.columns]
    st.dataframe(display_df.style.format("{:.4f}"), use_container_width=True, height=min(250, 70 + 36 * len(display_df)))
else:
    st.info("Cross-validation results file not found.")

st.write("")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.html("""<div class="chart-heading">Model comparison</div><div class="chart-subheading">Accuracy, precision, recall, F1, ROC-AUC and PR-AUC across candidate models.</div>""")
    p = FIGURES / "08_model_comparison.png"
    if p.exists(): show_figure(p, width=620)
with c2:
    st.html("""<div class="chart-heading">ROC discrimination</div><div class="chart-subheading">How effectively the selected classifier separates attrition from non-attrition.</div>""")
    p = FIGURES / "09_roc_curve.png"
    if p.exists(): show_figure(p, width=620)

st.write("")
c3, c4 = st.columns(2, gap="large")
with c3:
    st.html("""<div class="chart-heading">Precision–recall behaviour</div><div class="chart-subheading">Performance under the imbalanced attrition target, where positive-class retrieval matters.</div>""")
    p = FIGURES / "10_precision_recall_curves.png"
    if p.exists(): show_figure(p, width=620)
with c4:
    st.html("""<div class="chart-heading">Classification errors</div><div class="chart-subheading">Held-out confusion matrix showing correct predictions and misclassifications.</div>""")
    p = FIGURES / "11_confusion_matrix.png"
    if p.exists(): show_figure(p, width=620)

st.write("")
s1, s2 = st.columns([1.7, 1], gap="large")
with s1:
    st.html("""<div class="chart-heading">Cross-validation stability</div><div class="chart-subheading">Fold-level score distribution provides a visual check on validation consistency.</div>""")
    p = FIGURES / "13_cv_score_distribution.png"
    if p.exists(): show_figure(p, width=760)
with s2:
    st.html("""<div class="insight-card"><div class="insight-title">Validation readout</div><div class="insight-text">The model is evaluated with stratified 5-fold validation and a separate held-out test set, keeping model selection evidence distinct from final evaluation.</div></div>""")
    st.write("")
    st.html("""<div class="insight-card"><div class="insight-title">What to inspect</div><div class="insight-text">Focus on ROC-AUC for discrimination, PR behaviour for the imbalanced target, and the confusion matrix for the operational error profile.</div></div>""")

# ============================================================
# PREDICTION DRIVERS — ENTERPRISE EXPLAINABILITY
# ============================================================

st.html("""
<div class="section">
    <div class="section-row">
        <div class="section-number">05</div>
        <div class="section-title">Prediction Drivers</div>
    </div>
    <div class="section-description">
        Model-level signals identified through permutation importance
        analysis on the selected Logistic Regression model.
    </div>
</div>
""")

# Documented permutation-importance results for the selected model.
importance_data = [
    ("OverTime", 0.1072),
    ("NumCompaniesWorked", 0.0256),
    ("MaritalStatus", 0.0232),
    ("YearsSinceLastPromotion", 0.0231),
    ("BusinessTravel", 0.0212),
]

max_importance = importance_data[0][1]

driver_rows = ""

for index, (feature, importance) in enumerate(importance_data, start=1):
    relative_width = (importance / max_importance) * 100

    driver_rows += f"""
    <div style="
        display:grid;
        grid-template-columns:36px 190px 1fr 75px;
        align-items:center;
        gap:14px;
        padding:14px 0;
        border-bottom:1px solid #eef2f7;
    ">
        <div style="
            color:#94a3b8;
            font-size:.72rem;
            font-weight:800;
        ">
            {index:02d}
        </div>

        <div style="
            color:#1e293b;
            font-size:.78rem;
            font-weight:750;
        ">
            {feature}
        </div>

        <div style="
            height:8px;
            background:#eef2f7;
            border-radius:999px;
            overflow:hidden;
        ">
            <div style="
                width:{relative_width:.1f}%;
                height:100%;
                background:linear-gradient(90deg,#4f46e5,#7c3aed);
                border-radius:999px;
            "></div>
        </div>

        <div style="
            text-align:right;
            color:#475569;
            font-size:.72rem;
            font-weight:800;
        ">
            {importance:.4f}
        </div>
    </div>
    """

d1, d2 = st.columns([1.7, 1], gap="large")

with d1:
    st.html(
        f"""
        <div class="driver-shell">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:12px;
                margin-bottom:.3rem;
            ">
                <div>
                    <div class="chart-heading">
                        Top Model Signals
                    </div>

                    <div class="chart-subheading">
                        Mean decrease in held-out ROC-AUC when
                        individual features are randomly shuffled.
                    </div>
                </div>

                <div class="driver-badge">
                    Permutation Importance
                </div>
            </div>

            {driver_rows}

            <div style="
                color:#94a3b8;
                font-size:.66rem;
                line-height:1.45;
                margin-top:.75rem;
            ">
                Higher values indicate greater sensitivity of model
                predictive performance to that feature.
            </div>

        </div>
        """
    )

with d2:
    st.html(
        """
        <div style="
            background:linear-gradient(145deg,#111827,#312e81);
            border-radius:22px;
            padding:1.5rem;
            color:white;
            box-shadow:0 15px 35px rgba(49,46,129,.18);
            min-height:100%;
        ">

            <div style="
                color:#c7d2fe;
                font-size:.65rem;
                font-weight:850;
                letter-spacing:.12em;
                text-transform:uppercase;
            ">
                Key observation
            </div>

            <div style="
                font-size:1.65rem;
                font-weight:900;
                letter-spacing:-.03em;
                margin-top:.6rem;
            ">
                OverTime
            </div>

            <div style="
                color:#cbd5e1;
                font-size:.75rem;
                line-height:1.6;
                margin-top:.45rem;
            ">
                The strongest model-level signal in the documented
                permutation-importance analysis.
            </div>

            <div style="
                margin-top:1.3rem;
                padding:1rem;
                border-radius:14px;
                background:rgba(255,255,255,.08);
                border:1px solid rgba(255,255,255,.10);
            ">
                <div style="
                    color:#a5b4fc;
                    font-size:.62rem;
                    font-weight:800;
                    text-transform:uppercase;
                    letter-spacing:.08em;
                ">
                    Importance score
                </div>

                <div style="
                    font-size:1.55rem;
                    font-weight:850;
                    margin-top:.25rem;
                ">
                    0.1072
                </div>
            </div>

            <div style="
                margin-top:1.2rem;
                color:#cbd5e1;
                font-size:.68rem;
                line-height:1.55;
            ">
                This is a global model-interpretability signal.
                It does not establish that overtime causes an
                individual employee to leave.
            </div>

        </div>
        """
    )

st.write("")

st.html(
    """
    <div class="driver-shell">

        <div class="chart-heading">
            Complete Feature-Importance Analysis
        </div>

        <div class="chart-subheading">
            Top 15 features from the held-out permutation-importance
            analysis generated by the experiment pipeline.
        </div>

    </div>
    """
)

feature_importance = FIGURES / "12_permutation_feature_importance.png"

if feature_importance.exists():
    show_figure(feature_importance, width=980)
else:
    st.warning("Permutation feature-importance figure not found.")

st.html(
    """
    <div style="
        margin-top:.8rem;
        padding:.85rem 1rem;
        border-radius:12px;
        background:#f8fafc;
        border:1px solid #e2e8f0;
        color:#64748b;
        font-size:.68rem;
        line-height:1.55;
    ">
        <strong style="color:#334155;">
            Interpretation note:
        </strong>
        Permutation importance describes how much predictive
        performance changes when a feature is shuffled. It is a
        global model-interpretability measure and should not be
        interpreted as an individual-level causal explanation.
    </div>
    """
)

# ============================================================
# PIPELINE ARCHITECTURE
# ============================================================

st.html("""
<div class="section">
    <div class="section-row">
        <div class="section-number">06</div>
        <div class="section-title">ML Pipeline Architecture</div>
    </div>
    <div class="section-description">
        The complete transformation and inference path behind the prototype.
    </div>
</div>
""")

architecture = FIGURES / "07_pipeline_architecture.png"

if architecture.exists():
    show_figure(architecture, width=980)

# ============================================================
# LEARNING CURVE
# ============================================================

st.html("""
<div class="section">
    <div class="section-row">
        <div class="section-number">07</div>
        <div class="section-title">Learning Behaviour</div>
    </div>
    <div class="section-description">
        Training-size behaviour used to understand how performance
        changes as more data is introduced.
    </div>
</div>
""")

learning_curve = FIGURES / "14_learning_curve.png"

if learning_curve.exists():
    show_figure(learning_curve, width=980)

# ============================================================
# RESPONSIBLE USE
# ============================================================

st.html(
    """
    <div style="
        margin-top:2rem;
        padding:1rem 1.1rem;
        border-radius:14px;
        background:#f8fafc;
        border:1px solid #e2e8f0;
        color:#64748b;
        font-size:.72rem;
        line-height:1.6;
    ">
        <strong style="color:#334155;">
            Responsible use:
        </strong>
        This prototype is intended for educational and analytical
        demonstration. Model predictions are statistical estimates
        and should not be used as the sole basis for employment,
        promotion, disciplinary, termination, or other consequential
        workforce decisions.
    </div>
    """
)