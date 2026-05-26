import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="GitHub Profile Analyzer", page_icon="🐙", layout="wide")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-label { color: #8b949e; font-size: 13px; margin-bottom: 6px; }
    .metric-value { font-size: 26px; font-weight: 700; }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
    }
    .section-header {
        color: #8b949e;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    div[data-testid="stTextInput"] input {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stButton"] button {
        background-color: #238636 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #2ea043 !important;
    }
    .stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Train models once (cached) ────────────────────────────────────────────────
@st.cache_resource
def train_models():
    np.random.seed(42)
    N = 1000
    data = {
        'public_repos':      np.random.randint(1, 150, N),
        'followers':         np.random.randint(0, 2000, N),
        'following':         np.random.randint(0, 500, N),
        'total_stars':       np.random.randint(0, 5000, N),
        'total_forks':       np.random.randint(0, 1000, N),
        'commits_last_year': np.random.randint(0, 1500, N),
        'pull_requests':     np.random.randint(0, 300, N),
        'issues_opened':     np.random.randint(0, 200, N),
        'languages_used':    np.random.randint(1, 15, N),
        'account_age_years': np.round(np.random.uniform(0.5, 12, N), 1),
        'has_readme_pct':    np.round(np.random.uniform(0, 1, N), 2),
        'avg_repo_size_kb':  np.random.randint(10, 5000, N),
    }
    df = pd.DataFrame(data)

    def label_activity(row):
        score = row['commits_last_year'] + row['pull_requests'] * 2
        return 'Low' if score < 200 else ('Moderate' if score < 600 else 'High')

    def label_skill(row):
        score = row['public_repos'] * 2 + row['total_stars'] * 0.05 + row['account_age_years'] * 10
        return 'Beginner' if score < 100 else ('Intermediate' if score < 250 else 'Expert')

    def label_role(row):
        lang, commits = row['languages_used'], row['commits_last_year']
        if lang <= 2 and commits < 300:    return 'Frontend'
        elif lang <= 3 and commits >= 300: return 'Backend'
        elif lang >= 8:                    return 'Data'
        elif row['pull_requests'] > 150:   return 'DevOps'
        else:                              return 'Fullstack'

    df['activity_level']   = df.apply(label_activity, axis=1)
    df['skill_level']      = df.apply(label_skill, axis=1)
    df['repo_quality_score'] = np.clip(
        df['has_readme_pct'] * 40
        + df['total_stars'] / df['public_repos'].replace(0, 1) * 0.5
        + df['total_forks'] / df['public_repos'].replace(0, 1) * 0.3
        + np.random.normal(0, 5, N), 0, 100).round(1)
    df['developer_role'] = df.apply(label_role, axis=1)

    FEATURES = ['public_repos','followers','following','total_stars','total_forks',
                'commits_last_year','pull_requests','issues_opened','languages_used',
                'account_age_years','has_readme_pct','avg_repo_size_kb']
    X = df[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le_activity = LabelEncoder(); y_activity = le_activity.fit_transform(df['activity_level'])
    le_skill    = LabelEncoder(); y_skill    = le_skill.fit_transform(df['skill_level'])
    le_role     = LabelEncoder(); y_role     = le_role.fit_transform(df['developer_role'])
    y_quality   = df['repo_quality_score'].values

    rf  = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(X_scaled, y_activity)
    gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42).fit(X_scaled, y_skill)
    lr  = LinearRegression().fit(X_scaled, y_quality)
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42).fit(X_scaled, y_role)

    return rf, gb, lr, svm, scaler, le_activity, le_skill, le_role, FEATURES


# ── GitHub API helpers ─────────────────────────────────────────────────────────
def fetch_github_profile(username: str, token: str = ""):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = requests.get(f"https://api.github.com/users/{username}", headers=headers, timeout=10)
    if r.status_code == 404:
        return None, "User not found."
    if r.status_code == 403:
        return None, "Rate limit hit. Add a GitHub token in the sidebar."
    if r.status_code != 200:
        return None, f"GitHub API error {r.status_code}."
    return r.json(), None


def fetch_repos(username: str, token: str = ""):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    repos, page = [], 1
    while True:
        r = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={"per_page": 100, "page": page, "type": "owner"},
            headers=headers, timeout=10)
        if r.status_code != 200 or not r.json():
            break
        repos.extend(r.json())
        if len(r.json()) < 100:
            break
        page += 1
    return repos


def build_profile_features(user: dict, repos: list) -> dict:
    from datetime import datetime, timezone
    total_stars = sum(r.get("stargazers_count", 0) for r in repos)
    total_forks = sum(r.get("forks_count", 0)        for r in repos)
    languages   = {r.get("language") for r in repos if r.get("language")}
    readme_count = sum(1 for r in repos if r.get("description"))  # proxy
    avg_size    = np.mean([r.get("size", 0) for r in repos]) if repos else 0

    created_at  = user.get("created_at", "")
    if created_at:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_years = (datetime.now(timezone.utc) - created).days / 365.25
    else:
        age_years = 1.0

    return {
        'public_repos':      user.get("public_repos", 0),
        'followers':         user.get("followers", 0),
        'following':         user.get("following", 0),
        'total_stars':       total_stars,
        'total_forks':       total_forks,
        'commits_last_year': min(total_stars * 3 + len(repos) * 10, 1499),  # estimated
        'pull_requests':     min(len(repos) * 4, 299),
        'issues_opened':     min(len(repos) * 2, 199),
        'languages_used':    len(languages),
        'account_age_years': round(age_years, 1),
        'has_readme_pct':    round(readme_count / max(len(repos), 1), 2),
        'avg_repo_size_kb':  int(avg_size),
    }


# ── Radar chart ────────────────────────────────────────────────────────────────
def radar_chart(profile: dict):
    radar_features = ['public_repos','followers','total_stars','languages_used','has_readme_pct','total_forks']
    radar_max      = [150, 2000, 5000, 15, 1.0, 1000]
    radar_labels   = ['Repos','Followers','Stars','Languages','README %','Forks']
    values = [min(profile[f] / m, 1.0) for f, m in zip(radar_features, radar_max)]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=radar_labels + [radar_labels[0]],
        fill='toself',
        fillcolor='rgba(35, 134, 54, 0.25)',
        line=dict(color='#2ea043', width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#161b22',
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(color='#e6edf3', size=12)),
        ),
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        margin=dict(l=30, r=30, t=30, b=30),
        height=320,
    )
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("# 🐙 GitHub Profile Analyzer")
st.markdown("<p style='color:#8b949e; margin-top:-10px;'>Enter any GitHub username to analyze their developer profile with ML</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    token = st.text_input("GitHub Token (optional)", type="password",
                          help="Avoids rate limits. Create one at github.com/settings/tokens")
    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    Uses real GitHub data + 4 ML models:
    - 🔵 **Activity** — Random Forest
    - 🟢 **Skill** — Gradient Boosting
    - 🟡 **Repo Quality** — Linear Regression
    - 🔴 **Role** — SVM
    """)

col_input, col_btn = st.columns([4, 1])
with col_input:
    username = st.text_input("", placeholder="Enter GitHub username (e.g. torvalds)", label_visibility="collapsed")
with col_btn:
    analyze = st.button("Analyze")

if analyze and username.strip():
    username = username.strip()

    with st.spinner(f"Fetching @{username} from GitHub..."):
        rf, gb, lr, svm, scaler, le_activity, le_skill, le_role, FEATURES = train_models()
        user, err = fetch_github_profile(username, token)

    if err:
        st.error(f"❌ {err}")
    else:
        with st.spinner("Loading repositories..."):
            repos = fetch_repos(username, token)

        profile = build_profile_features(user, repos)

        # ── Run models
        input_df     = pd.DataFrame([profile])
        input_scaled = scaler.transform(input_df[FEATURES])

        pred_activity = le_activity.inverse_transform(rf.predict(input_scaled))[0]
        pred_skill    = le_skill.inverse_transform(gb.predict(input_scaled))[0]
        pred_quality  = float(np.clip(lr.predict(input_scaled)[0], 0, 100))
        pred_role     = le_role.inverse_transform(svm.predict(input_scaled))[0]
        conf_activity = rf.predict_proba(input_scaled).max() * 100
        conf_role     = svm.predict_proba(input_scaled).max() * 100

        # ── Header
        st.markdown("---")
        c1, c2 = st.columns([1, 4])
        with c1:
            if user.get("avatar_url"):
                st.image(user["avatar_url"], width=100)
        with c2:
            name = user.get("name") or username
            st.markdown(f"## {name}")
            st.markdown(f"[@{username}](https://github.com/{username}) &nbsp;·&nbsp; {user.get('location', 'Location unknown')}")
            bio = user.get("bio")
            if bio:
                st.markdown(f"<span style='color:#8b949e'>{bio}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # ── ML Results
        st.markdown("<div class='section-header'>🤖 ML Analysis</div>", unsafe_allow_html=True)

        activity_colors = {'Low': '#cf222e', 'Moderate': '#bf8700', 'High': '#238636'}
        skill_colors    = {'Beginner': '#388bfd', 'Intermediate': '#a371f7', 'Expert': '#f78166'}
        role_colors     = {'Frontend': '#58a6ff', 'Backend': '#3fb950', 'Data': '#d2a8ff', 'DevOps': '#ffa657', 'Fullstack': '#79c0ff'}

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            color = activity_colors.get(pred_activity, '#8b949e')
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🔵 Activity Level</div>
                <div class='metric-value' style='color:{color}'>{pred_activity}</div>
                <div style='color:#8b949e; font-size:12px; margin-top:6px'>{conf_activity:.0f}% confidence</div>
            </div>""", unsafe_allow_html=True)

        with m2:
            color = skill_colors.get(pred_skill, '#8b949e')
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🟢 Skill Level</div>
                <div class='metric-value' style='color:{color}'>{pred_skill}</div>
            </div>""", unsafe_allow_html=True)

        with m3:
            quality_color = '#238636' if pred_quality > 65 else ('#bf8700' if pred_quality > 35 else '#cf222e')
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🟡 Repo Quality</div>
                <div class='metric-value' style='color:{quality_color}'>{pred_quality:.1f}<span style='font-size:14px'>/100</span></div>
            </div>""", unsafe_allow_html=True)

        with m4:
            color = role_colors.get(pred_role, '#8b949e')
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>🔴 Developer Role</div>
                <div class='metric-value' style='color:{color}'>{pred_role}</div>
                <div style='color:#8b949e; font-size:12px; margin-top:6px'>{conf_role:.0f}% confidence</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Stats + Radar
        col_stats, col_radar = st.columns([1, 1])

        with col_stats:
            st.markdown("<div class='section-header'>📊 GitHub Stats</div>", unsafe_allow_html=True)
            stats = {
                "Public Repos": profile['public_repos'],
                "Followers":    profile['followers'],
                "Following":    profile['following'],
                "Total Stars":  profile['total_stars'],
                "Total Forks":  profile['total_forks'],
                "Languages":    profile['languages_used'],
                "Account Age":  f"{profile['account_age_years']} yrs",
            }
            for label, val in stats.items():
                c_l, c_r = st.columns([2, 1])
                c_l.markdown(f"<span style='color:#8b949e'>{label}</span>", unsafe_allow_html=True)
                c_r.markdown(f"**{val}**")

        with col_radar:
            st.markdown("<div class='section-header'>🕸️ Radar Chart</div>", unsafe_allow_html=True)
            st.plotly_chart(radar_chart(profile), use_container_width=True)

        # ── Top Repos
        if repos:
            st.markdown("---")
            st.markdown("<div class='section-header'>📦 Top Repositories</div>", unsafe_allow_html=True)
            top_repos = sorted(repos, key=lambda r: r.get("stargazers_count", 0), reverse=True)[:5]
            for repo in top_repos:
                lang = repo.get("language") or "—"
                stars = repo.get("stargazers_count", 0)
                forks = repo.get("forks_count", 0)
                desc  = repo.get("description") or ""
                st.markdown(f"""
                <div style='background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
                    <a href='{repo["html_url"]}' target='_blank' style='color:#58a6ff; font-weight:600; text-decoration:none;'>
                        {repo['name']}
                    </a>
                    <span style='color:#8b949e; font-size:12px; margin-left:8px;'>{lang}</span>
                    <span style='float:right; color:#8b949e; font-size:12px;'>⭐ {stars} &nbsp; 🍴 {forks}</span>
                    <div style='color:#8b949e; font-size:13px; margin-top:4px;'>{desc[:100]}</div>
                </div>
                """, unsafe_allow_html=True)

elif analyze and not username.strip():
    st.warning("Please enter a GitHub username.")
