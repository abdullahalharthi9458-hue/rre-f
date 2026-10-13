"""
RRE-F | لوحة القيادة الاستباقية للرأي العام
محاكاة توضيحية لردود فعل أصحاب المصلحة تجاه قرار مرتقب، مع تأثير اجتماعي متبادل بين الشرائح،
وإمكانية معايرة معاملات الشرائح ببيانات استطلاع فعلية (CSV أو Excel).

التشغيل:
    pip install streamlit pandas numpy openpyxl
    streamlit run app.py
"""

import copy
import io
import json

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# إعدادات الصفحة والتنسيق (اتجاه عربي كامل)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="RRE-F | لوحة القيادة الاستباقية للرأي العام",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* الاتجاه العربي على المحتوى فقط، لا على حاويات التخطيط (حتى لا ينكسر الشريط الجانبي على الجوال) */
    .block-container, [data-testid="stMainBlockContainer"], [data-testid="stSidebarUserContent"] {
        direction: rtl; text-align: right;
    }
    [data-testid="stMetricValue"] { font-size: 1.9rem; direction: rtl; }
    [data-testid="stMetricLabel"] { direction: rtl; }
    /* المنزلقات تبقى من اليسار لليمين حتى لا ينعكس اتجاهها وتظهر الإشارة السالبة صحيحة */
    [data-baseweb="slider"] { direction: ltr; }
    @media (max-width: 640px) {
        .main-title { font-size: 21px; }
        .sub-title  { font-size: 14px; }
        [data-testid="stMetricValue"] { font-size: 1.6rem; }
    }
    .main-title { font-size: 28px; font-weight: 700; color: #1E3A8A; margin-bottom: 0; }
    .sub-title  { font-size: 16px; color: #4B5563; margin-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">منظومة هندسة السمعة العكسية والذكاء الاستباقي (RRE-F)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">لوحة القيادة التنفيذية ومختبر المحاكاة لقرارات وأزمات الرأي العام</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# المعاملات الافتراضية للشرائح (تُستخدم عند غياب بيانات الاستطلاع)
#   weight            : نسبتها من العينة
#   sens              : مدى الحساسية (شدة التفاعل مع أثر القرار) من 0 إلى 1
#   transparency_gain : مقدار تحسّن موقفها مع الشفافية (0 إلى 0.8)
#   bias              : ميل مسبق نابع من الثقة بالجهة (-0.3 إلى 0.3)
#   influence         : قوة تأثيرها في آراء الآخرين (0.5 إلى 3.0)
# ------------------------------------------------------------------
DEFAULT_SEGMENTS = {
    "الجمهور العام": dict(weight=0.65, sens=[0.3, 0.9], transparency_gain=0.35, bias=-0.05, influence=1.0),
    "المستثمرون": dict(weight=0.15, sens=[0.2, 0.6], transparency_gain=0.50, bias=0.05, influence=1.2),
    "الجهات الرقابية": dict(weight=0.10, sens=[0.1, 0.5], transparency_gain=0.60, bias=0.00, influence=1.5),
    "الإعلام": dict(weight=0.10, sens=[0.4, 1.0], transparency_gain=0.30, bias=-0.10, influence=2.5),
}
SEG_NAMES = list(DEFAULT_SEGMENTS)
OPPOSE_THRESHOLD = -0.2  # من يقل موقفه عن هذا يُعد معارضاً

# ------------------------------------------------------------------
# معايرة المعاملات من بيانات الاستطلاع
# ------------------------------------------------------------------
COL_SEGMENT = "الشريحة"
COL_IMPACT = "شدة_التأثر"       # → الحساسية
COL_CLARITY = "أثر_التوضيح"     # → التجاوب مع الشفافية
COL_TRUST = "الثقة_بالمؤسسة"    # → الميل المسبق
COL_REACH = "قوة_التأثير"       # → قوة التأثير في الآخرين
LIKERT_COLS = [COL_IMPACT, COL_CLARITY, COL_TRUST, COL_REACH]
MIN_N = 10  # أقل عدد استجابات صالحة لمعايرة معامل في شريحة

SURVEY_QUESTIONS = {
    COL_IMPACT: "إلى أي مدى سيؤثر قرار كهذا على حياتك أو عملك؟ (1 = لا يؤثر، 5 = يؤثر بشدة)",
    COL_CLARITY: "لو شرحت الجهة أسباب القرار مسبقاً بوضوح، هل سيتحسن موقفك منه؟ (1 = لا، 5 = بالتأكيد)",
    COL_TRUST: "ما مدى ثقتك بالجهة صاحبة القرار؟ (1 = لا أثق، 5 = أثق تماماً)",
    COL_REACH: "كم مرة تشارك رأيك في مثل هذه القضايا مع الآخرين أو على وسائل التواصل؟ (1 = أبداً، 5 = دائماً)",
}

SEGMENT_ALIASES = {
    "الجمهور العام": ["جمهور", "مواطن", "عملاء", "public"],
    "المستثمرون": ["مستثمر", "مساهم", "investor"],
    "الجهات الرقابية": ["رقاب", "حكوم", "تنظيم", "regulat"],
    "الإعلام": ["إعلام", "اعلام", "صحف", "صحاف", "مؤثر", "media"],
}


def normalize_segment(value):
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    for name in SEG_NAMES:
        if s == name:
            return name
    for name, keys in SEGMENT_ALIASES.items():
        if any(k in s for k in keys):
            return name
    return None


def likert(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    return v.where(v.between(1, 5))


def read_survey(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    for enc in ("utf-8-sig", "cp1256"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("تعذّرت قراءة الملف. احفظه بصيغة CSV UTF-8 أو Excel.")


def calibrate(df: pd.DataFrame, use_survey_weights: bool):
    """يحوّل إجابات الاستطلاع إلى معاملات الشرائح. أي معامل بلا بيانات كافية يبقى على قيمته الافتراضية."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if COL_SEGMENT not in df.columns:
        raise ValueError(f"الملف يجب أن يحتوي عمود «{COL_SEGMENT}».")
    present = [c for c in LIKERT_COLS if c in df.columns]
    if not present:
        raise ValueError("لا يوجد في الملف أي من أعمدة الأسئلة: " + "، ".join(LIKERT_COLS))

    profiles = copy.deepcopy(DEFAULT_SEGMENTS)
    df["_seg"] = df[COL_SEGMENT].map(normalize_segment)
    unknown = int(df["_seg"].isna().sum())
    sources = {}

    for name in SEG_NAMES:
        sub = df[df["_seg"] == name]
        p = profiles[name]
        changed = []

        def enough(col):
            if col not in sub.columns:
                return None
            v = likert(sub[col]).dropna()
            return v if len(v) >= MIN_N else None

        v = enough(COL_IMPACT)
        if v is not None:
            x = (v - 1) / 4
            center, spread = float(x.mean()), max(float(x.std(ddof=0)), 0.05)
            p["sens"] = [round(max(0.05, center - spread), 2), round(min(1.0, center + spread), 2)]
            changed.append("الحساسية")
        v = enough(COL_CLARITY)
        if v is not None:
            p["transparency_gain"] = round((float(v.mean()) - 1) / 4 * 0.8, 2)
            changed.append("الشفافية")
        v = enough(COL_TRUST)
        if v is not None:
            p["bias"] = round((float(v.mean()) - 3) / 2 * 0.3, 2)
            changed.append("الميل")
        v = enough(COL_REACH)
        if v is not None:
            p["influence"] = round(0.5 + (float(v.mean()) - 1) / 4 * 2.5, 2)
            changed.append("التأثير")

        n = len(sub)
        if changed:
            sources[name] = f"معاير من {n} استجابة"
        elif n:
            sources[name] = f"افتراضي ({n} استجابة غير كافية)"
        else:
            sources[name] = "افتراضي (لا توجد استجابات)"

    if use_survey_weights:
        counts = df["_seg"].value_counts()
        total = counts.sum()
        if total:
            for name in SEG_NAMES:
                profiles[name]["weight"] = round(float(counts.get(name, 0)) / total, 3)

    return profiles, sources, unknown, len(df)


def template_csv() -> bytes:
    rows = [
        ["الجمهور العام", 4, 3, 2, 2],
        ["المستثمرون", 3, 4, 3, 3],
        ["الجهات الرقابية", 2, 5, 4, 2],
        ["الإعلام", 4, 2, 2, 5],
    ]
    df = pd.DataFrame(rows, columns=[COL_SEGMENT] + LIKERT_COLS)
    return df.to_csv(index=False).encode("utf-8-sig")


def profiles_table(profiles: dict, sources: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "الشريحة": name,
                "الوزن": f"{p['weight'] * 100:.0f}%",
                "الحساسية": f"{p['sens'][0]} – {p['sens'][1]}",
                "التجاوب مع الشفافية": p["transparency_gain"],
                "الميل المسبق": p["bias"],
                "قوة التأثير": p["influence"],
                "المصدر": sources.get(name, "افتراضي"),
            }
            for name, p in profiles.items()
        ]
    )


# ------------------------------------------------------------------
# محرك المحاكاة
# ------------------------------------------------------------------
def friction_from_mean(mean_stance: float) -> float:
    """تحويل متوسط الموقف (-1..1) إلى معامل مقاومة (0..100)."""
    return float(np.clip((1.0 - mean_stance) * 50.0, 0.0, 100.0))


def classify(friction: float) -> tuple:
    """يعيد (تسمية قصيرة للمؤشر، وصف تفصيلي)."""
    if friction > 65:
        return "🔴 حرج", "أزمة متوقعة"
    if friction > 40:
        return "🟠 متوسط", "يتطلب تعديل السرد"
    return "🟢 آمن", "ضمن النطاق الآمن"


@st.cache_data(show_spinner=False)
def simulate(profiles_json: str, policy_impact: float, transparency: float, n_agents: int,
             social_weight: float, rounds: int, seed: int) -> dict:
    profiles = json.loads(profiles_json)
    names = list(profiles)
    k = len(names)
    rng = np.random.default_rng(seed)

    weights = np.array([profiles[s]["weight"] for s in names], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(k)
    seg_idx = rng.choice(k, size=n_agents, p=weights / weights.sum())

    def per_agent(key):
        return np.array([profiles[s][key] for s in names], dtype=float)[seg_idx]

    lo = np.array([profiles[s]["sens"][0] for s in names])[seg_idx]
    hi = np.array([profiles[s]["sens"][1] for s in names])[seg_idx]
    sens = rng.uniform(lo, hi)
    gain = per_agent("transparency_gain")
    bias = per_agent("bias")
    infl = per_agent("influence")
    noise = rng.normal(0.0, 0.08, n_agents)

    # الموقف الأولي: الأثر يُضخَّم بالحساسية، والشفافية تضيف دائماً (لا تضاعف السلبية)
    s0 = np.clip(policy_impact * (0.5 + sens) + gain * transparency + bias + noise, -1.0, 1.0)

    masks = [seg_idx == j for j in range(k)]

    def seg_means(s):
        return np.array([s[m].mean() if m.any() else 0.0 for m in masks])

    def snapshot(s):
        row = {name: friction_from_mean(mv) for name, mv, m in zip(names, seg_means(s), masks) if m.any()}
        row["الإجمالي"] = friction_from_mean(s.mean())
        return row

    # التأثير الاجتماعي (نموذج Friedkin-Johnsen مبسط):
    # كل وكيل يتأثر بمتوسط شريحته وبالرأي العام الموزون بقوة تأثير كل شريحة، مع تمسكه بموقفه الأولي.
    s = s0.copy()
    history = [snapshot(s)]
    for _ in range(rounds):
        global_signal = np.average(s, weights=infl)
        signal = 0.5 * seg_means(s)[seg_idx] + 0.5 * global_signal
        s = np.clip((1.0 - social_weight) * s0 + social_weight * signal, -1.0, 1.0)
        history.append(snapshot(s))

    rows = []
    for name, mv, m in zip(names, seg_means(s), masks):
        if not m.any():
            continue
        rows.append({
            "شريحة أصحاب المصلحة": name,
            "نسبة المقاومة (%)": round(friction_from_mean(mv), 1),
            "نسبة المعارضين (%)": round(float(np.mean(s[m] < OPPOSE_THRESHOLD)) * 100, 1),
            "عدد الوكلاء": int(m.sum()),
        })

    return {
        "friction": round(friction_from_mean(s.mean()), 1),
        "initial_friction": round(friction_from_mean(s0.mean()), 1),
        "mean_stance": round(float(s.mean()), 2),
        "oppose_share": round(float(np.mean(s < OPPOSE_THRESHOLD)) * 100, 1),
        "polarization": round(float(s.std()), 2),
        "segments": pd.DataFrame(rows),
        "history": pd.DataFrame(history).round(1).rename_axis("الجولة"),
    }


@st.cache_data(show_spinner=False)
def transparency_sweep(profiles_json, policy_impact, n_agents, social_weight, rounds, seed) -> pd.DataFrame:
    levels = np.round(np.arange(0.0, 1.01, 0.1), 1)
    rows = [
        {"مستوى الشفافية": float(t),
         "معامل المقاومة (%)": simulate(profiles_json, policy_impact, float(t), n_agents,
                                        social_weight, rounds, seed)["friction"]}
        for t in levels
    ]
    return pd.DataFrame(rows).set_index("مستوى الشفافية")


# ------------------------------------------------------------------
# الشريط الجانبي
# ------------------------------------------------------------------
st.sidebar.header("إعدادات القرار المرتقب")
st.sidebar.caption("تتحدث النتائج تلقائياً عند تغيير أي مؤشر.")

policy_impact = st.sidebar.slider("الأثر المباشر للقرار (سلبي ← إيجابي)", -1.0, 1.0, -0.3, 0.1)
transparency = st.sidebar.slider("مستوى الشفافية والتوضيح المسبق", 0.0, 1.0, 0.4, 0.1)
social_weight = st.sidebar.slider(
    "قوة التأثير الاجتماعي المتبادل", 0.0, 0.9, 0.5, 0.1,
    help="0 = كل شخص يقرر وحده، وكلما ارتفعت زاد تأثر الآراء ببعضها (خاصة بالإعلام).",
)
n_agents = st.sidebar.selectbox("حجم العينة (عدد الوكلاء)", [500, 1000, 2000, 5000], index=1)

st.sidebar.divider()
st.sidebar.subheader("معايرة الشرائح ببيانات استطلاع")
uploaded = st.sidebar.file_uploader("ارفع ملف الاستطلاع (CSV أو Excel)", type=["csv", "xlsx", "xls"])
use_survey_weights = st.sidebar.checkbox(
    "استخدم نسب الشرائح من الاستطلاع",
    value=False,
    help="فعّلها فقط إذا كانت عينة الاستطلاع تمثّل أحجام الشرائح الحقيقية.",
)
st.sidebar.download_button("تنزيل قالب الاستطلاع", template_csv(), "rre_f_survey_template.csv", "text/csv")

with st.sidebar.expander("إعدادات متقدمة"):
    rounds = st.slider("عدد جولات التفاعل", 1, 20, 8)
    seed = st.number_input("بذرة العشوائية (لتثبيت النتائج)", 0, 9999, 42, 1)
    st.caption("غيّر البذرة لرؤية عينة مختلفة؛ ثباتها يجعل المقارنة بين السيناريوهات عادلة.")

# ------------------------------------------------------------------
# تحميل المعاملات (افتراضية أو معايرة)
# ------------------------------------------------------------------
profiles = copy.deepcopy(DEFAULT_SEGMENTS)
sources = {name: "افتراضي" for name in SEG_NAMES}
calibrated = False

if uploaded is not None:
    try:
        survey = read_survey(uploaded)
        profiles, sources, unknown, total_rows = calibrate(survey, use_survey_weights)
        calibrated = any(v.startswith("معاير") for v in sources.values())
        if calibrated:
            st.sidebar.success(f"تمت المعايرة من {total_rows} استجابة.")
        else:
            st.sidebar.warning(f"لا توجد شريحة فيها {MIN_N} استجابات صالحة على الأقل؛ بقيت المعاملات الافتراضية.")
        if unknown:
            st.sidebar.warning(f"{unknown} صفاً بشريحة غير معروفة تم تجاهلها.")
    except Exception as e:
        st.sidebar.error(f"خطأ في ملف الاستطلاع: {e}")

profiles_json = json.dumps(profiles, ensure_ascii=False, sort_keys=True)

# ------------------------------------------------------------------
# التشغيل والعرض
# ------------------------------------------------------------------
res = simulate(profiles_json, policy_impact, transparency, int(n_agents), social_weight, rounds, int(seed))
sweep = transparency_sweep(profiles_json, policy_impact, int(n_agents), social_weight, rounds, int(seed))
friction = res["friction"]
amplification = round(friction - res["initial_friction"], 1)

if calibrated:
    st.caption("✅ النتائج مبنية على معاملات معايرة من بيانات الاستطلاع المرفوع.")
else:
    st.caption("ℹ️ النتائج مبنية على معاملات افتراضية. ارفع ملف استطلاع من الشريط الجانبي لمعايرتها.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("معامل المقاومة العام", f"{friction}%",
          delta=f"{amplification:+} بعد التأثير الاجتماعي", delta_color="inverse")
status_label, status_desc = classify(friction)
c2.metric("حالة الاستقرار السردي", status_label, help=status_desc)
c3.metric("نسبة المعارضين", f"{res['oppose_share']}%")
c4.metric("درجة الاستقطاب", res["polarization"],
          help="الانحراف المعياري للمواقف: كلما ارتفع زاد انقسام الرأي.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["المقاومة حسب الشريحة", "تطور الرأي عبر الجولات", "أثر الشفافية", "المعاملات والاستطلاع"]
)

with tab1:
    st.bar_chart(res["segments"].set_index("شريحة أصحاب المصلحة")[["نسبة المقاومة (%)"]])
    st.dataframe(res["segments"], hide_index=True, use_container_width=True)

with tab2:
    st.line_chart(res["history"])
    st.caption("الجولة 0 = ردة الفعل الأولى قبل أن يتأثر الناس ببعضهم.")

with tab3:
    st.line_chart(sweep)
    st.caption("معامل المقاومة المتوقع عند كل مستوى شفافية، مع تثبيت بقية المؤشرات.")

with tab4:
    st.markdown("**المعاملات المستخدمة حالياً في المحاكاة**")
    st.dataframe(profiles_table(profiles, sources), hide_index=True, use_container_width=True)
    st.markdown("**كيف تجهّز الاستبيان؟**")
    st.markdown(
        f"يحتوي الملف عمود **{COL_SEGMENT}** (الجمهور العام، المستثمرون، الجهات الرقابية، الإعلام)، "
        f"وعموداً لكل سؤال من الأسئلة التالية بمقياس من 1 إلى 5. تكفي بعض الأعمدة، "
        f"وأي معامل تقل استجاباته عن {MIN_N} في شريحة يبقى على قيمته الافتراضية."
    )
    for col, q in SURVEY_QUESTIONS.items():
        st.markdown(f"- **{col}**: {q}")

st.divider()

# ------------------------------------------------------------------
# التوصيات (مبنية على النتائج الفعلية لا على العتبة فقط)
# ------------------------------------------------------------------
st.subheader("التوصيات الاستراتيجية لصانع القرار")

seg_df = res["segments"].set_index("شريحة أصحاب المصلحة")
top_seg = seg_df["نسبة المقاومة (%)"].idxmax()
top_val = seg_df["نسبة المقاومة (%)"].max()
best_t = float(sweep["معامل المقاومة (%)"].idxmin())
best_f = float(sweep["معامل المقاومة (%)"].min())

recs = []
if friction > 65:
    st.error("تنبيه أحمر: المقاومة تتجاوز الحد الآمن بكثير.")
    recs += [
        "تأجيل إعلان القرار وإعادة بناء السرد المؤسسي قبل الإطلاق.",
        "عقد لقاءات مغلقة مع الجهات المؤثرة لاستطلاع المخاوف الحقيقية.",
    ]
elif friction > 40:
    st.warning("تنبيه أصفر: مقاومة متوسطة تتطلب تدخلاً اتصالياً مدروساً.")
    recs += [
        "تعديل زاوية الطرح للتركيز على المنافع بعيدة المدى.",
        "تجهيز المتحدثين بقائمة الأسئلة الحرجة وإجابات موحدة.",
    ]
else:
    st.success("المؤشر في النطاق الآمن.")
    recs += [
        "المضي في خطة الإطلاق المعتمدة.",
        "متابعة ردود الفعل الحية خلال أول 72 ساعة.",
    ]

recs.append(f"الشريحة الأعلى مقاومة هي **{top_seg}** ({top_val}%)، فابدأ التواصل بها.")

if best_f < friction - 5 and best_t > transparency:
    recs.append(f"رفع الشفافية إلى **{best_t}** يخفض المقاومة المتوقعة إلى نحو **{best_f}%**.")

if amplification > 3:
    recs.append(f"التأثير المتبادل رفع المقاومة **{amplification}** نقطة؛ جهّز رسائل الإعلام قبل الإعلان لا بعده.")
elif amplification < -3:
    recs.append(f"التأثير المتبادل خفّض المقاومة **{abs(amplification)}** نقطة؛ وظّف الأصوات المؤيدة كسفراء للقرار.")

for i, r in enumerate(recs, 1):
    st.markdown(f"{i}. {r}")

st.info(
    "💡 ملاحظة منهجية: هذه محاكاة توضيحية تفيد في مقارنة السيناريوهات واستكشاف الاتجاهات، لا في التنبؤ الدقيق. "
    "معايرة الشرائح ببيانات استطلاع حقيقية تقرّب النتائج من الواقع، وتزداد موثوقيتها كلما كبرت العينة وتنوعت."
)
