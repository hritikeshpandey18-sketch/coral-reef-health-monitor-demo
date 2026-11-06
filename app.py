"""
app.py
Streamlit prototype for the AI-Based Coral Reef Health Monitoring System.

Run:
    streamlit run app.py

Flow: welcome (name, role) -> analyse photos (upload / camera / samples)
-> verdict, confidence, per-model votes, Grad-CAM -> downloadable
certificate (PNG/PDF) and CSV reports -> session dashboard.

Inference goes through src.inference.prepare(), which reproduces the
training-data preparation exactly. Every accuracy figure on screen is read
from reports/ at runtime, never typed in, so the app cannot drift from the
paper.
"""

import datetime as dt
import hashlib
import html
import json
import random

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import certificate as cert
from src import config as C
from src import contrib
from src.gradcam import compute_heatmap, overlay_heatmap
from src.inference import predict, prepare
from src.models import load_trained_model
from src.preprocessing import quality_report

st.set_page_config(page_title="Coral Reef Health Monitor", page_icon="🪸",
                   layout="wide", initial_sidebar_state="expanded")

NICE = {"mobilenetv2": "MobileNetV2", "resnet50": "ResNet50",
        "efficientnetb0": "EfficientNetB0", "densenet121": "DenseNet121",
        "baseline": "Baseline CNN"}
ENSEMBLE = "ensemble"
AUTHORS = "Mohammed Saqlain Shaikh (264529) · Hritikesh Pandey (264530)"
GREEN, SAND, CORAL = "#2ECC9A", "#F2D48A", "#FF7F66"


# ======================================================================
# STYLE
# ======================================================================
def _fish(color: str, flip: bool = False) -> str:
    t = ' transform="scale(-1,1) translate(-64,0)"' if flip else ""
    return (f'<svg viewBox="0 0 64 32" xmlns="http://www.w3.org/2000/svg"><g{t}>'
            f'<path d="M4 16 C14 2 38 2 50 16 C38 30 14 30 4 16Z" fill="{color}"/>'
            f'<path d="M48 16 L62 5 L60 16 L62 27Z" fill="{color}"/>'
            f'<circle cx="14" cy="13" r="2.2" fill="#062A43"/></g></svg>')


def _reef_svg() -> str:
    rng = random.Random(3)
    def ridge(base, amp, n, color, op):
        pts, x = [f"0,200 0,{base}"], 0
        while x < 1600:
            w = rng.uniform(25, 70)
            pts += [f"{x + w * .3:.0f},{base - rng.uniform(.3, 1) * amp:.0f}",
                    f"{x + w * .5:.0f},{base - rng.uniform(.6, 1.3) * amp:.0f}",
                    f"{x + w:.0f},{base}"]
            x += w
        pts.append("1600,200")
        return f'<polygon points="{" ".join(pts)}" fill="{color}" opacity="{op}"/>'
    return ('<svg viewBox="0 0 1600 200" preserveAspectRatio="none" '
            'xmlns="http://www.w3.org/2000/svg">'
            + ridge(150, 70, 0, "#0A4A6A", .55) + ridge(175, 55, 0, "#062F48", .85)
            + ridge(195, 35, 0, "#031C2E", 1) + "</svg>")


def inject_style():
    r = random.Random(11)
    bubbles = "".join(
        f'<span style="left:{r.uniform(2, 98):.1f}%;width:{(s := r.uniform(5, 16)):.0f}px;'
        f'height:{s:.0f}px;animation-duration:{r.uniform(10, 22):.1f}s;'
        f'animation-delay:{r.uniform(0, 14):.1f}s"></span>' for _ in range(24))
    fish = "".join(
        f'<div class="fish {"rtl" if i % 2 else ""}" style="top:{top}%;width:{w}px;'
        f'animation-duration:{dur}s;animation-delay:{delay}s">{_fish(c, i % 2 == 1)}</div>'
        for i, (top, w, dur, delay, c) in enumerate([
            (22, 34, 38, 0, "rgba(255,160,122,.55)"), (48, 22, 30, 9, "rgba(143,227,240,.45)"),
            (63, 28, 44, 17, "rgba(255,210,120,.45)"), (35, 18, 26, 4, "rgba(200,240,255,.35)")]))
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Playfair+Display:wght@700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }}
.stApp {{
  background:
    radial-gradient(ellipse 70% 45% at 50% -8%, rgba(170,238,255,.40), transparent 70%),
    radial-gradient(ellipse 40% 30% at 85% 15%, rgba(120,220,255,.12), transparent 70%),
    linear-gradient(180deg, #0c6690 0%, #0a4f75 20%, #073d5c 45%, #052c45 72%, #031a2b 100%);
  background-attachment: fixed;
}}
.stApp::before {{ content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background: repeating-linear-gradient(100deg, rgba(255,255,255,.05) 0 36px, transparent 36px 120px);
  -webkit-mask-image: linear-gradient(180deg,#000 0%,transparent 60%);
          mask-image: linear-gradient(180deg,#000 0%,transparent 60%);
  animation: sway 16s ease-in-out infinite alternate; }}
.stApp::after {{ content:""; position:fixed; inset:0; pointer-events:none; z-index:0;   /* caustics */
  background:
    radial-gradient(circle at 20% 30%, rgba(255,255,255,.06) 0 6%, transparent 7%),
    radial-gradient(circle at 70% 20%, rgba(255,255,255,.05) 0 5%, transparent 6%),
    radial-gradient(circle at 45% 55%, rgba(255,255,255,.04) 0 7%, transparent 8%);
  background-size: 420px 420px; animation: drift 30s linear infinite; opacity:.8; }}
@keyframes sway {{ from {{transform:translateX(-2%)}} to {{transform:translateX(2%)}} }}
@keyframes drift {{ from {{background-position:0 0}} to {{background-position:420px 840px}} }}
.ocean {{ position:fixed; inset:0; pointer-events:none; z-index:0; overflow:hidden; }}
.ocean .bubble span {{ position:absolute; bottom:-30px; border-radius:50%;
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,.6), rgba(255,255,255,.05));
  border:1px solid rgba(255,255,255,.25); animation: rise linear infinite; }}
@keyframes rise {{ 0%{{transform:translate(0,0);opacity:0}} 10%{{opacity:.8}}
  50%{{transform:translate(14px,-55vh)}} 100%{{transform:translate(-10px,-110vh);opacity:0}} }}
.ocean .fish {{ position:absolute; left:-80px; animation: swim linear infinite; }}
.ocean .fish.rtl {{ left:auto; right:-80px; animation-name: swimrtl; }}
@keyframes swim {{ 0%{{transform:translate(0,0)}} 50%{{transform:translate(55vw,-18px)}} 100%{{transform:translate(112vw,6px)}} }}
@keyframes swimrtl {{ 0%{{transform:translate(0,0)}} 50%{{transform:translate(-55vw,14px)}} 100%{{transform:translate(-112vw,-8px)}} }}
.ocean .reef {{ position:absolute; left:0; right:0; bottom:0; height:22vh; }}
.ocean .reef svg {{ width:100%; height:100%; }}
@media (prefers-reduced-motion: reduce) {{
  .ocean .bubble span, .ocean .fish, .stApp::before, .stApp::after {{ animation:none; display:none; }} }}

[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: rgba(3,26,43,.86); backdrop-filter: blur(12px);
  border-right: 1px solid rgba(255,255,255,.08); }}
.block-container {{ position:relative; z-index:1; padding-top:1.6rem; padding-bottom:6rem; max-width:1240px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: rgba(255,255,255,.055);
  border-color: rgba(255,255,255,.13) !important; border-radius: 18px; backdrop-filter: blur(8px); }}
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; background: rgba(255,255,255,.05);
  padding:.35rem; border-radius: 14px; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px; padding: .45rem 1rem; }}
.stTabs [aria-selected="true"] {{ background: rgba(255,127,102,.18); }}
.stDownloadButton button, .stButton button {{ border-radius: 12px; font-weight:600; }}

.glass {{ background: rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14);
  border-radius:20px; padding:1.3rem 1.6rem; backdrop-filter: blur(9px);
  box-shadow: 0 10px 34px rgba(0,0,0,.28); margin-bottom:1rem; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:2.2px; font-size:.72rem; color:#8FE3F0; font-weight:700; }}
.hero h1 {{ font-family:'Playfair Display', Georgia, serif; font-size:2.7rem; margin:.15rem 0 .35rem; }}
.hero p {{ font-size:1.05rem; opacity:.85; margin:0; max-width:760px; }}
.kpi {{ text-align:center; padding:.9rem .6rem; }}
.kpi .num {{ font-size:1.9rem; font-weight:800; color:#8FE3F0; line-height:1.1; }}
.kpi .lbl {{ font-size:.72rem; opacity:.75; text-transform:uppercase; letter-spacing:1.2px; margin-top:.25rem; }}
.welcome {{ max-width:620px; margin:7vh auto 1.2rem; text-align:center; }}
.welcome .logo {{ font-size:4.2rem; filter: drop-shadow(0 6px 18px rgba(255,127,102,.45)); }}
.welcome h1 {{ font-family:'Playfair Display', Georgia, serif; font-size:2.6rem; margin:.5rem 0 .4rem; }}
.chips span {{ display:inline-block; margin:.25rem .2rem; padding:.28rem .7rem; border-radius:999px;
  background:rgba(143,227,240,.12); border:1px solid rgba(143,227,240,.3); font-size:.8rem; }}
.verdict {{ border-radius:18px; padding:1.2rem 1.4rem; margin-bottom:.8rem; }}
.verdict .label {{ font-family:'Playfair Display', Georgia, serif; font-size:2.3rem; font-weight:700; letter-spacing:.5px; }}
.verdict .sub {{ opacity:.88; font-size:.93rem; }}
.v-healthy  {{ background:linear-gradient(135deg, rgba(46,204,154,.32), rgba(46,204,154,.08)); border:1px solid rgba(46,204,154,.65); }}
.v-bleached {{ background:linear-gradient(135deg, rgba(242,212,138,.30), rgba(255,244,214,.06)); border:1px solid rgba(242,212,138,.75); }}
.flag {{ background:rgba(255,127,102,.16); border:1px solid rgba(255,127,102,.7); border-radius:12px;
  padding:.6rem .9rem; margin:.4rem 0; font-size:.9rem; }}
.vote {{ display:flex; justify-content:space-between; padding:.35rem .1rem;
  border-bottom:1px solid rgba(255,255,255,.08); font-size:.9rem; }}
.vote b.h {{ color:{GREEN}; }} .vote b.b {{ color:{SAND}; }}
.footer {{ text-align:center; opacity:.62; font-size:.8rem; margin-top:2.5rem; }}
</style>
<div class="ocean"><div class="bubble">{bubbles}</div>{fish}<div class="reef">{_reef_svg()}</div></div>
""", unsafe_allow_html=True)


# ======================================================================
# DATA (numbers read from reports/)
# ======================================================================
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(run: str):
    return load_trained_model(C.ARTIFACT_DIR / f"{run}.keras")


def installed_runs() -> list[str]:
    return sorted(p.stem for p in C.ARTIFACT_DIR.glob("*.keras") if not p.stem.startswith("_"))


def model_options() -> list[str]:
    runs = set(installed_runs())
    members = [m for m in C.ENSEMBLE_MEMBERS if f"{m}_enhanced" in runs]
    opts = [ENSEMBLE] if len(members) >= 2 else []
    ranked = [C.PRIMARY_MODEL] + [m for m in C.MODELS if m not in (C.PRIMARY_MODEL, "baseline")] + ["baseline"]
    opts += [f"{m}_enhanced" for m in ranked if f"{m}_enhanced" in runs]
    opts += sorted(r for r in runs if r.endswith("_raw"))
    return opts


def label_of(opt: str) -> str:
    if opt == ENSEMBLE:
        return "Ensemble of 4 CNNs  ★ recommended"
    name, _, tag = opt.rpartition("_")
    return NICE.get(name, name) + ("  (raw-image variant)" if tag == "raw" else "")


def accuracy_of(opt: str) -> dict | None:
    if opt == ENSEMBLE:
        f = C.REPORT_DIR / "table8_ensemble_crossval.json"
    else:
        name, _, tag = opt.rpartition("_")
        f = C.REPORT_DIR / f"cv_{name}_{'cv' if tag == 'enhanced' else tag}.json"
    if not f.exists():
        return None
    a = json.loads(f.read_text())["aggregate"]
    return {"mean": a["accuracy"]["mean"], "std": a["accuracy"]["std"],
            "f1": a["f1_macro"]["mean"], "f1_std": a["f1_macro"]["std"]}


def acc_text(opt: str) -> str:
    a = accuracy_of(opt)
    return f"{100 * a['mean']:.1f}% ± {100 * a['std']:.1f} (5-fold CV)" if a else "not measured"


def greeting() -> str:
    h = dt.datetime.now().hour
    return "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"


def sample_images(n: int = 3) -> list:
    out = []
    for cls in C.CLASS_NAMES:
        # Hosted demo has no data/; it ships these same picks in DEMO_SAMPLE_DIR.
        files = (sorted((C.SPLIT_DIR / "test" / cls).glob("*"))
                 or sorted((C.DEMO_SAMPLE_DIR / cls).glob("*")))
        out += random.Random(C.RANDOM_SEED).sample(files, min(n, len(files)))
    return out


class _File:                                  # uniform wrapper for all input sources
    def __init__(self, name, data):
        self.name, self._data = name, data
    def getvalue(self):
        return self._data


# ======================================================================
# WELCOME
# ======================================================================
def welcome_screen():
    st.markdown("""
<div class="welcome glass">
  <div class="logo">🪸</div>
  <div class="eyebrow">MSc IT · Research prototype</div>
  <h1>Coral Reef Health Monitor</h1>
  <p style="opacity:.86;margin:0 0 .6rem">AI-assisted assessment of coral bleaching from underwater
  photographs - with underwater image restoration and visual explanations of every decision.</p>
  <div class="chips"><span>🌊 Underwater enhancement</span><span>🧠 4-model ensemble</span>
  <span>🔥 Grad-CAM explanations</span><span>📜 Printable certificates</span></div>
</div>""", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid, st.form("welcome", border=True):
        name = st.text_input("Your name", max_chars=40, placeholder="e.g. Hritikesh Pandey")
        role = st.selectbox("I am a…", ["Student", "Researcher / marine biologist",
                                        "Examiner / faculty", "Diver / citizen scientist",
                                        "Just exploring"])
        email = st.text_input("Email (optional)", max_chars=120, placeholder="you@example.org")
        org = st.text_input("Institution or organisation (optional)", max_chars=60)
        if visitor_db().configured:
            st.caption("These details are saved in the project's research database so we know who "
                       "used the prototype. Photos are never stored from this interface.")
        else:
            st.caption("These details are only used to personalise this session; nothing is saved.")
        go = st.form_submit_button("Dive in  🌊", type="primary", width="stretch")
    if go:
        if not name.strip():
            mid.warning("Please tell us your name first.")
        elif email.strip() and not contrib.EMAIL_RE.match(email.strip()):
            mid.warning("That email address does not look valid.")
        else:
            st.session_state.update(user=name.strip(), role=role, org=org.strip(),
                                    email=email.strip(), history=[],
                                    visitor_id=record_visit(name, role, email, org))
            st.rerun()


@st.cache_resource
def visitor_db() -> contrib.ContribDB:
    """Own replica file: the engine locks it, and the React API server may be
    running at the same time with replica.db."""
    return contrib.ContribDB("replica-streamlit.db")


def record_visit(name: str, role: str, email: str, org: str) -> int | None:
    """Store the welcome details in the Turso database (see src/contrib.py).
    A database problem must not lock anyone out of the app, so it only warns."""
    if not visitor_db().configured:
        return None
    try:
        return visitor_db().visit({"name": name, "role": role, "email": email,
                                 "institution": org}, "streamlit")
    except Exception as e:                      # network, auth, validation
        st.toast(f"Your details could not be saved: {e}")
        return None


# ======================================================================
# SIDEBAR
# ======================================================================
def sidebar() -> dict:
    with st.sidebar:
        st.markdown(f"### 👋 {html.escape(st.session_state.user)}")
        st.caption(" · ".join(x for x in (st.session_state.role, st.session_state.get("org")) if x))
        if st.button("Sign out", width="stretch"):
            for k in ("user", "role", "org", "email", "visitor_id", "saved_site", "history", "results"):
                st.session_state.pop(k, None)
            st.rerun()
        st.divider()

        st.markdown("#### 🧠 Model")
        opts = model_options()
        if not opts:
            st.error(f"No trained model in {C.ARTIFACT_DIR}. Run `python -m src.train` first.")
            st.stop()
        choice = st.selectbox("Model", opts, format_func=label_of, label_visibility="collapsed")
        a = accuracy_of(choice)
        if a:
            st.markdown(f'<div class="glass kpi"><div class="num">{100 * a["mean"]:.1f}%</div>'
                        f'<div class="lbl">± {100 * a["std"]:.1f} · 5-fold CV accuracy</div></div>',
                        unsafe_allow_html=True)
        if choice == ENSEMBLE:
            st.caption("Averages the predictions of MobileNetV2, ResNet50, EfficientNetB0 and "
                       "DenseNet121 - the most accurate option measured.")
        elif choice.startswith("baseline"):
            st.caption("⚠️ Trained from scratch; failed in 4 of 5 CV folds. For comparison only.")

        st.markdown("#### ⚙️ Settings")
        trained_raw = choice.endswith("_raw")
        enhance = st.toggle("Underwater enhancement", value=not trained_raw,
                            help="Gray-World → CLAHE → bilateral denoise → unsharp mask. "
                                 "Keep it matched to the model's training data.")
        if enhance == trained_raw:
            st.warning("This model was trained on "
                       f"{'raw' if trained_raw else 'enhanced'} images - predictions will "
                       "be less reliable.")
        show_cam = st.toggle("Grad-CAM explanation", value=True)
        floor = st.slider("Flag predictions below", 0.50, 0.95, 0.70, 0.05,
                          help="Results under this confidence are marked for expert review.")
        st.markdown("#### 📍 Survey details")
        site = st.text_input("Survey site (for the certificate)", max_chars=60,
                             placeholder="e.g. Gulf of Mannar, transect 3")
        vid = st.session_state.get("visitor_id")
        if vid and site.strip() and site.strip() != st.session_state.get("saved_site"):
            try:
                visitor_db().set_visitor_site(vid, site)
                st.session_state.saved_site = site.strip()
            except Exception as e:
                st.toast(f"The survey site could not be saved: {e}")
    return dict(choice=choice, enhance=enhance, show_cam=show_cam, floor=floor, site=site.strip())


# ======================================================================
# ANALYSIS
# ======================================================================
def run_models(choice: str) -> dict:
    members = C.ENSEMBLE_MEMBERS if choice == ENSEMBLE else [choice.rpartition("_")[0]]
    runs = [f"{m}_enhanced" for m in members] if choice == ENSEMBLE else [choice]
    return {m: load_model(r) for m, r in zip(members, runs)}


def explain(models: dict, x, idx) -> np.ndarray | None:
    """Mean Grad-CAM over the models, each resized to the input size."""
    maps = []
    for m in models.values():
        try:
            h, _, _ = compute_heatmap(m, x, class_index=idx)
            maps.append(cv2.resize(h.astype("float32"), C.IMG_SIZE))
        except Exception:                                    # noqa: BLE001
            continue
    if not maps:
        return None
    hm = np.mean(maps, axis=0)
    return hm / (hm.max() + 1e-8)


def analyse(file, s: dict, models: dict) -> dict | None:
    raw = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        st.error(f"Could not read {file.name} as an image.")
        return None
    processed, x = prepare(raw, enhance=s["enhance"])
    out = predict(models, x)
    idx, probs = out["index"], out["probs"]
    label, conf = C.CLASS_NAMES[idx], float(probs[idx])
    flagged = conf < s["floor"]
    now = dt.datetime.now()
    aid = "CRH-" + now.strftime("%Y%m%d") + "-" + hashlib.sha1(
        file.getvalue() + now.isoformat().encode()).hexdigest()[:4].upper()

    shown = cv2.resize(processed, C.IMG_SIZE)
    hm = explain(models, x, idx) if s["show_cam"] else None
    overlay = overlay_heatmap(shown, hm) if hm is not None else shown

    with st.container(border=True):
        st.markdown(f'<span class="eyebrow">Assessment {aid}</span>'
                    f'<h3 style="margin:.1rem 0 .8rem">{html.escape(file.name)}</h3>',
                    unsafe_allow_html=True)
        left, right = st.columns([1.45, 1], gap="large")
        with left:
            t1, t2, t3 = st.tabs(["Side by side", "Original", "What the model sees"])
            with t1:
                cols = st.columns(2 if hm is not None else 1)
                cols[0].image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), caption="Original", width="stretch")
                if hm is not None:
                    cols[1].image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                  caption="Grad-CAM - regions that drove the decision", width="stretch")
            t2.image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), width="stretch")
            t3.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), width="stretch",
                     caption=("Enhanced" if s["enhance"] else "Unprocessed") +
                             f" at training scale ({processed.shape[1]}×{processed.shape[0]} px)")
        with right:
            healthy = label == "healthy"
            meaning = ("Coral tissue shows normal pigmentation." if healthy else
                       "Pale or white tissue - consistent with loss of symbiotic algae.")
            st.markdown(f'<div class="verdict {"v-healthy" if healthy else "v-bleached"}">'
                        f'<div class="sub">Assessment</div>'
                        f'<div class="label">{"🟢" if healthy else "⚪"} {label.upper()}</div>'
                        f'<div class="sub">{meaning}</div></div>', unsafe_allow_html=True)
            st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence **{conf:.1%}**")
            if flagged:
                st.markdown('<div class="flag">⚠️ <b>Low confidence</b> - refer this image to '
                            'an expert before acting on it.</div>', unsafe_allow_html=True)
            if len(models) > 1:
                votes = "".join(
                    f'<div class="vote"><span>{NICE[n]}</span><b class="'
                    f'{"h" if C.CLASS_NAMES[int(p.argmax())] == "healthy" else "b"}">'
                    f'{C.CLASS_NAMES[int(p.argmax())]} {p.max():.0%}</b></div>'
                    for n, p in out["per_model"].items())
                agree = round(out["agreement"] * len(models))
                st.markdown(f'<div style="margin-top:.6rem"><span class="eyebrow">Model votes · '
                            f'{agree} of {len(models)} agree</span>{votes}</div>',
                            unsafe_allow_html=True)

        info = {"label": label, "confidence": conf,
                "time": now.strftime("%d %b %Y, %I:%M %p"), "assessment_id": aid,
                "model": label_of(s["choice"]).replace("  ★ recommended", ""),
                "model_accuracy": acc_text(s["choice"]),
                "agreement": (f"{round(out['agreement'] * len(models))} of {len(models)} models"
                              if len(models) > 1 else None),
                "site": s["site"] or None, "enhanced": s["enhance"],
                "authors": AUTHORS, "image_name": file.name}
        certificate = cert.render(st.session_state.user, raw, overlay, info)

        d1, d2, d3, _ = st.columns([1, 1, 1, 1.4])
        stem = file.name.rsplit(".", 1)[0]
        d1.download_button("📜 Certificate (PDF)", cert.to_pdf(certificate),
                           file_name=f"{aid}_{stem}.pdf", mime="application/pdf",
                           key=f"pdf{aid}", type="primary", width="stretch")
        d2.download_button("🖼️ Certificate (PNG)", cert.to_png(certificate),
                           file_name=f"{aid}_{stem}.png", mime="image/png",
                           key=f"png{aid}", width="stretch")
        with d3.popover("👁️ Preview", width="stretch"):
            st.image(certificate, width="stretch")

        with st.expander("📊 Image quality before / after enhancement"):
            q = pd.DataFrame({"Original": quality_report(raw), "Processed": quality_report(processed)})
            st.dataframe(q.rename(index={"colourfulness": "Colourfulness",
                                         "entropy": "Entropy (bits)", "rms_contrast": "RMS contrast"}),
                         width="stretch")
            st.caption("Dataset-wide, the enhancement raised contrast and entropy but lowered "
                       "colourfulness in 92 % of images (paper, Table 2).")

    return {"time": now.strftime("%Y-%m-%d %H:%M:%S"), "assessment_id": aid, "image": file.name,
            "prediction": label, "confidence": round(conf, 4), "needs_review": flagged,
            "model": info["model"], "enhanced": s["enhance"], "site": s["site"],
            "analyst": st.session_state.user}


def page_analyse(s: dict):
    source = st.segmented_control("Image source", ["📁 Upload", "📷 Camera", "🧪 Try a sample"],
                                  default="📁 Upload", label_visibility="collapsed")
    files = []
    if source == "📁 Upload":
        files = st.file_uploader("Drop underwater coral photographs here",
                                 type=["jpg", "jpeg", "png", "bmp", "webp"],
                                 accept_multiple_files=True) or []
    elif source == "📷 Camera":
        shot = st.camera_input("Point the camera at a coral photograph")
        if shot:
            files = [_File(f"camera_{dt.datetime.now():%H%M%S}.jpg", shot.getvalue())]
    elif source == "🧪 Try a sample":
        samples = sample_images()
        st.caption("Held-out test photographs the model never trained on. Pick any.")
        cols = st.columns(len(samples))
        for i, (col, p) in enumerate(zip(cols, samples)):
            col.image(str(p), width="stretch")
            if col.button(f"Analyse {p.parent.name}", key=f"s{i}", width="stretch"):
                st.session_state.sample = str(p)
        if st.session_state.get("sample"):
            p = st.session_state.sample
            files = [_File(p.split("/")[-1], open(p, "rb").read())]
            st.caption(f"True label of this sample: **{p.split('/')[-2]}**")

    if not files:
        st.markdown('<div class="glass" style="text-align:center;padding:2.4rem">'
                    '<div style="font-size:2.6rem">🐠</div><b>Ready when you are.</b><br>'
                    '<span style="opacity:.8">Upload one or more JPG/PNG coral photos, use your '
                    'camera, or try a sample from the test set.</span></div>', unsafe_allow_html=True)
        return

    models = run_models(s["choice"])
    rows = []
    for f in files:
        with st.spinner(f"Analysing {f.name}…"):
            r = analyse(f, s, models)
        if r:
            rows.append(r)
    if not rows:
        return
    hist = st.session_state.setdefault("history", [])
    seen = {(h["image"], h["model"], h["enhanced"]) for h in hist}
    hist += [r for r in rows if (r["image"], r["model"], r["enhanced"]) not in seen]
    st.session_state.history = hist[-300:]
    if len(rows) > 1:
        st.info(f"{len(rows)} images analysed - see the **Dashboard** tab for the batch summary.")


def kpis(df: pd.DataFrame):
    vals = [(len(df), "images analysed"), ((df.prediction == "healthy").sum(), "healthy"),
            ((df.prediction == "bleached").sum(), "bleached"),
            (f"{100 * df.confidence.mean():.0f}%" if len(df) else "-", "mean confidence"),
            (int(df.needs_review.sum()) if len(df) else 0, "need review")]
    for col, (n, l) in zip(st.columns(len(vals)), vals):
        col.markdown(f'<div class="glass kpi"><div class="num">{n}</div><div class="lbl">{l}</div></div>',
                     unsafe_allow_html=True)


def page_dashboard():
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("Analyse some images first - the dashboard summarises everything from this session.")
        return
    df = pd.DataFrame(hist)
    kpis(df)
    bleach_rate = (df.prediction == "bleached").mean()
    st.markdown(f'<div class="glass"><span class="eyebrow">Session bleaching index</span>'
                f'<h2 style="margin:.2rem 0">{100 * bleach_rate:.0f}% of photographs assessed as bleached</h2>'
                f'<span style="opacity:.8">Across {len(df)} image(s). Treat as indicative - per-image '
                f'accuracy is about 3 in 4.</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    counts = df.prediction.value_counts().rename_axis("class").reset_index(name="n")
    donut = alt.Chart(counts).mark_arc(innerRadius=70, cornerRadius=6).encode(
        theta="n:Q", color=alt.Color("class:N", scale=alt.Scale(domain=["healthy", "bleached"],
                                                                 range=[GREEN, SAND])),
        tooltip=["class", "n"]).properties(height=280, title="Assessment mix")
    hist_c = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("confidence:Q", bin=alt.Bin(step=.05), title="confidence"),
        y=alt.Y("count()", title="images"),
        color=alt.Color("prediction:N", scale=alt.Scale(domain=["healthy", "bleached"],
                                                        range=[GREEN, SAND]))
    ).properties(height=280, title="Confidence distribution")
    with c1, st.container(border=True):
        st.altair_chart(donut.configure(background="transparent"), width="stretch")
    with c2, st.container(border=True):
        st.altair_chart(hist_c.configure(background="transparent"), width="stretch")
    st.dataframe(df, width="stretch", hide_index=True,
                 column_config={"confidence": st.column_config.ProgressColumn(
                     "confidence", min_value=0, max_value=1, format="%.2f")})
    a, b, _ = st.columns([1, 1, 3])
    a.download_button("⬇️ Download report (CSV)", df.to_csv(index=False),
                      file_name=f"coral_assessments_{dt.date.today()}.csv", mime="text/csv",
                      width="stretch")
    if b.button("🗑️ Clear session", width="stretch"):
        st.session_state.history = []
        st.rerun()


def page_about():
    st.markdown('<div class="glass"><span class="eyebrow">Pipeline</span><h3>From photograph to assessment</h3>'
                '<ol style="line-height:1.95;margin-bottom:0">'
                '<li><b>Standardise</b> - resized to the scale of the training photos (300 px).</li>'
                '<li><b>Colour correction</b> - Gray-World white balance restores red light absorbed by water.</li>'
                '<li><b>Contrast</b> - CLAHE on the luminance channel.</li>'
                '<li><b>Denoise and sharpen</b> - bilateral filter, then unsharp mask.</li>'
                '<li><b>Classify</b> - ImageNet-pretrained CNNs fine-tuned on coral photographs; the '
                'ensemble averages four of them.</li>'
                '<li><b>Explain</b> - Grad-CAM shows which regions drove the decision.</li></ol></div>',
                unsafe_allow_html=True)
    rows = []
    for opt in [ENSEMBLE] + [f"{m}_enhanced" for m in C.MODELS] + [f"{C.PRIMARY_MODEL}_raw"]:
        a = accuracy_of(opt)
        if a:
            rows.append({"Model": label_of(opt).replace("  ★ recommended", ""),
                         "Accuracy": a["mean"], "± SD": a["std"],
                         "Macro F1": f"{a['f1']:.3f} ± {a['f1_std']:.3f}"})
    if rows:
        st.markdown('<div class="glass"><span class="eyebrow">Measured performance</span>'
                    '<h3 style="margin-bottom:0">5-fold cross-validation, 896 photographs</h3></div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1, format="%.3f"),
            "± SD": st.column_config.NumberColumn(format="%.3f")})
        st.caption("The four single pretrained models are statistically indistinguishable; the ensemble "
                   "is significantly better than three of them (McNemar, reports/table8).")
    st.markdown('<div class="glass"><span class="eyebrow">Limitations</span><h3>Read before relying on a result</h3>'
                '<ul style="line-height:1.85;margin-bottom:0">'
                '<li>Roughly <b>3 in 4</b> photographs are classified correctly - use it to triage, not to decide.</li>'
                '<li>Trained on 896 Flickr photos: clearer and better lit than routine survey footage.</li>'
                '<li>One label per photo; real reef frames often mix healthy and bleached colonies.</li>'
                '<li>Grad-CAM sometimes highlights open water or fish rather than coral - the models may '
                'partly rely on scene context.</li>'
                '<li>Best results: a close, well-lit photo filling the frame with coral.</li></ul></div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------- reef atlas
# Same content as the React "Reef atlas" page; both read C.REEF_ATLAS_JSON.
_MAP_LIBS = [("d3-array", "d3-array.min.js"), ("d3-geo", "d3-geo.min.js"),
             ("topojson-client", "topojson-client.min.js")]


@st.cache_data
def reef_atlas() -> dict:
    return json.loads(C.REEF_ATLAS_JSON.read_text(encoding="utf-8"))


@st.cache_data
def _map_scripts() -> str:
    """Inline the map libraries from web/node_modules so the map works offline;
    fall back to the CDN when the web app has not been installed."""
    out = []
    for pkg, f in _MAP_LIBS:
        local = C.WEB_DIR / "node_modules" / pkg / "dist" / f
        if local.exists():
            out.append(f"<script>{local.read_text(encoding='utf-8')}</script>")
        else:
            out.append(f'<script src="https://cdn.jsdelivr.net/npm/{pkg}@3/dist/{f}"></script>')
    return "".join(out)


def _reef_map_html(atlas: dict) -> str:
    land = C.WORLD_LAND_JSON.read_text(encoding="utf-8")
    return f"""
<style>
body {{ margin:0; font:15px/1.55 'Inter','Segoe UI',system-ui,sans-serif; color:#e9f7fa; background:transparent; }}
.wrap {{ display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; align-items:start; }}
@media (max-width:820px) {{ .wrap {{ grid-template-columns:minmax(0,1fr); }} }}
.box {{ background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14); border-radius:20px; padding:16px; min-width:0; }}
svg {{ display:block; width:100%; height:auto; touch-action:manipulation; }}
.sea {{ fill:rgba(2,30,48,.55); stroke:rgba(255,255,255,.14); }}
.grat {{ fill:none; stroke:rgba(160,225,240,.08); stroke-width:.5; }}
.land {{ fill:rgba(233,247,250,.16); stroke:rgba(233,247,250,.3); stroke-width:.5; }}
.tropic {{ fill:none; stroke:{CORAL}; stroke-dasharray:3 4; stroke-width:.7; opacity:.45; }}
.pin {{ cursor:pointer; outline:none; }}
.pin .halo {{ opacity:.25; }} .pin.on .halo {{ opacity:.4; }}
.pin .dot {{ stroke:#03192a; stroke-width:1.6; }}
.pin .ring {{ fill:none; stroke:#e9f7fa; stroke-width:1.8; }}
.pin:focus-visible .dot {{ stroke:#8FE3F0; stroke-width:2.5; }}
.pin.crit .halo {{ animation:pulse 2.4s ease-in-out infinite; }}
@keyframes pulse {{ 50% {{ opacity:.07; }} }}
@media (prefers-reduced-motion:reduce) {{ .pin.crit .halo {{ animation:none; }} }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin:12px 0; font-size:13px; }}
.legend span, .chips button {{ display:inline-flex; align-items:center; gap:7px; }}
i.sw {{ width:10px; height:10px; border-radius:50%; display:inline-block; flex:none; }}
.muted {{ color:rgba(220,240,246,.7); }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
.chips button {{ font:inherit; font-size:13px; color:inherit; cursor:pointer; border-radius:999px; padding:6px 12px;
  border:1px solid rgba(255,255,255,.16); background:rgba(255,255,255,.06); }}
.chips button[aria-pressed="true"] {{ background:#8FE3F0; color:#032033; border-color:transparent; font-weight:600; }}
.chips button:focus-visible {{ outline:2px solid #8FE3F0; outline-offset:2px; }}
.card {{ display:flex; flex-direction:column; gap:14px; }}
.card h2 {{ font:700 1.6rem/1.2 'Playfair Display',Georgia,serif; margin:0 0 2px; }}
.card p {{ margin:0; }}
.status {{ align-self:flex-start; padding:4px 11px; border-radius:999px; color:#03192a; font-size:11.5px;
  font-weight:700; letter-spacing:.1em; text-transform:uppercase; }}
dl {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:0; }}
dl div {{ background:rgba(255,255,255,.07); border-radius:12px; padding:10px 12px; min-width:0; }}
dt {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:rgba(220,240,246,.7); }}
dd {{ margin:3px 0 0; font-weight:600; font-variant-numeric:tabular-nums; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:2.2px; font-size:.72rem; color:#8FE3F0; font-weight:700; }}
.years {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }}
.years b {{ font-size:12px; padding:3px 8px; border-radius:6px; background:rgba(255,107,94,.16); color:#ffb4a2; }}
</style>
<div class="wrap">
  <div class="box">
    <svg id="map" viewBox="0 0 960 500" role="group" aria-label="World map of coral reef regions"></svg>
    <div class="legend" id="legend"></div>
    <div class="chips" id="chips" role="group" aria-label="Choose a reef"></div>
  </div>
  <article class="box card" id="card" aria-live="polite"></article>
</div>
{_map_scripts()}
<script>
const ATLAS = {json.dumps(atlas)};
const LAND = {land};
const LEVEL = ATLAS.levels, REEFS = ATLAS.reefs, NS = "http://www.w3.org/2000/svg";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]));
const el = (tag, attrs, parent) => {{ const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }};

const svg = document.getElementById("map");
const proj = d3.geoNaturalEarth1().fitExtent([[8, 8], [952, 492]], {{type: "Sphere"}});
const path = d3.geoPath(proj);
el("path", {{class: "sea", d: path({{type: "Sphere"}})}}, svg);
el("path", {{class: "grat", d: path(d3.geoGraticule10())}}, svg);
el("path", {{class: "land", d: path(topojson.feature(LAND, LAND.objects.land))}}, svg);
[23.44, -23.44].forEach((lat) => el("path", {{class: "tropic",
  d: path({{type: "LineString", coordinates: Array.from({{length: 73}}, (_, i) => [-180 + i * 5, lat])}})}}, svg));
const pinLayer = el("g", {{}}, svg);
const pins = {{}};
for (const r of REEFS) {{
  const [x, y] = proj([r.lon, r.lat]);
  const g = el("g", {{transform: `translate(${{x}},${{y}})`, class: `pin ${{r.level}}`, tabindex: 0, role: "button",
    "aria-label": `${{r.name}}, ${{LEVEL[r.level].label}}`}}, pinLayer);
  el("circle", {{class: "halo", r: 13, fill: LEVEL[r.level].color}}, g);
  el("circle", {{class: "ring", r: 9.5, visibility: "hidden"}}, g);
  el("circle", {{class: "dot", r: 5.5, fill: LEVEL[r.level].color}}, g);
  el("circle", {{r: 16, fill: "transparent"}}, g);
  g.addEventListener("click", () => select(r.id));
  g.addEventListener("keydown", (e) => {{ if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); select(r.id); }} }});
  pins[r.id] = g;
}}
document.getElementById("legend").innerHTML = Object.values(LEVEL)
  .map((l) => `<span><i class="sw" style="background:${{l.color}}"></i>${{l.label}}</span>`).join("")
  + '<span class="muted">Dashed lines: the tropics, where reef-building corals grow</span>';
const chips = document.getElementById("chips");
for (const r of REEFS) {{
  const b = document.createElement("button");
  b.type = "button"; b.dataset.id = r.id;
  b.innerHTML = `<i class="sw" style="background:${{LEVEL[r.level].color}}"></i>${{esc(r.name)}}`;
  b.addEventListener("click", () => select(r.id));
  chips.appendChild(b);
}}

function select(id) {{
  const r = REEFS.find((x) => x.id === id), lv = LEVEL[r.level];
  for (const k in pins) {{
    const on = k === id;
    pins[k].classList.toggle("on", on);
    pins[k].querySelector(".ring").setAttribute("visibility", on ? "visible" : "hidden");
    pins[k].setAttribute("aria-pressed", on);
  }}
  pinLayer.appendChild(pins[id]);
  for (const b of chips.children) b.setAttribute("aria-pressed", b.dataset.id === id);
  document.getElementById("card").innerHTML = `
    <span class="status" style="background:${{lv.color}}">${{lv.label}}</span>
    <div><h2>${{esc(r.name)}}</h2><p class="muted" style="font-size:13px">${{esc(r.where)}}</p></div>
    <dl>${{r.facts.map(([k, v]) => `<div><dt>${{esc(k)}}</dt><dd>${{esc(v)}}</dd></div>`).join("")}}</dl>
    <p>${{esc(r.note)}}</p>
    <div><div class="eyebrow">Major bleaching or damage years</div>
      <div class="years">${{r.events.map((y) => `<b>${{y}}</b>`).join("")}}</div></div>`;
  fit();
}}
// Grow the Streamlit iframe to fit, since the card stacks under the map on phones.
function fit() {{
  window.parent.postMessage({{isStreamlitMessage: true, type: "streamlit:setFrameHeight",
    height: document.documentElement.scrollHeight}}, "*");
}}
addEventListener("resize", fit);
select("gbr");
</script>"""


def page_reefs():
    atlas = reef_atlas()
    st.markdown('<div class="flag" style="display:flex;gap:.8rem;align-items:flex-start;font-size:.95rem;'
                'padding:.9rem 1.1rem;margin:0 0 1rem"><span style="flex:none;width:26px;height:26px;'
                'border-radius:50%;background:#ff6b5e;color:#03192a;display:grid;place-items:center;'
                'font-weight:800">!</span><span><b style="color:#ffb4a2">Coral reefs are in crisis.</b> '
                "The fourth global coral bleaching event, confirmed in April 2024, has exposed about 84% "
                "of the world's reef area to bleaching-level heat stress. The IUCN now lists 44% of "
                "reef-building coral species as threatened with extinction.</span></div>",
                unsafe_allow_html=True)
    st.markdown('<div class="glass"><span class="eyebrow">Reef atlas</span>'
                '<h3>The rainforests of the sea are turning white</h3>'
                '<p style="opacity:.85;margin:0">Coral reefs cover less than one percent of the ocean floor, '
                'yet about a quarter of all marine species depend on them. Tap a reef on the map to see where '
                'it is, how big it is, and how badly it has been hit.</p></div>', unsafe_allow_html=True)
    components.html(_reef_map_html(atlas), height=760)

    figs = "".join(
        f'<div style="background:rgba(255,255,255,.06);border-radius:14px;padding:1rem">'
        f'<div style="font:700 2rem/1 \'Playfair Display\',Georgia,serif;color:{CORAL}">{html.escape(f["value"])}</div>'
        f'<p style="margin:.4rem 0">{html.escape(f["text"])}</p>'
        f'<span style="font-size:.75rem;opacity:.7">{html.escape(f["source"])}</span></div>'
        for f in atlas["figures"])
    st.markdown('<div class="glass"><span class="eyebrow">Global picture</span>'
                "<h3>The state of the world's reefs</h3>"
                '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.8rem">'
                f"{figs}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="glass"><span class="eyebrow">Why it matters</span><h3>Reefs breathe with light</h3>'
                "<p>A coral is an animal, but it lives in partnership with microscopic algae called "
                "zooxanthellae inside its tissue. The algae photosynthesise: they use sunlight to make sugars "
                "and release oxygen. They supply up to 90% of the coral's food, and the oxygen they release "
                "helps sustain the fish, shellfish and other life packed into the reef.</p>"
                "<p>The ocean as a whole produces roughly half of the oxygen on Earth, most of it from "
                "free-floating phytoplankton. Reefs are a small but very productive part of that system. Their "
                "real value is the life they support: fisheries, medicines, tourism, and natural breakwaters "
                "that protect coastlines.</p>"
                '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.8rem;margin:.6rem 0">'
                f'<div style="border-top:2px solid {GREEN};padding-top:.5rem"><b>HEALTHY</b><br>Algae in the coral '
                "tissue photosynthesise, giving the coral colour, sugar and oxygen.</div>"
                '<div style="border-top:2px solid #ff6b5e;padding-top:.5rem"><b>HEAT STRESS</b><br>Water 1 °C or '
                "more above the usual summer maximum for several weeks.</div>"
                '<div style="border-top:2px solid #ff6b5e;padding-top:.5rem"><b>BLEACHED</b><br>The algae are '
                "expelled, the white skeleton shows through, and the coral starves.</div></div>"
                '<p style="opacity:.75;font-size:.9rem;margin:0">A bleached coral is not yet dead. If temperatures '
                "fall quickly it can take its algae back and recover. Repeated heatwaves leave too little time "
                "for that recovery.</p></div>", unsafe_allow_html=True)

    acts = "".join(f'<div style="border-top:1px solid rgba(255,255,255,.14);padding-top:.7rem">'
                   f'<b>{html.escape(a["title"])}</b><p style="opacity:.75;margin:.3rem 0 0;font-size:.92rem">'
                   f'{html.escape(a["text"])}</p></div>' for a in atlas["actions"])
    st.markdown('<div class="glass"><span class="eyebrow">What can be done</span><h3>Protecting what remains</h3>'
                '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem">'
                f"{acts}</div></div>", unsafe_allow_html=True)
    st.caption("Figures are rounded from public sources including IUCN, NOAA Coral Reef Watch, GCRMN, GBRMPA, "
               "UNESCO and the IPCC. The threat level for each region is this page's summary of recently "
               "reported bleaching and die-off, not an official IUCN category. Check the original source "
               "before citing any figure.")


# ======================================================================
# MAIN
# ======================================================================
inject_style()

if "user" not in st.session_state:
    welcome_screen()
else:
    s = sidebar()
    user = html.escape(st.session_state.user)
    st.markdown(f'<div class="glass hero"><div class="eyebrow">Coral Reef Health Monitor · '
                f'{dt.date.today():%A, %d %B %Y}</div><h1>{greeting()}, {user} 👋</h1>'
                f'<p>Assess coral bleaching from underwater photographs. Every result comes with a '
                f'confidence score, a visual explanation and a printable certificate.</p></div>',
                unsafe_allow_html=True)
    kpis(pd.DataFrame(st.session_state.get("history", []),
                      columns=["prediction", "confidence", "needs_review"]))
    t1, t2, t3, t4 = st.tabs(["🔬  Analyse", "📊  Dashboard", "📘  About the model", "🗺️  Reef atlas"])
    with t1:
        page_analyse(s)
    with t2:
        page_dashboard()
    with t3:
        page_about()
    with t4:
        page_reefs()

st.markdown(f'<div class="footer">🪸 AI-Based Coral Reef Health Monitoring Using Underwater Image '
            f'Processing · MSc IT Part II research project<br>{AUTHORS}</div>', unsafe_allow_html=True)
