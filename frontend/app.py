import streamlit as st
import requests
import html
import re
import streamlit.components.v1 as components
from typing import List, Optional, Tuple

# ============ CONFIG ============
st.set_page_config(page_title="Similarity Highlighter", layout="wide")
BACKEND_BASE = st.secrets.get("BACKEND_BASE", "http://localhost:8000")
MIN_PREFIX_WORDS = 3
USER1_ID = "user1"
USER2_ID = "user2"

# ============ STATE ============
for k, v in {
    # match indices & previous text
    "u1_match_idx": None, "u2_match_idx": None,
    "u1_prev_text": "", "u2_prev_text": "",

    # file-change signatures
    "u1_file_sig": None, "u2_file_sig": None,

    # textareas (must exist before first render)
    "u1_text": "", "u2_text": "",

    # clear-queue flags (handled BEFORE any widget renders)
    "u1_clear_next": False, "u2_clear_next": False,

    # optional UX
    "auto_clear_after_match": False,

    # scoring method shared across both panes
    "scoring_method": "json",  # "json" or "logprob"

    # similarity threshold
    "similarity_threshold": 0.5,  # default threshold for match detection
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _apply_clear(prefix: str) -> None:
    """Apply a clear to text and related state.
    IMPORTANT: this must run *before* widgets are instantiated.
    """
    st.session_state[f"{prefix}_text"] = ""
    st.session_state[f"{prefix}_match_idx"] = None
    st.session_state[f"{prefix}_prev_text"] = ""
    st.session_state[f"{prefix}_clear_next"] = False


def _queue_clear(prefix: str) -> None:
    """Safe way to request a clear from callbacks or post-match logic."""
    st.session_state[f"{prefix}_clear_next"] = True


# === HANDLE QUEUED CLEARS (must run before any widget is created) ===
if st.session_state.u1_clear_next:
    _apply_clear("u1")
if st.session_state.u2_clear_next:
    _apply_clear("u2")


# ============ HELPERS ============

def parse_disposition_filename(content_disposition: Optional[str]) -> Optional[str]:
    if not content_disposition:
        return None
    m = re.search(r"filename\*=(?:UTF-8'' )?([^;]+)".replace(" ",""), content_disposition, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip().strip('"')
        try:
            from urllib.parse import unquote
            return unquote(name)
        except Exception:
            return name
    m2 = re.search(r'filename="([^"]+)"', content_disposition, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)
    m3 = re.search(r'filename=([^;]+)', content_disposition, flags=re.IGNORECASE)
    if m3:
        return m3.group(1).strip().strip('"')
    return None


def fetch_sentences_and_name(user_id: str, user: int) -> Tuple[List[str], Optional[str]]:
    try:
        r = requests.get(f"{BACKEND_BASE}/file/{user_id}/{user}", timeout=10)
        if r.status_code == 404:
            return [], None
        r.raise_for_status()
        filename = parse_disposition_filename(r.headers.get("Content-Disposition"))
        text = r.text
        sentences = [s for s in text.split("\n") if s.strip()]
        return sentences, filename
    except Exception as e:
        st.sidebar.error(f"Failed to fetch file for User {user}: {e}")
        return [], None


def request_similarity(user_id: str, user: int, text: str, method: str, threshold: float = 0.0):
    payload = {
        "user_id": user_id,
        "user": user,
        "text": text,
        "min_prefix_words": MIN_PREFIX_WORDS,
        "method": method,
        "threshold": threshold,
        # "top_k": 1,  # default is 1
    }
    r = requests.post(f"{BACKEND_BASE}/similarity", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def render_scrollable_sentences(
    sentences: List[str],
    match_idx: Optional[int],
    color: str,
    height: int = 360,
    key: str = "list",
):
    # palettes
    if color == "red":
        bg, border = "#fde8e8", "#f8b4b4"
        base_color, off_color = "#ef4444", "#e5e7eb"  # on red, off gray
    else:
        bg, border = "#e1effe", "#a4cafe"
        base_color, off_color = "#3b82f6", "#e5e7eb"  # on blue, off gray

    has_match = match_idx is not None

    items_html = []
    for i, s in enumerate(sentences):
        safe_text = html.escape(s)
        is_match = (match_idx is not None and i == match_idx)
        cls = "row match" if is_match else "row"
        items_html.append(
            f'<div id="row-{key}-{i}" class="{cls}">'
            f'<span class="num">{i+1}</span>'
            f'<span class="txt">{safe_text}</span>'
            f'</div>'
        )

    target_id = f"row-{key}-{match_idx}" if match_idx is not None else ""

    html_content = f"""
    <div class="wrap">
      <div class="caption">
        <span class="status">
          <span class="lamp {'blink' if has_match else ''}" aria-label="match status"></span>
          <span class="cap-text">Shows all {len(sentences)} (auto-scrolls to match)</span>
        </span>
      </div>
      <div id="scroller-{key}" class="scroller">
        {''.join(items_html)}
      </div>
    </div>

    <style>
      .wrap {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
      .caption {{ font-size: 12px; color: #666; margin-bottom: 6px; }}
      .status {{ display: inline-flex; align-items: center; gap: 8px; }}
      .cap-text {{ vertical-align: middle; }}

      .scroller {{ border: 1px solid #e5e7eb; border-radius: 10px; height: {height}px; overflow-y: auto; padding: 8px; scroll-behavior: smooth; background: white; }}
      .row {{ display: grid; grid-template-columns: 48px 1fr; gap: 10px; align-items: start; padding: 6px 8px; border-radius: 8px; border: 1px solid transparent; margin-bottom: 6px; word-break: break-word; }}
      .row .num {{ color: #6b7280; font-variant-numeric: tabular-nums; }}
      .row.match {{ background: {bg}; border-color: {border}; color: #111111; }}

      /* Lamp */
      .lamp {{
        width: 14px; height: 14px; border-radius: 9999px; display: inline-block;
        background: {off_color};
        box-shadow: 0 0 0 0 rgba(0,0,0,0);
        position: relative;
        outline: 1px solid rgba(255,255,255,0.5);
      }}

      .lamp.blink {{
        animation: hardBlink 800ms steps(1, end) infinite;
      }}

      @keyframes hardBlink {{
        0%   {{ background: {off_color}; box-shadow: 0 0 0 0 rgba(0,0,0,0); }}
        49%  {{ background: {off_color}; box-shadow: 0 0 0 0 rgba(0,0,0,0); }}
        50%  {{ background: {base_color}; box-shadow: 0 0 10px 4px {base_color}; }}
        100% {{ background: {base_color}; box-shadow: 0 0 10px 4px {base_color}; }}
      }}
    </style>

    <script>
      (function() {{
        const targetId = {repr(target_id)};
        const container = document.getElementById("scroller-{key}");
        if (targetId && container) {{
          const el = document.getElementById(targetId);
          if (el) {{
            const top = el.offsetTop - (container.clientHeight / 2) + (el.clientHeight / 2);
            container.scrollTop = Math.max(0, top);
          }}
        }}
      }})();
    </script>
    """

    components.html(html_content, height=height + 40, scrolling=False)


# ============ SIDEBAR ============
st.sidebar.title("Upload base texts")

up1 = st.sidebar.file_uploader("Upload file for User 1", type=["txt"], key="up1")
if up1 is not None and st.sidebar.button("Set User 1 file", use_container_width=True, key="btn_u1"):
    try:
        files = {"file": (up1.name, up1.getvalue(), "text/plain")}
        r = requests.post(f"{BACKEND_BASE}/upload/{USER1_ID}/1", files=files, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to upload for User 1: {e}")
    else:
        _queue_clear("u1")
        st.sidebar.success(f"Uploaded {up1.name} for User 1")
        st.rerun()

up2 = st.sidebar.file_uploader("Upload file for User 2", type=["txt"], key="up2")
if up2 is not None and st.sidebar.button("Set User 2 file", use_container_width=True, key="btn_u2"):
    try:
        files = {"file": (up2.name, up2.getvalue(), "text/plain")}
        r = requests.post(f"{BACKEND_BASE}/upload/{USER2_ID}/2", files=files, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to upload for User 2: {e}")
    else:
        _queue_clear("u2")
        st.sidebar.success(f"Uploaded {up2.name} for User 2")
        st.rerun()

st.sidebar.markdown("---")
# ===== Scoring mode switcher =====
mode_label = st.sidebar.radio(
    "Scoring method",
    ("Prompt → numeric score", "Yes/No logprob (OpenAI)"),
    help=(
        "Choose how to score similarity.\n\n"
        "• Prompt → numeric: the model returns a JSON {score} in [0,1].\n"
        "• Yes/No logprob: the model must support token logprobs; we use P(Yes)."
    ),
)

st.session_state.scoring_method = "json" if mode_label.startswith("Prompt") else "logprob"

st.sidebar.markdown("---")
# ===== Similarity threshold slider =====
st.sidebar.markdown("**Similarity Threshold**")
threshold_value = st.sidebar.slider(
    "Minimum score for match detection",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.similarity_threshold,
    step=0.05,
    help=(
        "Set the minimum similarity score required to consider a match found.\n\n"
        "• 0.0: Any similarity score will be considered a match\n"
        "• 0.5: Moderate similarity required (recommended)\n"
        "• 0.8: High similarity required\n"
        "• 1.0: Only perfect matches"
    ),
)
st.session_state.similarity_threshold = threshold_value

# ============ FETCH SENTENCES ============
sent_u1, name_u1 = fetch_sentences_and_name(USER1_ID, 1)
sent_u2, name_u2 = fetch_sentences_and_name(USER2_ID, 2)

# Detect base text changes and reset highlights (before any rendering)
sig1 = (name_u1 or "", len(sent_u1))
if st.session_state.u1_file_sig != sig1:
    st.session_state.u1_file_sig = sig1
    _queue_clear("u1")
    st.rerun()

sig2 = (name_u2 or "", len(sent_u2))
if st.session_state.u2_file_sig != sig2:
    st.session_state.u2_file_sig = sig2
    _queue_clear("u2")
    st.rerun()

# ============ LAYOUT ============
colL, colM, colR = st.columns([1.6, 1.2, 1.6])

# --- MIDDLE: forms + similarity ---
with colM:
    # User 1
    st.subheader("User 1 Input (matches Window 1)")

    def _on_clear_u1():
        _queue_clear("u1")

    with st.form("u1_form", clear_on_submit=False):
        st.text_area(
            "Type for User 1",
            key="u1_text",
            height=120,
            label_visibility="collapsed",
            placeholder="Type for User 1…",
        )
        c1, c2 = st.columns(2)
        with c1:
            submitted_u1 = st.form_submit_button("🔍 Match (U1)", use_container_width=True)
        with c2:
            st.form_submit_button(
                "🧹 New input",
                use_container_width=True,
                on_click=_on_clear_u1,
            )

    if st.session_state.u1_clear_next:
        st.rerun()

    u1_val = st.session_state.get("u1_text", "")
    if not u1_val.strip():
        st.session_state.u1_match_idx = None

    if submitted_u1 and len(u1_val.split()) >= MIN_PREFIX_WORDS:
        with st.spinner("Matching…"):
            try:
                res = request_similarity(USER1_ID, 1, u1_val, st.session_state.scoring_method, st.session_state.similarity_threshold)
            except requests.RequestException as e:
                st.warning(f"U1 similarity failed: {e}")
            else:
                st.session_state.u1_match_idx = res.get("match_index") if res.get("match_found") else None
                llm_ms = res.get("llm_elapsed_ms")
                if llm_ms is not None:
                    st.caption(f"LLM time (U1): {llm_ms:.0f} ms")
                if res.get("match_found"):
                    best_score = res.get("best_score")
                    if isinstance(best_score, (int, float)):
                        st.metric("Best score (U1)", f"{best_score:.3f}")
                    else:
                        st.caption("Best score (U1): n/a")
                if st.session_state.auto_clear_after_match and st.session_state.u1_match_idx is not None:
                    _queue_clear("u1")

    # User 2
    st.subheader("User 2 Input (matches Window 2)")

    def _on_clear_u2():
        _queue_clear("u2")

    with st.form("u2_form", clear_on_submit=False):
        st.text_area(
            "Type for User 2",
            key="u2_text",
            height=120,
            label_visibility="collapsed",
            placeholder="Type for User 2…",
        )
        c3, c4 = st.columns(2)
        with c3:
            submitted_u2 = st.form_submit_button("🔎 Match (U2)", use_container_width=True)
        with c4:
            st.form_submit_button(
                "🧹 New input",
                use_container_width=True,
                on_click=_on_clear_u2,
            )

    if st.session_state.u2_clear_next:
        st.rerun()

    u2_val = st.session_state.get("u2_text", "")
    if not u2_val.strip():
        st.session_state.u2_match_idx = None

    if submitted_u2 and len(u2_val.split()) >= MIN_PREFIX_WORDS:
        with st.spinner("Matching…"):
            try:
                res = request_similarity(USER2_ID, 2, u2_val, st.session_state.scoring_method, st.session_state.similarity_threshold)
            except requests.RequestException as e:
                st.warning(f"U2 similarity failed: {e}")
            else:
                st.session_state.u2_match_idx = res.get("match_index") if res.get("match_found") else None
                llm_ms = res.get("llm_elapsed_ms")
                if llm_ms is not None:
                    st.caption(f"LLM time (U2): {llm_ms:.0f} ms")
                if res.get("match_found"):
                    best_score = res.get("best_score")
                    if isinstance(best_score, (int, float)):
                        st.metric("Best score (U2)", f"{best_score:.3f}")
                    else:
                        st.caption("Best score (U2): n/a")
                if st.session_state.auto_clear_after_match and st.session_state.u2_match_idx is not None:
                    _queue_clear("u2")

# --- LEFT: Window 1 ---
with colL:
    header_1 = f"Window 1 — {name_u1}" if name_u1 else "Window 1 — (no file uploaded)"
    st.subheader(header_1)
    if not sent_u1:
        st.info("Upload a file for User 1 in the sidebar to see sentences here.")
    render_scrollable_sentences(
        sentences=sent_u1,
        match_idx=st.session_state.u1_match_idx,
        color="red",
        height=360,
        key="left",
    )

# --- RIGHT: Window 2 ---
with colR:
    header_2 = f"Window 2 — {name_u2}" if name_u2 else "Window 2 — (no file uploaded)"
    st.subheader(header_2)
    if not sent_u2:
        st.info("Upload a file for User 2 in the sidebar to see sentences here.")
    render_scrollable_sentences(
        sentences=sent_u2,
        match_idx=st.session_state.u2_match_idx,
        color="blue",
        height=360,
        key="right",
    )