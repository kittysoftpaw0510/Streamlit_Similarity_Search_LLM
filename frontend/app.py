import streamlit as st
import requests
import html
import re
import streamlit.components.v1 as components
from typing import List, Optional, Tuple

# ============ CONFIG ============
st.set_page_config(page_title="Similarity Highlighter", layout="wide")
BACKEND_BASE = st.secrets.get("BACKEND_BASE", "http://localhost:8000")
MIN_PREFIX_WORDS = 5
USER1_ID = "user1"
USER2_ID = "user2"

# ============ STATE ============
for k, v in {
    "u1_typing": False, "u2_typing": False,
    "u1_match_idx": None, "u2_match_idx": None,
    "u1_prev_text": "", "u2_prev_text": "",
    "u1_file_sig": None, "u2_file_sig": None,   # signature to detect base text changes
    "u1_text": "", "u2_text": "",               # ensure textareas exist in session_state
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def _reset_user(prefix: str) -> None:
    """Reset per-user UI state (highlight, typing flag, previous text, input content)."""
    st.session_state[f"{prefix}_match_idx"] = None
    st.session_state[f"{prefix}_prev_text"] = ""
    st.session_state[f"{prefix}_typing"] = False
    st.session_state[f"{prefix}_text"] = ""

# ============ HELPERS ============
def parse_disposition_filename(content_disposition: Optional[str]) -> Optional[str]:
    """
    Extract filename from  Content-Disposition: attachment; filename="xyz.txt"
    Returns None if missing/unparseable.
    """
    if not content_disposition:
        return None
    # Try RFC 5987 / plain
    # filename*=UTF-8''encoded, or filename="..."
    m = re.search(r'filename\*=(?:UTF-8\'\')?([^;]+)', content_disposition, flags=re.IGNORECASE)
    if m:
        name = m.group(1)
        # strip surrounding quotes if any and percent-decode best-effort
        name = name.strip().strip('"')
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
    """
    GET /file/{user_id}/{user} -> return (sentences, filename)
    If file not present (404), return ([], None) and let UI handle it.
    """
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

def request_similarity(user_id: str, user: int, text: str):
    payload = {
        "user_id": user_id,
        "user": user,
        "text": text,
        "min_prefix_words": MIN_PREFIX_WORDS
    }
    r = requests.post(f"{BACKEND_BASE}/similarity", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def upload_file_to_user_slot(user_id: str, user: int, file) -> None:
    """
    POST /upload/{user_id}/{user} with multipart form (file)
    """
    files = {
        "file": (file.name, file.getvalue(), "text/plain")
    }
    r = requests.post(f"{BACKEND_BASE}/upload/{user_id}/{user}", files=files, timeout=20)
    r.raise_for_status()

def render_scrollable_sentences(
    sentences: List[str],
    match_idx: Optional[int],
    color: str,
    height: int = 360,
    key: str = "list"
):
    # color palette
    if color == "red":
        bg = "#fde8e8"
        border = "#f8b4b4"
        lamp = "🔴"
    else:
        bg = "#e1effe"
        border = "#a4cafe"
        lamp = "🔵"

    # Build list HTML safely with per-row IDs
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
      <div class="caption">{lamp} Shows all {len(sentences)} (auto-scrolls to match)</div>
      <div id="scroller-{key}" class="scroller">
        {''.join(items_html)}
      </div>
    </div>

    <style>
      .wrap {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      }}
      .caption {{
        font-size: 12px; color: #666; margin-bottom: 6px;
      }}
      .scroller {{
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        height: {height}px;
        overflow-y: auto;
        padding: 8px;
        scroll-behavior: smooth;
        background: white;
      }}
      .row {{
        display: grid;
        grid-template-columns: 48px 1fr;
        gap: 10px;
        align-items: start;
        padding: 6px 8px;
        border-radius: 8px;
        border: 1px solid transparent;
        margin-bottom: 6px;
        word-break: break-word;
      }}
      .row .num {{
        color: #6b7280;
        font-variant-numeric: tabular-nums;
      }}
      .row.match {{
        background: {bg};
        border-color: {border};
        color: #111111;
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
        upload_file_to_user_slot(USER1_ID, 1, up1)
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to upload for User 1: {e}")
    else:
        _reset_user("u1")  # clear highlight and text immediately
        st.sidebar.success(f"Uploaded {up1.name} for User 1")
        st.rerun()  # don't wrap this in try/except

up2 = st.sidebar.file_uploader("Upload file for User 2", type=["txt"], key="up2")
if up2 is not None and st.sidebar.button("Set User 2 file", use_container_width=True, key="btn_u2"):
    try:
        upload_file_to_user_slot(USER2_ID, 2, up2)
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to upload for User 2: {e}")
    else:
        _reset_user("u2")  # clear highlight and text immediately
        st.sidebar.success(f"Uploaded {up2.name} for User 2")
        st.rerun()

# ============ FETCH SENTENCES ============
sent_u1, name_u1 = fetch_sentences_and_name(USER1_ID, 1)
sent_u2, name_u2 = fetch_sentences_and_name(USER2_ID, 2)

# Detect base text changes and reset highlights (before any rendering)
sig1 = (name_u1 or "", len(sent_u1))
if st.session_state.u1_file_sig != sig1:
    st.session_state.u1_file_sig = sig1
    _reset_user("u1")

sig2 = (name_u2 or "", len(sent_u2))
if st.session_state.u2_file_sig != sig2:
    st.session_state.u2_file_sig = sig2
    _reset_user("u2")

# ============ LAYOUT (process inputs first, then render lists) ============
# IMPORTANT: compute matches BEFORE rendering left/right to avoid a one-rerun lag.
colL, colM, colR = st.columns([1.6, 1.2, 1.6])

# --- MIDDLE: forms + similarity (Ctrl+Enter submits) ---
with colM:
    # User 1 input
    st.subheader("User 1 Input (matches Window 1)")
    ib1, ib2 = st.columns(2)
    with ib1:
        if st.button("▶️ Start typing (U1)"):
            st.session_state.u1_typing = True
            st.session_state.u1_match_idx = None
    with ib2:
        if st.button("⏹ End typing (U1)"):
            st.session_state.u1_typing = False
            st.session_state.u1_match_idx = None

    with st.form("u1_form", clear_on_submit=False):
        st.text_area(
            "Type for User 1",
            key="u1_text",
            height=120,
            label_visibility="collapsed",
            placeholder="Start typing for User 1…",
        )
        submitted_u1 = st.form_submit_button("🔍 Match (U1)", use_container_width=True)

    u1_val = st.session_state.get("u1_text", "")
    # Optional nicety: clear highlight when the input is empty
    if not u1_val.strip():
        st.session_state.u1_match_idx = None

    if submitted_u1 or (
        st.session_state.u1_typing
        and u1_val != st.session_state.get("u1_prev_text", "")
        and len(u1_val.split()) >= MIN_PREFIX_WORDS
    ):
        st.session_state.u1_prev_text = u1_val
        try:
            with st.spinner("Matching…"):
                res = request_similarity(USER1_ID, 1, u1_val)
            st.session_state.u1_match_idx = res.get("match_index") if res.get("match_found") else None
        except Exception as e:
            st.warning(f"U1 similarity failed: {e}")

    # User 2 input
    st.subheader("User 2 Input (matches Window 2)")
    ib3, ib4 = st.columns(2)
    with ib3:
        if st.button("▶️ Start typing (U2)"):
            st.session_state.u2_typing = True
            st.session_state.u2_match_idx = None
    with ib4:
        if st.button("⏹ End typing (U2)"):
            st.session_state.u2_typing = False
            st.session_state.u2_match_idx = None

    with st.form("u2_form", clear_on_submit=False):
        st.text_area(
            "Type for User 2",
            key="u2_text",
            height=120,
            label_visibility="collapsed",
            placeholder="Start typing for User 2…",
        )
        submitted_u2 = st.form_submit_button("🔎 Match (U2)", use_container_width=True)

    u2_val = st.session_state.get("u2_text", "")
    # Optional nicety: clear highlight when the input is empty
    if not u2_val.strip():
        st.session_state.u2_match_idx = None

    if submitted_u2 or (
        st.session_state.u2_typing
        and u2_val != st.session_state.get("u2_prev_text", "")
        and len(u2_val.split()) >= MIN_PREFIX_WORDS
    ):
        st.session_state.u2_prev_text = u2_val
        try:
            with st.spinner("Matching…"):
                res = request_similarity(USER2_ID, 2, u2_val)
            st.session_state.u2_match_idx = res.get("match_index") if res.get("match_found") else None
        except Exception as e:
            st.warning(f"U2 similarity failed: {e}")

# --- LEFT: Window 1 (User 1 base text) ---
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

# --- RIGHT: Window 2 (User 2 base text) ---
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
