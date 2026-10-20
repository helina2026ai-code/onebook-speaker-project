"""Streamlit customer-service application."""

from __future__ import annotations

import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

from app_config import read_settings
from qa_engine import MAX_QUESTION_CHARS, Settings, SettingsError, SupabaseQueryError, get_service

LOGGER = logging.getLogger(__name__)
EXAMPLES = [
    "第一次諮詢要怎麼預約？",
    "預約後可以改時間嗎？",
    "可以刷卡或分期付款嗎？",
    "療程前正在吃藥，需要停藥嗎？",
]
CSS = """
<style>
.stMainBlockContainer { max-width: 1000px; padding-top: 2.5rem; padding-bottom: 2rem; }
.care-brand { color: #13756d; font-weight: 700; letter-spacing: .12em; font-size: 13px; }
[data-testid="stMarkdownContainer"] h1 { font-size: clamp(28px, 5vw, 42px); line-height: 1.4; margin: .5rem 0; }
.care-intro { color: #586873; line-height: 1.9; margin-bottom: 1.5rem; }
[data-testid="stChatMessage"] { border-radius: 16px; }
@media (max-width: 640px) {
  .stMainBlockContainer { padding-top: 4rem; }
  [data-testid="stMarkdownContainer"] h1 { font-size: 28px !important; }
}
</style>
"""


def respond(message: str, settings: Settings | None = None) -> str:
    return respond_with_source(message, settings)[0]


def respond_with_source(message: str, settings: Settings | None = None) -> tuple[str, str]:
    message = message.strip()
    if not message:
        return "請先輸入您的問題。", ""
    if len(message) > MAX_QUESTION_CHARS:
        return f"請將問題縮短至 {MAX_QUESTION_CHARS} 字以內，再次送出。", ""
    try:
        settings = settings if settings is not None else read_settings()
        service = get_service(settings)
        try:
            return service.answer(message), ""
        except SupabaseQueryError as exc:
            LOGGER.warning("Supabase query unavailable (%s).", exc.code)
            if not settings.allow_local_fallback:
                return "目前雲端問答服務暫時無法使用，請稍後再試或聯繫診所客服。", ""
            answer = service.answer(message, local_only=True)
            return answer, "目前依內建問答資料回覆。"
    except ValueError:
        LOGGER.warning("FAQ input or configuration could not be processed.")
        return "暫時無法處理這個問題，請簡短描述您的需求後再試一次。若問題持續，請聯繫診所客服。", ""
    except Exception:
        LOGGER.error("FAQ search is temporarily unavailable.")
        return "目前查詢服務暫時無法使用，請稍後再試，或透過診所官方管道聯繫客服。", ""


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pop("pending_question", None)
    st.session_state.pop("question", None)


def use_example(question: str) -> None:
    st.session_state.pending_question = question


def main() -> None:
    st.set_page_config(page_title="醫美諮詢室｜線上客服", page_icon="💬", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('''<div class="care-brand">CARE / 醫美諮詢室</div>
        <h1 id="care-heading">從一個問題，開始了解。</h1>
        <p class="care-intro">預約方式、療程資訊、術前準備與術後照護。<br>
        提出您的問題，取得即時回覆。</p>''', unsafe_allow_html=True)

    try:
        try:
            secrets = st.secrets.to_dict()
        except FileNotFoundError:
            secrets = {}
        settings = read_settings(secrets=secrets)
    except SettingsError as exc:
        st.error(exc.user_message)
        st.stop()
    except Exception:
        st.error("無法讀取服務設定。請確認 .env 可正常讀取，以及 Streamlit Secrets 使用有效的 TOML 格式。")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.sidebar:
        st.subheader("今天想了解什麼？")
        st.write("初次諮詢與預約、費用與付款、療程準備，以及術後照護與回診。")
        st.caption("請一次描述一個問題，並附上療程或部位，方便找到相關資訊。")
        st.divider()
        st.markdown("**需要個別評估？**")
        st.write("療程選擇、用藥及身體狀況，請安排醫師諮詢。")
        st.button("清除對話", on_click=clear_conversation, width="stretch")

    if not st.session_state.messages:
        st.markdown("**您可以直接輸入問題，或從這些問題開始：**")
        columns = st.columns(2)
        for index, example in enumerate(EXAMPLES):
            columns[index % 2].button(example, key=f"example_{index}", on_click=use_example, args=(example,), width="stretch")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # User and source text are never treated as HTML or instructions.
            st.markdown(message["content"])
            if message.get("source_note"):
                st.caption(message["source_note"])

    prompt = st.chat_input("輸入您的問題，例如：第一次諮詢需要帶什麼？", key="question", max_chars=MAX_QUESTION_CHARS)
    pending_question = st.session_state.pop("pending_question", None)
    question = prompt or pending_question
    if question and question.strip():
        question = question.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("正在查詢相關資訊…"):
                answer, source_note = respond_with_source(question, settings)
            st.markdown(answer)
            if source_note:
                st.caption(source_note)
        st.session_state.messages.append({"role": "assistant", "content": answer, "source_note": source_note})
        # Keep each session bounded without caching or sharing conversation text.
        st.session_state.messages = st.session_state.messages[-40:]

    st.caption("本服務依問答資料提供一般資訊；實際療程、價格與預約安排，請由診所及醫師確認。若有突發或嚴重不適，請立即就醫。")


if __name__ == "__main__":
    main()
