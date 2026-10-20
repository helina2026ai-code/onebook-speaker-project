"""Neural FAQ retrieval using the same model and embeddings as qa.sql."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
from threading import Lock
from urllib.parse import urlsplit
from collections.abc import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
DIMENSIONS = 384
MAX_QUESTION_CHARS = 500
LOGGER = logging.getLogger(__name__)
FALLBACK = "目前的問答資料中沒有足夠符合的內容。請補充想了解的療程、部位或具體問題，或透過診所官方管道聯繫客服。"
URGENT_REPLY = "若您目前已出現呼吸或吞嚥困難、突然看不清楚、劇烈疼痛，或注射部位皮膚變白、變灰、變藍，請立即就醫或聯絡當地緊急救護服務，並告知施作項目與時間。請同步通知施作診所，不要等待線上客服回覆。"
URGENT_PATTERN = re.compile(
    r"呼吸困難|無法呼吸|不能呼吸|喘不過氣|吞嚥困難|無法吞嚥|不能吞嚥|"
    r"視力模糊|視線模糊|看不清楚|看不見|視力改變|視力下降|失明|"
    r"劇痛|劇烈疼痛|皮膚變白|皮膚變灰|皮膚變藍|"
    r"difficulty breathing|difficulty swallowing|vision loss|blurred vision",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Match:
    question: str
    answer: str
    similarity: float


class SettingsError(ValueError):
    """Configuration error with a fixed, credential-free message for the UI."""

    def __init__(self, message: str, user_message: str):
        super().__init__(message)
        self.user_message = user_message


class SupabaseQueryError(RuntimeError):
    """Safe diagnostic code; never includes request headers or response bodies."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Settings:
    threshold: float = 0.45
    supabase_url: str = ""
    supabase_key: str = field(default="", repr=False)
    supabase_key_name: str = ""
    allow_local_fallback: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        return cls.from_mapping(os.environ)

    @classmethod
    def from_mapping(cls, values: Mapping) -> Settings:
        threshold_message = "QA_MATCH_THRESHOLD 必須是 -1 到 1 之間的數字，例如 0.45。"
        try:
            threshold = float(values.get("QA_MATCH_THRESHOLD", "0.45"))
        except (TypeError, ValueError):
            raise SettingsError("QA_MATCH_THRESHOLD must be between -1 and 1.", threshold_message) from None
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise SettingsError("QA_MATCH_THRESHOLD must be between -1 and 1.", threshold_message)
        url = str(values.get("SUPABASE_URL", "") or "").strip().rstrip("/")
        key_name = next((name for name in ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY") if str(values.get(name, "") or "").strip()), "")
        key = str(values.get(key_name, "") or "").strip() if key_name else ""
        if bool(url) != bool(key):
            raise SettingsError(
                "Set both SUPABASE_URL and a Supabase API key, or neither.",
                "請同時設定 SUPABASE_URL，以及 SUPABASE_ANON_KEY 或 SUPABASE_SERVICE_ROLE_KEY 其中一種金鑰。若要使用本機問答，請將 URL 與金鑰都留空。",
            )
        if url:
            try:
                parsed = urlsplit(url)
                valid_url = parsed.scheme == "https" and bool(parsed.netloc) and not any((parsed.username, parsed.password, parsed.path, parsed.query, parsed.fragment))
            except ValueError:
                valid_url = False
            if not valid_url:
                raise SettingsError(
                    "SUPABASE_URL must be an HTTPS project URL without a path or credentials.",
                    "SUPABASE_URL 請填專案的 HTTPS 網址，例如 https://your-project.supabase.co；不要填資料庫連線字串或 /rest/v1 路徑。",
                )
        fallback_value = str(values.get("QA_ALLOW_LOCAL_FALLBACK", "true")).strip().lower()
        if fallback_value not in {"true", "false", "1", "0", "yes", "no"}:
            raise SettingsError("QA_ALLOW_LOCAL_FALLBACK must be true or false.", "QA_ALLOW_LOCAL_FALLBACK 請設定為 true 或 false。")
        return cls(threshold=threshold, supabase_url=url, supabase_key=key, supabase_key_name=key_name,
                   allow_local_fallback=fallback_value in {"true", "1", "yes"})


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["question", "answer"]:
            raise ValueError("QA.csv must contain exactly question and answer columns.")
        rows = list(reader)
    if not rows or any(None in row or any(not isinstance(row.get(name), str) or not row[name].strip() for name in ("question", "answer")) for row in rows):
        raise ValueError("QA.csv contains an empty or malformed record.")
    return rows


def read_sql_vectors(sql_path: Path, csv_path: Path, rows: list[dict[str, str]]) -> np.ndarray | None:
    """Read inert INSERT literals; never execute the supplied SQL."""
    if not sql_path.exists():
        return None
    sql = sql_path.read_text(encoding="utf-8-sig")
    checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    required = [f"-- Source SHA-256: {checksum}", f"-- Model: {MODEL_NAME}", f"-- Model revision: {MODEL_REVISION}"]
    if not all(line in sql.splitlines()[:20] for line in required):
        return None
    pattern = re.compile(r"^    \('((?:[^']|'')*)', '((?:[^']|'')*)', '(\[[^\]]+\])'::extensions\.vector\(384\)\)[,]?$", re.MULTILINE)
    entries = pattern.findall(sql)
    pairs = [(q.replace("''", "'"), a.replace("''", "'")) for q, a, _ in entries]
    if pairs != [(row["question"], row["answer"]) for row in rows]:
        return None
    try:
        vectors = np.asarray([json.loads(v) for _, _, v in entries], dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if vectors.shape != (len(rows), DIMENSIONS) or not np.isfinite(vectors).all():
        return None
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        return None
    vectors.setflags(write=False)
    return vectors


class FAQService:
    def __init__(self, settings: Settings, csv_path: Path = ROOT / "QA.csv", sql_path: Path = ROOT / "qa.sql"):
        # Imports stay here so configuration and data checks can run without loading a model.
        import torch
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from sentence_transformers import SentenceTransformer

        self.settings = settings
        self._lock = Lock()
        self._csv_path = csv_path
        self._sql_path = sql_path
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        offline = os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
        try:
            model_path = snapshot_download(MODEL_NAME, revision=MODEL_REVISION, local_files_only=True)
        except LocalEntryNotFoundError:
            if offline:
                raise
            model_path = snapshot_download(
                MODEL_NAME, revision=MODEL_REVISION,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            )
        # Load the resolved directory, preventing tokenizer metadata lookups even
        # in library versions that ignore local_files_only for model IDs.
        self.model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
        self.model.max_seq_length = 512
        self.rows: list[dict[str, str]] = []
        self.vectors: np.ndarray | None = None
        if not settings.supabase_url:
            self._ensure_local_data()

    def _ensure_local_data(self) -> None:
        if self.vectors is not None:
            return
        with self._lock:
            if self.vectors is not None:
                return
            rows = load_rows(self._csv_path)
            vectors = read_sql_vectors(self._sql_path, self._csv_path, rows)
            if vectors is None:
                LOGGER.info("QA source changed or SQL embeddings unavailable; rebuilding in memory.")
                texts = [r["question"] + "\n" + r["answer"] for r in rows]
                if any(len(self.model.tokenizer(text, truncation=False)["input_ids"]) > 512 for text in texts):
                    raise ValueError("A QA record exceeds the model's 512-token limit; shorten or split it first.")
                vectors = self.model.encode(texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True)
            self.rows = rows
            self.vectors = vectors

    def search(self, question: str, count: int = 3, *, local_only: bool = False) -> list[Match]:
        use_remote = bool(self.settings.supabase_url) and not local_only
        if not use_remote:
            self._ensure_local_data()
        with self._lock:
            if len(self.model.tokenizer(question, truncation=False)["input_ids"]) > 512:
                raise ValueError("問題較長，請縮短後再送出。")
            query = self.model.encode(question, normalize_embeddings=True, convert_to_numpy=True)
        if use_remote:
            return self._search_supabase(query, count)
        assert self.vectors is not None
        scores = self.vectors @ query
        indices = np.argsort(-scores, kind="stable")[:count]
        return [Match(self.rows[i]["question"], self.rows[i]["answer"], float(scores[i])) for i in indices]

    def _search_supabase(self, query: np.ndarray, count: int) -> list[Match]:
        import httpx

        try:
            matches = self._request_supabase(query, count, self.settings.threshold)
            # A zero-result search may just be below threshold. Probe without a
            # cutoff before treating the remote dataset as unreadable or empty.
            if not matches and not self._request_supabase(query, 1, -1.0):
                raise SupabaseQueryError("no_readable_qa")
            return matches
        except httpx.TimeoutException:
            raise SupabaseQueryError("timeout") from None
        except httpx.HTTPStatusError as exc:
            raise SupabaseQueryError(f"http_{exc.response.status_code}") from None
        except httpx.HTTPError:
            raise SupabaseQueryError("connection") from None
        except (ValueError, KeyError, TypeError):
            raise SupabaseQueryError("invalid_response") from None

    def _request_supabase(self, query: np.ndarray, count: int, threshold: float) -> list[Match]:
        import httpx

        response = httpx.post(
            self.settings.supabase_url + "/rest/v1/rpc/match_qa",
            headers={"apikey": self.settings.supabase_key, "Authorization": "Bearer " + self.settings.supabase_key},
            json={"query_embedding": query.tolist(), "match_count": count, "match_threshold": threshold},
            timeout=httpx.Timeout(10.0, connect=3.0),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("Invalid search response.")
        matches = []
        for row in data:
            if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
                raise ValueError("Invalid search record.")
            similarity = float(row["similarity"])
            if not math.isfinite(similarity) or not row["question"].strip() or not row["answer"].strip():
                raise ValueError("Invalid search values.")
            matches.append(Match(row["question"], row["answer"], similarity))
        return sorted(matches, key=lambda match: match.similarity, reverse=True)[:count]

    def answer(self, question: str, *, local_only: bool = False) -> str:
        question = question.strip()
        if not question:
            return "請輸入想了解的問題，例如：第一次諮詢要怎麼預約？"
        if len(question) > MAX_QUESTION_CHARS:
            return f"請將問題縮短至 {MAX_QUESTION_CHARS} 字以內，再次送出。"
        # Urgent symptoms bypass similarity ranking, which can miss critical wording.
        if URGENT_PATTERN.search(question):
            return URGENT_REPLY
        matches = self.search(question, local_only=local_only)
        if not matches or matches[0].similarity < self.settings.threshold:
            return FALLBACK
        return matches[0].answer


@lru_cache(maxsize=1)
def get_service(settings: Settings | None = None) -> FAQService:
    return FAQService(settings if settings is not None else Settings.from_env())
