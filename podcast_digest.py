"""
Podcast Digest bot.

Daily run:
  1. Read each channel's YouTube RSS feed and find new videos.
  2. Skip Shorts and livestreams; check duration.
  3. Classify each video (one cheap text call for the whole batch).
     Personal-finance / off-topic videos get a one-line mention only.
  4. For stock-market videos:
       a. Transcribe in 20-minute chunks (Flash-Lite, low video resolution).
       b. Extract every learning into a structured digest (Flash, one call).
       c. Completeness check against the transcript (Flash-Lite) and merge.
  5. Send each digest to Telegram (split into parts if long) plus the
     transcript as a .txt file.

Public-repo safety: logs print only counts, video IDs and titles (public data).
Never print summaries, transcripts, keys or API responses.
"""

from __future__ import annotations

import html
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel, Field

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ROOT = Path(__file__).parent
CHANNELS_FILE = ROOT / "channels.json"
STATE_FILE = ROOT / "state.json"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Model chains: first model that works wins; a model whose daily quota is
# used up is skipped for the rest of the run.
SMART_MODELS = os.environ.get(
    "SMART_MODELS",
    "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite",
).split(",")
LITE_MODELS = os.environ.get(
    "LITE_MODELS",
    "gemini-3.5-flash-lite,gemini-3.1-flash-lite",
).split(",")

CHUNK_SECONDS = 20 * 60            # transcript chunk size
CHUNK_GAP_SECONDS = 40             # pause between video calls (tokens/min limit)
DAILY_VIDEO_BUDGET = 7 * 3600      # free tier allows 8h of YouTube a day
MAX_VIDEOS_PER_RUN = 15
MIN_SECONDS = 180                  # anything shorter is treated as a Short
FIRST_RUN_LOOKBACK_HOURS = 48      # on the very first run, only look back 2 days
MAX_ATTEMPTS = 3                   # give up on a video after this many failures
TG_LIMIT = 3900                    # Telegram hard limit is 4096 chars

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
YT_COOKIES = {"CONSENT": "YES+1", "SOCS": "CAI"}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            log("state.json unreadable; starting fresh")
    return {}


def save_state(state: dict) -> None:
    # keep the processed list from growing forever
    processed = state.get("processed", {})
    if len(processed) > 3000:
        keep = sorted(processed.items(), key=lambda kv: kv[1].get("date", ""))[-3000:]
        state["processed"] = dict(keep)
    STATE_FILE.write_text(json.dumps(state, indent=1, sort_keys=True))


# ----------------------------------------------------------------------------
# YouTube discovery
# ----------------------------------------------------------------------------
def resolve_channel_id(handle: str, state: dict) -> Optional[str]:
    cache = state.setdefault("channel_ids", {})
    if handle in cache:
        return cache[handle]
    try:
        r = requests.get(f"https://www.youtube.com/@{handle}", headers=UA,
                         cookies=YT_COOKIES, timeout=30)
        m = (re.search(r'"externalId":"(UC[\w-]{22})"', r.text)
             or re.search(r'youtube\.com/channel/(UC[\w-]{22})', r.text)
             or re.search(r'"channelId":"(UC[\w-]{22})"', r.text))
        if m:
            cache[handle] = m.group(1)
            return m.group(1)
    except requests.RequestException as e:
        log(f"channel lookup failed for @{handle}: {type(e).__name__}")
    log(f"could not find channel id for @{handle}")
    return None


ATOM = "{http://www.w3.org/2005/Atom}"
YT = "{http://www.youtube.com/xml/schemas/2015}"
MEDIA = "{http://search.yahoo.com/mrss/}"


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    videos = []
    for e in root.findall(f"{ATOM}entry"):
        vid = e.findtext(f"{YT}videoId")
        if not vid:
            continue
        group = e.find(f"{MEDIA}group")
        desc = group.findtext(f"{MEDIA}description") if group is not None else ""
        videos.append({
            "id": vid,
            "title": e.findtext(f"{ATOM}title") or "",
            "published": e.findtext(f"{ATOM}published") or "",
            "description": (desc or "")[:2000],
        })
    return videos


def fetch_feed(channel_id: str) -> list[dict]:
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return parse_feed(r.text)


def is_short(video_id: str) -> Optional[bool]:
    """/shorts/ID answers 200 for a Short and redirects for a normal video."""
    try:
        r = requests.head(f"https://www.youtube.com/shorts/{video_id}", headers=UA,
                          cookies=YT_COOKIES, allow_redirects=False, timeout=20)
        if r.status_code == 200:
            return True
        if 300 <= r.status_code < 400:
            return False
    except requests.RequestException:
        pass
    return None


def video_info(video_id: str) -> dict:
    """Duration and live status from the watch page. Missing values stay None."""
    info = {"seconds": None, "live": False, "upcoming": False}
    try:
        r = requests.get(f"https://www.youtube.com/watch?v={video_id}", headers=UA,
                         cookies=YT_COOKIES, timeout=30)
        t = r.text
        m = re.search(r'"lengthSeconds":"(\d+)"', t)
        if m:
            info["seconds"] = int(m.group(1))
        info["live"] = bool(re.search(r'"isLiveContent":true', t)) or '"isLive":true' in t
        info["upcoming"] = '"isUpcoming":true' in t
    except requests.RequestException as e:
        log(f"watch page failed for {video_id}: {type(e).__name__}")
    return info


# ----------------------------------------------------------------------------
# Gemini helpers
# ----------------------------------------------------------------------------
class QuotaExhausted(Exception):
    pass


class EndOfVideo(Exception):
    pass


_client: Optional[genai.Client] = None
_dead_models: set[str] = set()
CALLS = {"smart": 0, "lite": 0}


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


QUOTA_HITS: list[str] = []


def _quota_name(err: Exception) -> str:
    """Short name of the quota that was hit, from Google's error text."""
    s = str(err)
    m = re.search(r"quota[_ ]?(?:metric|id)?['\":\s]+([\w./-]+)", s, re.I)
    return m.group(1)[:80] if m else s[:120]


def _is_daily_quota(err: Exception) -> bool:
    s = str(err).lower()
    return "perday" in s or "per day" in s or "requests_per_day" in s or "daily" in s


def call_gemini(chain: list[str], contents, config: types.GenerateContentConfig,
                kind: str, end_probe: bool = False):
    """Try each model in turn. Returns the response. Raises QuotaExhausted when
    every model in the chain is out of daily quota. With end_probe, an HTTP 400
    means the requested video segment is past the end of the video."""
    last_err: Optional[Exception] = None
    for model in chain:
        model = model.strip()
        if not model or model in _dead_models:
            continue
        cfg = config
        for attempt in range(3):
            try:
                resp = client().models.generate_content(
                    model=model, contents=contents, config=cfg)
                CALLS[kind] += 1
                return resp
            except genai_errors.ClientError as e:
                last_err = e
                code = getattr(e, "code", None)
                if code == 429 and _is_daily_quota(e):
                    QUOTA_HITS.append(f"{model}: {_quota_name(e)}")
                    log(f"{model}: daily quota used up ({_quota_name(e)}), moving on")
                    _dead_models.add(model)
                    break
                if code == 429:
                    wait = 65 * (attempt + 1)
                    log(f"{model}: per-minute limit, waiting {wait}s")
                    time.sleep(wait)
                    continue
                if code == 400 and end_probe:
                    raise EndOfVideo() from e
                if code == 400 and getattr(cfg, "thinking_config", None) is not None \
                        and "think" in str(e).lower():
                    cfg = cfg.model_copy(update={"thinking_config": None})
                    continue  # this model doesn't take the thinking setting
                if code in (400, 403, 404):
                    # model not available to this key, or bad request for it
                    log(f"{model}: HTTP {code} ({str(e)[:200]}), trying next model")
                    if code == 404:
                        _dead_models.add(model)
                    break
                raise
            except genai_errors.ServerError as e:
                last_err = e
                if attempt >= 1:  # retried once already; failed calls still use quota
                    log(f"{model}: server error again ({str(e)[:150]}), trying next model")
                    break
                log(f"{model}: server error ({str(e)[:150]}), retrying in 30s")
                time.sleep(30)
    if all(m.strip() in _dead_models for m in chain if m.strip()):
        raise QuotaExhausted(kind)
    raise RuntimeError(f"all {kind} models failed: {type(last_err).__name__}: {str(last_err)[:200]}")


# ----------------------------------------------------------------------------
# Step 1: classify (one call for the whole batch)
# ----------------------------------------------------------------------------
CONTENT_TYPES = [
    "guest_podcast", "stock_analysis", "ipo_analysis", "event_coverage",
    "framework_education", "market_outlook", "personal_finance", "other",
]


class Classified(BaseModel):
    id: str
    content_type: str = Field(description="one of: " + ", ".join(CONTENT_TYPES))
    one_line: str = Field(description="One plain sentence on what the video covers")


class ClassifiedList(BaseModel):
    videos: list[Classified]


CLASSIFY_PROMPT = """You sort Indian stock-market YouTube videos by type, using only the title and description.

Types:
- guest_podcast: interview or conversation with a guest (fund manager, investor, founder, analyst)
- stock_analysis: analysis of one or a few specific listed companies
- ipo_analysis: review of an IPO or upcoming issue
- event_coverage: coverage of an event - Budget, RBI policy, results season, a market crash, a policy change, global news
- framework_education: teaches an investing framework, method, checklist or concept (valuation, screening, reading financials)
- market_outlook: views on the market, sectors or themes, without a guest
- personal_finance: tax, insurance, credit cards, loans, salary, retirement planning, mutual fund basics, budgeting, real estate, gold as personal savings
- other: promos, channel updates, unrelated topics

If a video mixes stock-market content with personal finance, pick the stock-market type.
Return one entry per video id, with a one-line description of what it covers.

Videos:
{videos}
"""


def classify(videos: list[dict]) -> dict[str, Classified]:
    payload = json.dumps([
        {"id": v["id"], "channel": v["channel_name"], "title": v["title"],
         "description": v["description"][:1200]} for v in videos
    ], ensure_ascii=False)
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=ClassifiedList,
        temperature=0.1)
    try:
        resp = call_gemini(LITE_MODELS, CLASSIFY_PROMPT.format(videos=payload), cfg, "lite")
        parsed: ClassifiedList = resp.parsed
        out = {c.id: c for c in parsed.videos}
    except Exception as e:  # noqa: BLE001
        log(f"classification failed ({type(e).__name__}); treating all as general")
        out = {}
    for v in videos:  # anything missing gets a safe default: full processing
        if v["id"] not in out:
            out[v["id"]] = Classified(id=v["id"], content_type="general", one_line=v["title"])
        elif out[v["id"]].content_type not in CONTENT_TYPES:
            out[v["id"]].content_type = "general"
    return out


# ----------------------------------------------------------------------------
# Step 2: transcribe in chunks
# ----------------------------------------------------------------------------
TRANSCRIBE_PROMPT = """Write a complete transcript of this video segment, in English.

- The segment runs from {start} to {end} of the full video. Use timestamps from the FULL video (the first line should be at or after {start}).
- Start each new speaker turn, and at least every 30 seconds, with a timestamp like [1:02:15] or [12:40].
- Label speakers by name if they are introduced (otherwise Host / Guest).
- If people speak Hindi or Hinglish, translate faithfully into English.
- Do NOT summarise, shorten or skip anything. Keep every number, percentage, year, price, company name, person, book and example exactly as spoken.
- If something on screen matters (a chart, table, company name, numbers on a slide), add it in square brackets, e.g. [Slide: FY25 revenue 1,240 cr].
- If the segment has no speech, reply only: NO_SPEECH
"""

TS_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]")


def fmt_ts(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def ts_to_seconds(ts: str) -> Optional[int]:
    if not ts:
        return None
    parts = re.findall(r"\d+", ts)
    if not parts or len(parts) > 3:
        return None
    nums = [int(p) for p in parts]
    while len(nums) < 3:
        nums.insert(0, 0)
    h, m, s = nums
    return h * 3600 + m * 60 + s


def _bracket_seconds(m: re.Match) -> int:
    a, b, c = m.group(1), m.group(2), m.group(3)
    if c is not None:
        return int(a) * 3600 + int(b) * 60 + int(c)
    return int(a) * 60 + int(b)


def fix_chunk_timestamps(text: str, start: int) -> str:
    """If the model used segment-relative timestamps, shift them to full-video time."""
    if start <= 0:
        return text
    first = TS_RE.search(text)
    if not first or _bracket_seconds(first) >= start - 90:
        return text
    return TS_RE.sub(lambda m: f"[{fmt_ts(_bracket_seconds(m) + start)}]", text)


UNKNOWN_LENGTH_CAP = 4 * 3600


def transcribe(video_id: str, seconds: Optional[int]) -> str:
    """Transcribe in chunks. If the length is unknown, keep going until the
    video runs out (empty chunk or an out-of-range error), up to 4 hours."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cfg = types.GenerateContentConfig(
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        temperature=0.1, max_output_tokens=32000)
    known = seconds is not None
    total = seconds if known else UNKNOWN_LENGTH_CAP
    pieces = []
    start = 0
    while start < total:
        end = min(start + CHUNK_SECONDS, total)
        if start:
            time.sleep(CHUNK_GAP_SECONDS)
        contents = types.Content(role="user", parts=[
            types.Part(file_data=types.FileData(file_uri=url),
                       video_metadata=types.VideoMetadata(
                           start_offset=f"{start}s", end_offset=f"{end}s")),
            types.Part(text=TRANSCRIBE_PROMPT.format(start=fmt_ts(start), end=fmt_ts(end))),
        ])
        try:
            resp = call_gemini(LITE_MODELS, contents, cfg, "lite",
                               end_probe=(not known and start > 0))
        except EndOfVideo:
            log(f"  reached end of video (about {fmt_ts(start)})")
            break
        text = (resp.text or "").strip()
        if text and text != "NO_SPEECH":
            pieces.append(fix_chunk_timestamps(text, start))
        elif not known and start > 0:
            break
        log(f"  transcribed {fmt_ts(start)}-{fmt_ts(end)}")
        start = end
    return "\n\n".join(pieces)


# ----------------------------------------------------------------------------
# Step 3: extract (structured) + completeness check
# ----------------------------------------------------------------------------
class Item(BaseModel):
    text: str
    timestamp: str = Field("", description="Where it is said, e.g. 34:10 or 1:02:15")
    tag: str = Field("", description="Short hashtag-style label for a theme/sector, e.g. powerTD, defence. Only for themes.")


class Stock(BaseModel):
    name: str
    nse_symbol: str = Field("", description="NSE symbol if known, e.g. KEI, RADICO. Empty if unsure.")
    stance: str = Field("Mentioned", description="Speaker's view: Bullish, Bearish, Neutral, Avoid, or Mentioned")
    thesis: str = Field("", description="Why - the speaker's full reasoning")
    details: list[str] = Field(default_factory=list, description="Every number, fact and point said about it")
    said_by: str = ""
    timestamp: str = ""


class Framework(BaseModel):
    name: str
    steps: list[str] = Field(default_factory=list)
    how_to_use: str = ""
    example: str = ""
    timestamp: str = ""


class Digest(BaseModel):
    content_type: str = Field(description="one of: " + ", ".join(CONTENT_TYPES))
    speakers: list[str] = Field(default_factory=list, description="Name and role, e.g. 'Rohan Mehta (guest, momentum investor)'")
    one_liner: str = ""
    verdict: str = Field("", description="Bottom-line view for stock or IPO analysis, else empty")
    stocks: list[Stock] = Field(default_factory=list)
    themes: list[Item] = Field(default_factory=list)
    frameworks: list[Framework] = Field(default_factory=list)
    key_facts: list[Item] = Field(default_factory=list, description="Numbers, data points, facts")
    ipo_details: list[Item] = Field(default_factory=list)
    bull_case: list[Item] = Field(default_factory=list)
    bear_case: list[Item] = Field(default_factory=list)
    valuation: list[Item] = Field(default_factory=list)
    event_impacts: list[Item] = Field(default_factory=list)
    market_calls: list[Item] = Field(default_factory=list, description="Calls on index/sectors with levels and timeframes")
    red_flags: list[Item] = Field(default_factory=list)
    takeaways: list[Item] = Field(default_factory=list, description="Lessons, mental models, process points, quotes worth keeping")
    books_people: list[Item] = Field(default_factory=list)


TYPE_FOCUS = {
    "guest_podcast": "a guest interview. Capture who the guest is, every stock and sector discussed, every framework or process they use (step by step), how they pick, size, hold and sell, mistakes and lessons they share, red flags they watch, market views, and books/people they mention.",
    "stock_analysis": "a stock analysis. Capture the business, every number (revenue, margins, growth, debt, order book, capacity, guidance), bull case, bear case, valuation (multiples, targets, comparisons), red flags, and the verdict.",
    "ipo_analysis": "an IPO analysis. Capture issue details (size, price band, dates, fresh issue vs OFS, use of proceeds, lot size, GMP if mentioned), financials, valuation vs listed peers, bull and bear points, red flags, and the subscribe/avoid verdict.",
    "event_coverage": "coverage of an event. Capture what happened with every number, which sectors and stocks are affected and how (positive/negative and why), market calls, and takeaways.",
    "framework_education": "a framework or concept lesson. Capture each framework step by step with thresholds and formulas, the worked examples (including stocks used as examples), common mistakes, and how to apply it.",
    "market_outlook": "a market outlook. Capture every call on the index, sectors and themes with levels and timeframes, the reasoning and data behind each, stocks mentioned, and risks.",
    "general": "a stock-market video. Capture every stock, theme, framework, number, view and lesson.",
    "personal_finance": "a personal finance video. Capture every rule, number and recommendation.",
    "other": "a video. Capture every substantive point.",
}

EXTRACT_PROMPT = """Below is the full transcript of a YouTube video from the channel "{channel}", titled "{title}". It is {focus}

This is NOT a summary. The reader will not watch the video, so every key learning must be here.
- Include EVERY stock or company mentioned, even in passing (stance "Mentioned" if no view). Put every number said about it in details.
- Keep numbers exactly as spoken, with units and time periods (e.g. "OPM 14% in FY24 to 18% in FY26E").
- Frameworks: list every step, threshold and formula. Include the example used.
- Attribute views to the person who said them.
- Give a timestamp (from the transcript) for every item.
- No limit on the number of items. If in doubt, include it.
- Leave a section empty if the video has nothing for it. Do not invent anything not in the transcript.
- content_type: re-check the type from the transcript ({types}).

TRANSCRIPT:
{transcript}
"""

CHECK_PROMPT = """You are checking notes taken from a video transcript for anything MISSED.

Compare the notes with the transcript. Return ONLY items that are missing from the notes or where the notes dropped an important number or step:
missed stocks or companies, numbers and facts, frameworks or steps, themes, market calls, red flags, lessons, books or people.
Use the same structure. Use empty lists for sections with nothing missed. Copy content_type, speakers and one_liner from the notes. Do not repeat items already in the notes.

NOTES (JSON):
{notes}

TRANSCRIPT:
{transcript}
"""

LIST_FIELDS = ["stocks", "themes", "frameworks", "key_facts", "ipo_details", "bull_case",
               "bear_case", "valuation", "event_impacts", "market_calls", "red_flags",
               "takeaways", "books_people"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())[:60]


def merge_missed(base: Digest, extra: Digest) -> int:
    """Append items from the check that are not already present. Returns count added."""
    added = 0
    for field in LIST_FIELDS:
        have = getattr(base, field)
        if field == "stocks":
            seen = {_norm(s.name) for s in have} | {_norm(s.nse_symbol) for s in have if s.nse_symbol}
            for s in getattr(extra, field):
                if _norm(s.name) in seen or (s.nse_symbol and _norm(s.nse_symbol) in seen):
                    # known stock: add any new detail lines
                    for h in have:
                        if _norm(h.name) == _norm(s.name) or (s.nse_symbol and _norm(h.nse_symbol) == _norm(s.nse_symbol)):
                            known = {_norm(d) for d in h.details}
                            for d in s.details:
                                if _norm(d) not in known:
                                    h.details.append(d)
                                    added += 1
                    continue
                have.append(s)
                seen.add(_norm(s.name))
                added += 1
        elif field == "frameworks":
            seen = {_norm(f.name) for f in have}
            for f in getattr(extra, field):
                if _norm(f.name) not in seen:
                    have.append(f)
                    added += 1
        else:
            seen = {_norm(i.text) for i in have}
            for i in getattr(extra, field):
                if _norm(i.text) not in seen:
                    have.append(i)
                    seen.add(_norm(i.text))
                    added += 1
    return added


JSON_SHAPE = """
Reply with ONE JSON object, no other text, in exactly this shape (lists may be empty):
{"content_type": "", "speakers": [""], "one_liner": "", "verdict": "",
 "stocks": [{"name": "", "nse_symbol": "", "stance": "Bullish|Bearish|Neutral|Avoid|Mentioned",
             "thesis": "", "details": [""], "said_by": "", "timestamp": "mm:ss"}],
 "themes": [{"text": "", "timestamp": "", "tag": ""}],
 "frameworks": [{"name": "", "steps": [""], "how_to_use": "", "example": "", "timestamp": ""}],
 "key_facts": [ITEM], "ipo_details": [ITEM], "bull_case": [ITEM], "bear_case": [ITEM],
 "valuation": [ITEM], "event_impacts": [ITEM], "market_calls": [ITEM], "red_flags": [ITEM],
 "takeaways": [ITEM], "books_people": [ITEM]}
where ITEM is {"text": "", "timestamp": ""}.
"""


def _thinking_low():
    try:
        return types.ThinkingConfig(thinking_level="low")
    except Exception:  # noqa: BLE001 - older SDK
        try:
            return types.ThinkingConfig(thinking_budget=2048)
        except Exception:  # noqa: BLE001
            return None


def parse_digest(text: str) -> Digest:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return Digest.model_validate(json.loads(t[start:end + 1]))


def ask_digest(chain: list[str], prompt: str, kind: str) -> Digest:
    """Ask each model in turn until one returns a usable digest."""
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json", temperature=0.2,
        max_output_tokens=32768, thinking_config=_thinking_low())
    all_quota = True
    last: Optional[Exception] = None
    for model in chain:
        if model.strip() in _dead_models:
            continue
        try:
            resp = call_gemini([model], prompt + JSON_SHAPE, cfg, kind)
            return parse_digest(resp.text)
        except QuotaExhausted:
            continue
        except Exception as e:  # noqa: BLE001
            all_quota = False
            last = e
            log(f"  {model}: no usable digest ({type(e).__name__}: {str(e)[:150]})")
    if all_quota and last is None:
        raise QuotaExhausted(kind)
    raise RuntimeError(f"no model produced a digest: {type(last).__name__}: {str(last)[:200]}")


def extract(video: dict, ctype: str, transcript: str) -> tuple[Digest, int]:
    prompt = EXTRACT_PROMPT.format(
        channel=video["channel_name"], title=video["title"],
        focus=TYPE_FOCUS.get(ctype, TYPE_FOCUS["general"]),
        types=", ".join(CONTENT_TYPES), transcript=transcript)
    digest = ask_digest(SMART_MODELS, prompt, "smart")

    added = 0
    try:
        check_prompt = CHECK_PROMPT.format(
            notes=digest.model_dump_json(), transcript=transcript)
        missed = ask_digest(LITE_MODELS, check_prompt, "lite")
        if missed is not None:
            added = merge_missed(digest, missed)
    except QuotaExhausted:
        log("  completeness check skipped (Flash-Lite daily quota used up)")
    except Exception as e:  # noqa: BLE001 - the check is a bonus, never fatal
        log(f"  completeness check skipped ({type(e).__name__})")
    return digest, added


# ----------------------------------------------------------------------------
# Step 4: format for Telegram
# ----------------------------------------------------------------------------
TYPE_LABEL = {
    "guest_podcast": "🎙️ Guest podcast",
    "stock_analysis": "🔍 Stock analysis",
    "ipo_analysis": "🆕 IPO analysis",
    "event_coverage": "📰 Event coverage",
    "framework_education": "🧠 Framework",
    "market_outlook": "🔭 Market outlook",
    "personal_finance": "💼 Personal finance",
    "general": "📺 New video",
    "other": "📺 New video",
}

SECTION_TITLE = {
    "verdict": "⚖️ Verdict",
    "stocks": "📈 Stocks mentioned",
    "themes": "🏭 Themes and sectors",
    "frameworks": "🧰 Frameworks",
    "key_facts": "🔢 Key numbers and facts",
    "ipo_details": "📝 Issue details",
    "bull_case": "🟢 Bull case",
    "bear_case": "🔴 Bear case",
    "valuation": "💰 Valuation",
    "event_impacts": "🎯 Impact on sectors and stocks",
    "market_calls": "🔭 Market calls",
    "red_flags": "🚩 Red flags",
    "takeaways": "💡 Takeaways",
    "books_people": "📚 Books and people mentioned",
}

TYPE_ORDER = {
    "guest_podcast": ["stocks", "themes", "frameworks", "key_facts", "market_calls", "red_flags", "takeaways", "books_people"],
    "stock_analysis": ["verdict", "stocks", "key_facts", "bull_case", "bear_case", "valuation", "red_flags", "takeaways"],
    "ipo_analysis": ["verdict", "ipo_details", "key_facts", "valuation", "bull_case", "bear_case", "red_flags", "stocks", "takeaways"],
    "event_coverage": ["key_facts", "event_impacts", "stocks", "themes", "market_calls", "takeaways"],
    "framework_education": ["frameworks", "key_facts", "stocks", "red_flags", "takeaways", "books_people"],
    "market_outlook": ["market_calls", "themes", "stocks", "key_facts", "red_flags", "takeaways"],
}
ALL_SECTIONS = ["verdict"] + LIST_FIELDS

STANCE_ICON = {"bullish": "🟢", "bearish": "🔴", "avoid": "⛔", "neutral": "⚪", "mentioned": "▫️"}


def esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def hashtag(raw: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]", "", raw or "")
    if not re.search(r"[A-Za-z]", tag):
        return ""
    return "#" + tag[:40]


def ts_link(video_id: str, ts: str) -> str:
    secs = ts_to_seconds(ts)
    if secs is None:
        return ""
    return f' <a href="https://youtu.be/{video_id}?t={secs}">{esc(fmt_ts(secs))}</a>'


def render_item(vid: str, it: Item, themed: bool = False) -> str:
    tag = ""
    if themed and it.tag:
        h = hashtag(it.tag)
        tag = f" {h}" if h else ""
    return f"• {esc(it.text)}{tag}{ts_link(vid, it.timestamp)}"


def render_stock(vid: str, s: Stock) -> str:
    tag = hashtag(s.nse_symbol) or hashtag(s.name.replace(" ", ""))
    icon = STANCE_ICON.get(s.stance.strip().lower(), "▫️")
    head = f"{icon} <b>{esc(s.name)}</b>"
    if tag:
        head += f" {tag}"
    head += f" · {esc(s.stance)}"
    if s.said_by:
        head += f" ({esc(s.said_by)})"
    head += ts_link(vid, s.timestamp)
    lines = [head]
    if s.thesis:
        lines.append(esc(s.thesis))
    lines += [f"  – {esc(d)}" for d in s.details]
    return "\n".join(lines)


def render_framework(vid: str, f: Framework) -> str:
    lines = [f"<b>{esc(f.name)}</b>{ts_link(vid, f.timestamp)}"]
    lines += [f"{i}. {esc(step)}" for i, step in enumerate(f.steps, 1)]
    if f.how_to_use:
        lines.append(f"<i>How to use:</i> {esc(f.how_to_use)}")
    if f.example:
        lines.append(f"<i>Example:</i> {esc(f.example)}")
    return "\n".join(lines)


def section_blocks(vid: str, d: Digest, section: str) -> list[str]:
    """Each returned block is self-contained HTML (safe to put in its own message)."""
    title = f"<b>{SECTION_TITLE[section]}</b>"
    if section == "verdict":
        return [f"{title}\n{esc(d.verdict)}"] if d.verdict else []
    items = getattr(d, section)
    if not items:
        return []
    if section == "stocks":
        rendered = [render_stock(vid, s) for s in items]
        sep = "\n\n"
    elif section == "frameworks":
        rendered = [render_framework(vid, f) for f in items]
        sep = "\n\n"
    else:
        rendered = [render_item(vid, i, themed=(section == "themes")) for i in items]
        sep = "\n"
    blocks, cur = [], title
    for r in rendered:
        if len(cur) + len(sep) + len(r) > TG_LIMIT - 200:
            blocks.append(cur)
            cur = f"{title} (cont.)"
        cur += ("\n" if cur.endswith("</b>") or cur.endswith("(cont.)") else sep) + r
    blocks.append(cur)
    return blocks


def render_digest(video: dict, d: Digest, added: int) -> list[str]:
    vid = video["id"]
    ctype = d.content_type if d.content_type in TYPE_LABEL else video.get("ctype", "general")
    header = [f"<b>{TYPE_LABEL.get(ctype, '📺 New video')} · {esc(video['channel_name'])}</b>",
              f"<b>{esc(video['title'])}</b>"]
    if d.speakers:
        header.append("👥 " + esc(", ".join(d.speakers)))
    meta = []
    if video.get("seconds"):
        meta.append(fmt_ts(video["seconds"]))
    meta.append(f'<a href="https://youtu.be/{vid}">Watch on YouTube</a>')
    header.append("⏱ " + " · ".join(meta))
    if d.one_liner:
        header.append(f"\n<b>📌 In one line</b>\n{esc(d.one_liner)}")
    blocks = ["\n".join(header)]

    order = TYPE_ORDER.get(ctype, ALL_SECTIONS)
    order = order + [s for s in ALL_SECTIONS if s not in order]  # never drop a section
    for sec in order:
        blocks += section_blocks(vid, d, sec)

    footer = "Guest's / creator's views, not advice"
    if added:
        footer += f" · completeness check added {added} point{'s' if added != 1 else ''}"
    blocks.append(f"<i>{esc(footer)}</i>\n#digest {hashtag(video['handle'])}")
    return pack_messages(blocks)


def pack_messages(blocks: list[str]) -> list[str]:
    msgs, cur = [], ""
    for b in blocks:
        if cur and len(cur) + 2 + len(b) > TG_LIMIT:
            msgs.append(cur)
            cur = b
        else:
            cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        msgs.append(cur)
    if len(msgs) > 1:
        msgs = [f"<i>Part {i}/{len(msgs)}</i>\n{m}" for i, m in enumerate(msgs, 1)]
    return msgs


def render_light_list(items: list[dict]) -> list[str]:
    lines = ["<b>💼 Also posted (personal finance / other)</b>"]
    for v in items:
        lines.append(f'• <b>{esc(v["channel_name"])}</b>: {esc(v["one_line"])} '
                     f'<a href="https://youtu.be/{v["id"]}">link</a>')
    return pack_messages(["\n".join(lines)])


# ----------------------------------------------------------------------------
# Telegram
# ----------------------------------------------------------------------------
def tg_send(text: str) -> None:
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML",
              "link_preview_options": {"is_disabled": True}},
        timeout=30)
    if r.status_code != 200:
        # retry once as plain text so a formatting slip never loses the content
        log(f"telegram HTML send failed ({r.status_code}); retrying as plain text")
        plain = re.sub(r"<[^>]+>", "", text)
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": html.unescape(plain)[:4096],
                  "link_preview_options": {"is_disabled": True}},
            timeout=30)
    time.sleep(1.1)


def tg_send_file(filename: str, content: str, caption: str) -> None:
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]},
        files={"document": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")},
        timeout=60)
    time.sleep(1.1)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def discover(state: dict, channels: list[dict]) -> list[dict]:
    processed = state.setdefault("processed", {})
    first_run = not processed and not state.get("initialised")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=FIRST_RUN_LOOKBACK_HOURS)
    known = state.setdefault("channels_seen", sorted(state.get("channel_ids", {}).keys()))
    new = []
    for ch in channels:
        is_new_channel = ch["handle"] not in known
        cid = ch.get("channel_id") or resolve_channel_id(ch["handle"], state)
        if not cid:
            continue
        try:
            feed = fetch_feed(cid)
        except Exception as e:  # noqa: BLE001
            log(f"feed failed for @{ch['handle']}: {type(e).__name__}")
            continue
        for v in feed:
            if v["id"] in processed:
                continue
            if first_run or is_new_channel:  # newly added channel: only the last 48h
                try:
                    pub = datetime.fromisoformat(v["published"].replace("Z", "+00:00"))
                except ValueError:
                    pub = cutoff
                if pub < cutoff:
                    processed[v["id"]] = {"date": v["published"][:10], "status": "seeded"}
                    continue
            v["handle"] = ch["handle"]
            v["channel_name"] = ch.get("name") or ch["handle"]
            new.append(v)
        if is_new_channel:
            known.append(ch["handle"])
    state["initialised"] = True
    # videos queued from earlier runs (they may have dropped off a busy feed)
    seen_ids = {v["id"] for v in new}
    for vid, v in list(state.get("pending", {}).items()):
        if vid in processed:
            state["pending"].pop(vid, None)
        elif vid not in seen_ids:
            new.append(v)
    new.sort(key=lambda v: v["published"])  # oldest first
    return new


def main() -> int:
    for name, val in [("GEMINI_API_KEY", GEMINI_API_KEY), ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
                      ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)]:
        if not val:
            log(f"missing secret {name}")
            return 1

    channels = json.loads(CHANNELS_FILE.read_text())["channels"]
    state = load_state()
    processed = state.setdefault("processed", {})
    attempts = state.setdefault("attempts", {})
    today = datetime.now(timezone.utc).date().isoformat()

    test_latest = os.environ.get("TEST_LATEST", "") == "1"
    if test_latest:
        # Test run: newest 4 feed entries per channel; Shorts are filtered below
        # and then only the newest real video per channel is kept.
        state["initialised"] = True
        new = []
        for ch in channels:
            cid = ch.get("channel_id") or resolve_channel_id(ch["handle"], state)
            if not cid:
                continue
            try:
                feed = sorted(fetch_feed(cid), key=lambda v: v["published"], reverse=True)
            except Exception as e:  # noqa: BLE001
                log(f"feed failed for @{ch['handle']}: {type(e).__name__}")
                continue
            for v in feed:
                if v["id"] in processed:
                    continue
                if len([n for n in new if n["handle"] == ch["handle"]]) < 4:
                    v["handle"] = ch["handle"]
                    v["channel_name"] = ch.get("name") or ch["handle"]
                    new.append(v)
                else:  # everything older counts as already seen
                    processed[v["id"]] = {"date": v["published"][:10], "status": "seeded"}
    else:
        new = discover(state, channels)
    log(f"{len(new)} new videos found")
    save_state(state)

    # Filter Shorts, livestreams, premieres not yet aired
    candidates = []
    for v in new:
        short = is_short(v["id"])
        info = video_info(v["id"])
        v["seconds"] = info["seconds"]
        if info["upcoming"]:
            log(f"skip (upcoming): {v['id']}")
            continue  # check again next run
        if info["live"]:
            processed[v["id"]] = {"date": today, "status": "live_skipped"}
            continue
        if short or (v["seconds"] is not None and v["seconds"] < MIN_SECONDS):
            processed[v["id"]] = {"date": today, "status": "short_skipped"}
            continue
        candidates.append(v)
    if test_latest:
        kept, seen_handles = [], set()
        for v in sorted(candidates, key=lambda v: v["published"], reverse=True):
            if v["handle"] in seen_handles:
                processed[v["id"]] = {"date": today, "status": "seeded"}
            else:
                seen_handles.add(v["handle"])
                kept.append(v)
        candidates = sorted(kept, key=lambda v: v["published"])
    save_state(state)
    log(f"{len(candidates)} videos after removing Shorts/live")
    if not candidates:
        return 0

    labels = classify(candidates)
    light, full = [], []
    for v in candidates:
        c = labels[v["id"]]
        v["ctype"] = c.content_type
        v["one_line"] = c.one_line
        (light if c.content_type in ("personal_finance", "other") else full).append(v)

    # Personal finance / other: one-line mentions only
    if light:
        for msg in render_light_list(light):
            tg_send(msg)
        for v in light:
            processed[v["id"]] = {"date": today, "status": "one_line"}
        save_state(state)

    budget = DAILY_VIDEO_BUDGET
    done, queued, failed, check_added = 0, 0, [], 0
    quota_out = False
    reasons: list[tuple[dict, str]] = []
    pending = state.setdefault("pending", {})

    def queue(v: dict) -> None:
        pending[v["id"]] = {k: v[k] for k in ("id", "title", "published", "description",
                                              "handle", "channel_name")}

    for v in full:
        secs = v["seconds"] or 3600  # unknown length: budget 1h; transcribe() finds the end
        if quota_out or done >= MAX_VIDEOS_PER_RUN or secs > budget:
            queued += 1
            queue(v)
            why = ("Gemini daily quota" if quota_out else
                   "max videos per run" if done >= MAX_VIDEOS_PER_RUN else
                   f"7h video budget ({fmt_ts(budget)} left, video {fmt_ts(secs)})")
            reasons.append((v, why))
            continue
        log(f"processing {v['id']} ({v['channel_name']}, {fmt_ts(secs)}, {v['ctype']})")
        try:
            transcript = transcribe(v["id"], v["seconds"])
            if not transcript.strip():
                raise RuntimeError("empty transcript")
            digest, added = extract(v, v["ctype"], transcript)
            check_added += added
            for msg in render_digest(v, digest, added):
                tg_send(msg)
            tg_send_file(f"{v['handle']}_{v['id']}_transcript.txt",
                         f"{v['title']}\nhttps://youtu.be/{v['id']}\n\n{transcript}",
                         "Full transcript")
            processed[v["id"]] = {"date": today, "status": "done", "type": digest.content_type}
            attempts.pop(v["id"], None)
            pending.pop(v["id"], None)
            budget -= secs
            done += 1
        except QuotaExhausted:
            log("Gemini daily quota reached; remaining videos wait for tomorrow")
            quota_out = True
            queued += 1
            queue(v)
            reasons.append((v, "Gemini daily quota"))
        except Exception as e:  # noqa: BLE001
            n = attempts.get(v["id"], 0) + 1
            attempts[v["id"]] = n
            log(f"  failed ({type(e).__name__}: {str(e)[:300]}), attempt {n}/{MAX_ATTEMPTS}")
            if n >= MAX_ATTEMPTS:
                processed[v["id"]] = {"date": today, "status": "failed"}
                attempts.pop(v["id"], None)
                pending.pop(v["id"], None)
                failed.append(v)
            else:
                queued += 1
                queue(v)
                reasons.append((v, f"error, will retry ({type(e).__name__}: {str(e)[:120]})"))
        save_state(state)

    # Run note only when something needs your attention
    notes = []
    for v, why in reasons:
        notes.append(f'Queued for next run: <a href="https://youtu.be/{v["id"]}">{esc(v["title"])}</a> '
                     f"({esc(v['channel_name'])}) - {esc(why)}")
    if QUOTA_HITS:
        notes.append("Limits hit: " + esc("; ".join(dict.fromkeys(QUOTA_HITS))))
    for v in failed:
        notes.append(f'Could not process <a href="https://youtu.be/{v["id"]}">{esc(v["title"])}</a> '
                     f"after {MAX_ATTEMPTS} tries")
    if notes:
        tg_send("<b>ℹ️ Digest run note</b>\n" + "\n".join(f"• {n}" for n in notes))

    log(f"done={done} queued={queued} failed={len(failed)} one_line={len(light)} "
        f"check_added={check_added} calls={CALLS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
