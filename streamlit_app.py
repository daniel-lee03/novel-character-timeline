# -*- coding: utf-8 -*-
# 실행: streamlit run --server.port 3000 --server.address 0.0.0.0 streamlit_app.py

import os
import re
from typing import List, Dict, Any, Tuple

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# --------------------------------
# 환경 로드
# --------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

st.set_page_config(
    page_title="소설 인물 등장 비율 타임라인",
    layout="wide",
    page_icon="📚",
)

st.title("📚 소설 인물 등장 비율 타임라인")
st.caption("Streamlit · GitHub Codespaces · (선택) Hugging Face Inference API")

# --------------------------------
# 유틸
# --------------------------------
HANGUL_ALNUM = r"[가-힣A-Za-z0-9]"

def split_sentences_kr(text: str) -> List[str]:
    """
    한국어에 간단히 작동하는 문장 분리기(가벼운 정규식 기반).
    마침표/물음표/느낌표/줄바꿈 등을 기준으로 분리 후 트림.
    """
    text = re.sub(r"\r\n|\r", "\n", text)
    chunks = re.split(r"(?<=[.!?…。！？])\s+|\n{2,}", text)
    sents = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        sents.extend([s.strip() for s in re.split(r"\n+", c) if s.strip()])
    return sents

def make_windows(items: List[str], window: int, step: int) -> List[Tuple[int, int]]:
    """슬라이딩 윈도우 구간(index start,end exclusive) 리스트 생성"""
    spans = []
    n = len(items)
    if n == 0:
        return spans
    i = 0
    while i < n:
        j = min(i + window, n)
        spans.append((i, j))
        if j == n:
            break
        i += step
    return spans

def compile_alias_regex(aliases: List[str]) -> re.Pattern:
    """
    별칭 리스트를 하나의 안전한 정규식으로 컴파일.
    한국어 단어 경계 유사 처리를 위해 주변에 한글/영문/숫자 아닌 경우만 매칭.
    """
    escaped = [re.escape(a.strip()) for a in aliases if a.strip()]
    if not escaped:
        escaped = ["a^"]  # 절대 매칭되지 않도록
    pattern = rf"(?<!{HANGUL_ALNUM})(?:{'|'.join(escaped)})(?!{HANGUL_ALNUM})"
    return re.compile(pattern)

def count_mentions(text: str, alias_re: re.Pattern) -> int:
    return len(alias_re.findall(text))

def soft_normalize_whitespace(text: str) -> str:
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t.strip()

# --------------------------------
# (선택) HF Inference API로 한국어 NER 호출
# --------------------------------
def hf_ner(text: str, model_id: str, token: str) -> List[Dict[str, Any]]:
    """
    Hugging Face Inference API Token Classification 호출.
    반환 예: [{"entity_group":"PER","word":"홍길동",...}, ...]
    """
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {
        "inputs": text,
        "parameters": {"aggregation_strategy": "simple"},
        "options": {"wait_for_model": True},
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # 리스트/딕셔너리 다양한 포맷 대응 → 평탄화
    flat: List[Any] = []
    def _flatten(x):
        if isinstance(x, list):
            for e in x:
                _flatten(e)
        else:
            flat.append(x)

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"])
    _flatten(data)
    return flat

def extract_person_aliases_by_ner(text: str, model_id: str, token: str, top_k: int = 10) -> List[str]:
    ents = hf_ner(text, model_id, token)
    persons = []
    for e in ents:
        group = (e.get("entity_group") or e.get("entity") or "").upper()
        if group.startswith("PER"):
            w = (e.get("word") or "").strip("# ").strip()
            if len(w) >= 2:
                persons.append(w)
    if not persons:
        return []
    s = pd.Series(persons).value_counts()
    return list(s.head(top_k).index)

# --------------------------------
# 사이드바 옵션
# --------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    mode = st.radio("등장인물 추출 모드", ["룰기반(권장)", "Hugging Face NER(API)"], help="수업/배포 안정성은 룰기반이 좋습니다.")
    window = st.number_input("윈도우(문장 수)", min_value=5, max_value=200, value=20, step=1)
    step = st.number_input("슬라이드 간격(문장 수)", min_value=1, max_value=200, value=10, step=1)
    show_mode = st.selectbox("차트 종류", ["선형(Line)", "면적(Stacked Area)"])
    smooth = st.checkbox("부드러운 곡선(spline)", value=True)
    show_counts = st.checkbox("등장 비율과 함께 개수도 툴팁에 표시", value=True)

    st.markdown("---")
    st.subheader("데이터 입력")
    txt_file = st.file_uploader(".txt 업로드 (선택)", type=["txt"])
    default_demo = st.checkbox("데모용 짧은 텍스트 자동 삽입", value=False)

# --------------------------------
# 본문 입력부
# --------------------------------
text_default = (
    "오늘을 얼마나 기다렸는지 모른다. 나는 지난밤 싸 놓은 짐을 다시 점검했다. "
    "세면도구, 수건, 휴대폰 충전기, 그리고 어젯밤 은수에게서 빌린 책. "
    "은수는 어젯밤 내게 말했다. '민수야, 꼭 돌아와.' 나는 웃으며 대답했다. "
    "버스 정류장에는 진호가 서 있었다. 진호는 내 어깨를 톡 치며 말했다. '가자.'"
)

if default_demo and not txt_file:
    novel_text = text_default
else:
    novel_text = ""

if txt_file is not None:
    try:
        novel_text = txt_file.read().decode("utf-8")
    except Exception:
        novel_text = txt_file.read().decode("cp949", errors="ignore")

novel_text = st.text_area(
    "소설 본문을 붙여넣으세요",
    value=soft_normalize_whitespace(novel_text),
    height=240,
    help="긴 본문일수록 분석이 풍부해집니다.",
)

# --------------------------------
# 등장인물 사전(룰기반) / HF NER 설정
# --------------------------------
char_map: Dict[str, List[str]] = {}

if mode == "룰기반(권장)":
    st.subheader("등장인물 사전")
    st.caption("형식 예시: 김민수=민수,민수씨 / 박은수=은수,은수양 / 진호=진호,진호형")
    aliases_str = st.text_area(
        "인물=별칭1,별칭2 한 줄에 한 인물",
        value="김민수=민수\n박은수=은수\n진호=진호",
        height=120
    )
    for line in aliases_str.splitlines():
        if "=" in line:
            name, al = line.split("=", 1)
            al_list = [name.strip()] + [a.strip() for a in al.split(",") if a.strip()]
            char_map[name.strip()] = sorted(set(al_list))
else:
    st.subheader("Hugging Face NER 설정")
    st.caption("한국어 NER 모델 ID를 입력하세요 (예: 'kogpt/bert-base-ner-korean' 등). 토큰은 .env 또는 아래에 입력.")
    model_id = st.text_input("모델 ID", value="", placeholder="예: tner/roberta-base-ner-korean (모델마다 상이)")
    hf_token_input = st.text_input("HF_TOKEN (선택, .env가 있으면 비워두세요)", type="password", value="")
    eff_token = hf_token_input.strip() or HF_TOKEN
    topk = st.slider("자동 추출할 상위 인물 수(별칭 후보)", 3, 20, 8)
    manual_plus = st.text_input("추가로 고정할 인물명(쉼표로)", value="")

    if st.button("NER로 인물 추출 실행", disabled=(not novel_text or not model_id)):
        with st.spinner("NER 추출 중..."):
            try:
                base_aliases = extract_person_aliases_by_ner(novel_text[:15000], model_id, eff_token, top_k=topk)
                if manual_plus.strip():
                    base_aliases.extend([a.strip() for a in manual_plus.split(",") if a.strip()])
                unique = []
                for a in base_aliases:
                    if a not in unique:
                        unique.append(a)
                char_map = {a: [a] for a in unique}
                st.success(f"인물 후보 {len(unique)}명: {', '.join(unique[:10])}{' ...' if len(unique) > 10 else ''}")
            except Exception as e:
                st.error(f"NER 실패: {e}")
                char_map = {}

    if char_map:
        st.markdown("**추출/편집용 인물 사전** (한 줄 = 인물=별칭1,별칭2)")
        seed = "\n".join([f"{k}={','.join(v)}" for k, v in char_map.items()])
        edited = st.text_area("인물 사전 편집", value=seed, height=160, key="ner_alias_edit")
        char_map = {}
        for line in edited.splitlines():
            if "=" in line:
                name, al = line.split("=", 1)
                al_list = [name.strip()] + [a.strip() for a in al.split(",") if a.strip()]
                char_map[name.strip()] = sorted(set(al_list))

# --------------------------------
# 분석 실행
# --------------------------------
run = st.button("분석 실행", type="primary", disabled=not (novel_text and char_map))

if run:
    if not char_map:
        st.warning("인물 사전이 비어 있습니다.")
        st.stop()

    sentences = split_sentences_kr(novel_text)
    if len(sentences) == 0:
        st.error("문장 분리 결과가 비었습니다. 입력 텍스트를 확인하세요.")
        st.stop()

    spans = make_windows(sentences, window=window, step=step)
    if not spans:
        st.error("윈도우 설정으로 생성된 구간이 없습니다. 윈도우/간격을 줄여보세요.")
        st.stop()

    # 인물별 정규식 컴파일
    alias_regex_map = {ch: compile_alias_regex(aliases) for ch, aliases in char_map.items()}

    # 윈도우 텍스트 구성 및 카운트
    rows = []
    for idx, (s, e) in enumerate(spans, start=1):
        chunk_text = " ".join(sentences[s:e])
        counts = {ch: count_mentions(chunk_text, alias_regex_map[ch]) for ch in char_map}
        total = sum(counts.values())
        for ch, c in counts.items():
            ratio = (c / total) if total > 0 else 0.0
            rows.append({
                "window_index": idx,
                "start_sent": s,
                "end_sent": e,
                "character": ch,
                "count": c,
                "ratio": ratio,
                "total_in_window": total,
            })

    df = pd.DataFrame(rows)
    st.success(f"문장 수: {len(sentences)} · 윈도우 개수: {len(spans)} · 인물 수: {len(char_map)}")

    # ----------------------
    # 차트
    # ----------------------
    st.subheader("시각화")

    characters = sorted(df["character"].unique().tolist())

    # ✅ fig 객체 생성
    fig = go.Figure()

    if show_mode == "선형(Line)":
        for ch in characters:
            sub = df[df["character"] == ch].sort_values("window_index")
            fig.add_trace(go.Scatter(
                x=sub["window_index"],
                y=sub["ratio"],
                mode="lines+markers",
                name=ch,
                line={"shape": "spline" if smooth else "linear"},
                # ⚠️ f-string 안에 %{y}를 넣지 말 것! → 문자열 결합으로 처리
                hovertemplate=(
                    "윈도우 %{x}<br>" +
                    f"{ch} 비율: " + "%{y:.2%}<br>" +
                    ("%{customdata}회 / 총 %{meta}회" if show_counts else "")
                ),
                customdata=sub["count"] if show_counts else None,
                meta=sub["total_in_window"] if show_counts else None,
            ))
        fig.update_layout(
            xaxis_title="소설 진행 (윈도우 인덱스)",
            yaxis_title="등장 비율",
            yaxis_tickformat=".0%",
            template="plotly_white",
            legend_title="인물",
            margin=dict(l=40, r=20, t=10, b=40),
        )
    else:
        # 누적 면적 그래프
        for ch in characters:
            sub = df[df["character"] == ch].sort_values("window_index")
            fig.add_trace(go.Scatter(
                x=sub["window_index"],
                y=sub["ratio"],
                stackgroup="one",
                mode="lines",
                name=ch,
                line={"shape": "spline" if smooth else "linear"},
                hovertemplate=(
                    "윈도우 %{x}<br>" +
                    f"{ch} 비율: " + "%{y:.2%}<br>" +
                    ("%{customdata}회 / 총 %{meta}회" if show_counts else "")
                ),
                customdata=sub["count"] if show_counts else None,
                meta=sub["total_in_window"] if show_counts else None,
            ))
        fig.update_layout(
            xaxis_title="소설 진행 (윈도우 인덱스)",
            yaxis_title="등장 비율(누적)",
            yaxis_tickformat=".0%",
            template="plotly_white",
            legend_title="인물",
            margin=dict(l=40, r=20, t=10, b=40),
        )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # 데이터 다운로드
    # ----------------------
    st.subheader("데이터 다운로드")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "CSV 다운로드",
        data=csv,
        file_name="character_timeline.csv",
        mime="text/csv",
    )

    with st.expander("윈도우 요약 테이블 보기"):
        table = (
            df.pivot_table(index="window_index", columns="character", values="ratio", aggfunc="first")
            .fillna(0.0)
        )
        st.dataframe(table.style.format("{:.1%}"))

st.markdown("---")
st.markdown("ⓒ 미림마이스터고 1학년 4반 2조 **속눈썹조** · 수업 목적의 교육용 템플릿")
