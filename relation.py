import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import stanza
import json, re
from pathlib import Path
from collections import defaultdict, Counter

# ================================================================
# 0) 경로 설정
# ================================================================
BASE_DIR   = Path(__file__).parent
TEXT_PATH  = BASE_DIR / "ggtest.text"
SENTI_PATH = BASE_DIR / "SentiWord_info.json"
NAME_PATH  = BASE_DIR / "name.json"

plt.rcParams["font.family"] = "Malgun Gothic"

NODE_STYLE = {
    "내담자": {"color": "#BDD9FF", "shape": "o", "size": 2500},
    "인물":   {"color": "#C8F2C2", "shape": "D", "size": 2000},
    "원인":   {"color": "#E8D9FF", "shape": "8", "size": 1800},
    "감정":   {"color": "#FFD6D6", "shape": "s", "size": 1800},
}

# ================================================================
# 1) 유니코드 형태소 엔진 (원형 복원용)
# ================================================================
ONSET   = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
NUCLEUS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
CODA    = list(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

def decompose(c):
    code = ord(c) - 0xAC00
    if code < 0 or code > 11171: return None
    return code // 588, (code // 28) % 21, code % 28

def compose(o, n, c): return chr(0xAC00 + o*588 + n*28 + c)

def set_coda(c, j):
    d = decompose(c)
    if not d: return c
    return compose(d[0], d[1], CODA.index(j) if j else 0)

def remove_coda(c): return set_coda(c, '')

L_IRREG = {'다르','모르','부르','고르','오르','이르','마르','누르','흐르','자르','두르','구르','나르','조르','치르'}
D_IRREG = {'듣','묻','걷','깨닫','싣','붓'}
S_IRREG = {'낫','잇','짓','긋','붓','젓','엿'}

def restore_irregular(stem):
    if not stem: return stem
    last = stem[-1]; d = decompose(last)
    if not d: return stem
    o_i, n_i, c_i = d
    nucleus = NUCLEUS[n_i]; coda = CODA[c_i].strip()
    # ㅂ불규칙
    if nucleus in ('ㅗ','ㅜ','ㅝ') and coda == '' and len(stem) >= 2:
        pd = decompose(stem[-2])
        if pd and CODA[pd[2]].strip() == '':
            return stem[:-2] + set_coda(stem[-2], 'ㅂ')
    # ㄷ불규칙
    if coda == 'ㄹ':
        cand = stem[:-1] + set_coda(last, 'ㄷ')
        if cand in D_IRREG: return cand
    # ㅅ불규칙
    if coda == '':
        cand = stem[:-1] + set_coda(last, 'ㅅ')
        if cand in S_IRREG: return cand
    # 르불규칙
    if coda == 'ㄹ' and len(stem) >= 2:
        reu = compose(ONSET.index('ㄹ'), NUCLEUS.index('ㅡ'), 0)
        cand = stem[:-2] + remove_coda(last) + reu
        if any(cand.endswith(s) for s in L_IRREG): return cand
    # 으불규칙
    if nucleus in ('ㅏ','ㅓ') and coda == '' and len(stem) >= 2:
        eu = compose(o_i, NUCLEUS.index('ㅡ'), 0)
        return stem[:-1] + eu
    return stem

JOSA_LIST_SORTED = sorted([
    ('이랑','coda'),('랑','no_coda'),('에서는','any'),('에서','any'),
    ('에게서','any'),('에게','any'),('으로','coda'),('로','no_coda'),
    ('이에요','coda'),('예요','no_coda'),('에는','any'),('에도','any'),('에','any'),
    ('과','coda'),('와','no_coda'),('은','coda'),('는','no_coda'),
    ('이','coda'),('가','no_coda'),('을','coda'),('를','no_coda'),
    ('의','any'),('도','any'),('만','any'),('까지','any'),('부터','any'),
    ('처럼','any'),('보다','any'),('마다','any'),('랑','any'),('하고','any'),
], key=lambda x: -len(x[0]))

def strip_josa(word):
    for josa, cond in JOSA_LIST_SORTED:
        if not word.endswith(josa): continue
        stem = word[:-len(josa)]
        if len(stem) < 1: continue
        if cond == 'any': return stem
        from_ = decompose(stem[-1])
        lc = CODA[from_[2]].strip() if from_ else ''
        if cond == 'coda' and lc: return stem
        if cond == 'no_coda' and not lc: return stem
    return word

ENDINGS_SORTED = sorted([
    ('합니다','하다'),('했습니다','하다'),('해요','하다'),('했어요','하다'),
    ('해서요','하다'),('해서','하다'),('하거든요','하다'),('하더니','하다'),
    ('하잖아요','하다'),('하지만','하다'),
    ('스러워요','스럽다'),('스러워서','스럽다'),('스러워','스럽다'),('스러운','스럽다'),
    ('로워요','롭다'),('로워서','롭다'),('로워','롭다'),('로운','롭다'),
    ('았었어요','다'),('었었어요','다'),
    ('아요','다'),('어요','다'),('았어요','다'),('었어요','다'),
    ('아서요','다'),('어서요','다'),('아서','다'),('어서','다'),
    ('아도','다'),('어도','다'),('았다','다'),('었다','다'),('였다','다'),
    ('겠어요','다'),('겠다','다'),('지만','다'),('는데','다'),('니까','다'),
    ('더라도','다'),('더라고','다'),('더니','다'),
    ('고는','다'),('고도','다'),('고서','다'),('고요','다'),('고','다'),
    ('네요','다'),('군요','다'),('구나','다'),('잖아요','다'),('잖아','다'),
    ('거든요','다'),('거든','다'),('냐','다'),('니','다'),
    ('는','다'),('은','다'),('던','다'),
    ('서는','다'),('서도','다'),('서요','다'),('서','다'),('요','다'),('도','다'),
], key=lambda x: -len(x[0]))

def normalize_to_base(word):
    """어절 → 사전 원형 복원"""
    word = strip_josa(word)
    for ending, suffix in ENDINGS_SORTED:
        if word.endswith(ending):
            stem = word[:-len(ending)]
            if len(stem) < 1: continue
            if suffix == 'da' and ending in ('아','어','아서','어서','아요','어요','서','요'):
                ld = decompose(stem[-1])
                if ld and NUCLEUS[ld[1]] in ('ㅐ','ㅔ') and ld[2] == 0:
                    stem = stem[:-1] + set_coda(stem[-1], 'ㅎ')
            stem = restore_irregular(stem)
            return stem + suffix
    return word


# ================================================================
# 2) 감정 사전 로드
# ================================================================
def load_senti(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"감정사전 없음: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    FUNC = {'것','수','때','곳','분','줄','데','바','지','가','나','이','그','저',
            '있','없','하','되','않','못','안','더','잘','좀','이다','아니','말'}

    root_pols = defaultdict(list)
    for item in data:
        pol = int(item.get('polarity', 0))
        if pol == 0: continue
        for r in item.get('word_root', '').split():
            r = r.strip()
            if not r: continue
            if len(r) == 1:
                if r not in FUNC: root_pols[r+'다'].append(pol)
                continue
            if r in FUNC: continue
            root_pols[r].append(pol)
            root_pols[r+'다'].append(pol)
            if not r.endswith('하') and not r.endswith('다'):
                root_pols[r+'하다'].append(pol)

    senti = {}
    for root, pols in root_pols.items():
        sv = Counter(1 if p > 0 else -1 for p in pols)
        dom = sv.most_common(1)[0][0]
        same = [p for p in pols if (p > 0) == (dom > 0)]
        senti[root] = max(same, key=abs)
    return senti


def load_name_map(path):
    p = Path(path)
    if not p.exists(): return {}, {}, set()
    with open(path, encoding="utf-8") as f:
        name_raw = json.load(f)
    name_map = {alias: main for main, aliases in name_raw.items() for alias in aliases}
    name_map.update({main: main for main in name_raw})
    return name_raw, name_map, set(name_raw.keys())


senti_dict  = load_senti(SENTI_PATH)
name_raw, name_map, PERSON_COMMON = load_name_map(NAME_PATH)

if not PERSON_COMMON:
    PERSON_COMMON = {"엄마","아빠","부모","친구","동생","언니","오빠","연인",
                     "선생님","상사","상담사","가족","동료","교수님","선배","후배"}

_re_kor = re.compile(r'[^가-힣]')


# ================================================================
# 3) Stanza 파이프라인
# ================================================================
# stanza.download("ko")  # 처음 한 번만 실행
nlp = stanza.Pipeline("ko", processors="tokenize,pos,lemma,depparse",
                       use_gpu=False, tokenize_pretokenized=False)


# ================================================================
# 4) 핵심 분석: Stanza depparse로 인물/원인/감정 추출
#
# 목표 구조: 인물 ─[원인]─▶ 감정
#
# Stanza depparse 활용 전략:
# ┌─────────────────────────────────────────────────────────┐
# │ "엄마가 잔소리해서 짜증났어요"                           │
# │                                                         │
# │  엄마가    → nsubj → head: 잔소리해서(advcl)           │
# │  잔소리해서 → advcl → head: 짜증났어요(root)  ← 원인절 │
# │  짜증났어요 → root                            ← 감정    │
# │                                                         │
# │ "친구가 울어서 걱정돼요"                                 │
# │  친구가    → nsubj → head: 울어서(advcl)               │
# │  울어서    → advcl → head: 걱정돼요(root)    ← 원인절  │
# │  걱정돼요  → root                            ← 감정    │
# └─────────────────────────────────────────────────────────┘
#
# 추출 로직:
# 1) root 토큰에서 감정 탐색
# 2) root의 advcl/ccomp 자식 → 원인절
# 3) 원인절의 nsubj → 인물
# 4) 원인절 텍스트 → 원인 레이블 (인물 제거 후)
# ================================================================

def lookup_emotion(surface, lemma):
    """표면형·lemma·원형복원 순으로 감정사전 조회"""
    import re as _re
    candidates = []

    def add_variants(raw):
        c = _re_kor.sub('', raw).strip()
        if len(c) < 2: return
        if c not in candidates: candidates.append(c)

        # 원형 복원
        rest = normalize_to_base(c)
        if rest and rest not in candidates: candidates.append(rest)

        base = rest.rstrip('다') if rest else c
        if len(base) >= 2:
            if base not in candidates: candidates.append(base)
            if base+'하다' not in candidates: candidates.append(base+'하다')
            if base+'나다' not in candidates: candidates.append(base+'나다')

        # 됐어요/돼요 → 되다 → 어근 추출: 걱정됐 → 걱정
        for suffix in ('됐어요','됐다','돼요','돼서','되어','되다'):
            if c.endswith(suffix):
                noun = c[:-len(suffix)]
                if len(noun) >= 2 and noun not in candidates:
                    candidates.append(noun)
                    candidates.append(noun+'하다')

        # 났어요/나요/나서 → 어근: 짜증났 → 짜증
        for suffix in ('났어요','났다','나요','나서','났'):
            if c.endswith(suffix):
                noun = c[:-len(suffix)]
                if len(noun) >= 2 and noun not in candidates:
                    candidates.append(noun)
                    candidates.append(noun+'나다')

    for raw in [surface, lemma]:
        if raw: add_variants(raw)

    best_emo, best_pol = None, 0.0
    for cand in candidates:
        # 인물/역할명은 감정으로 오인식 방지
        if cand in PERSON_COMMON or cand in name_map:
            continue
        pol = senti_dict.get(cand, 0)
        if pol != 0 and abs(pol) > abs(best_pol):
            best_emo, best_pol = cand, pol
    return best_emo, best_pol


def clean_person(word):
    """조사 제거 후 인물명 추출"""
    w = _re_kor.sub('', str(word)).strip()
    if len(w) < 2: return ''
    return strip_josa(w)


def get_subtree_text(words, root_id, exclude_ids=None):
    """
    Stanza 의존트리에서 root_id 기준 서브트리 텍스트 추출.
    exclude_ids: 제외할 토큰 id 집합 (인물 토큰 등)
    """
    if exclude_ids is None: exclude_ids = set()

    children_map = defaultdict(list)
    for w in words:
        children_map[w.head].append(w.id)

    ids = set()
    stack = [root_id]
    while stack:
        cur = stack.pop()
        if cur in exclude_ids: continue
        ids.add(cur)
        for ch in children_map[cur]:
            if ch not in ids:
                stack.append(ch)

    return " ".join(w.text for w in words if w.id in sorted(ids))


def clean_cause_label(text, person=None):
    """
    원인 텍스트 정제:
    - 인물명 제거
    - 연결어미 제거 (해서, 아서, 어서, 서 등)
    - 부사 제거
    """
    if not text: return None

    # 인물 제거
    if person:
        text = re.sub(re.escape(person), '', text).strip()
        text = re.sub(r'^(이|가|은|는|이랑|랑|과|와)\s*', '', text).strip()

    # 앞 조사 찌꺼기 제거
    text = re.sub(r'^(이|가|은|는|을|를|의|에|도|만)\s+', '', text).strip()

    # 연결어미 제거
    text = re.sub(r'(해줘서|해서|줘서|워서|와서|아서|어서|서|고|며|면서|니까|니|요)$', '', text).strip()

    # 너무짧거나 비어있으면 None
    if not text or len(text) < 2: return None
    return text


def analyze_sentence(sent):
    """
    Stanza sentence 하나를 분석해서
    (person, cause_label, emotion, polarity) 반환.

    핵심 전략:
    1. root 토큰에서 감정 탐색
    2. root의 advcl/ccomp/obl 자식 → 원인절 후보
    3. 원인절 내 nsubj → 인물
    4. 원인절 텍스트 정제 → 원인 레이블
    5. 인물 못 찾으면 root의 nsubj에서 탐색
    """
    words = sent.words

    # id → word 맵
    id_map = {w.id: w for w in words}

    # 1) root 찾기 + 감정 탐색
    root_word = None
    emotion, polarity = None, 0.0
    for w in words:
        if w.deprel == 'root':
            root_word = w
            emotion, polarity = lookup_emotion(w.text, w.lemma)
            break

    # root에서 감정 못 찾으면 전체 토큰 순회
    if not emotion:
        for w in words:
            if w.upos not in ('ADJ','VERB','NOUN'): continue
            emo, pol = lookup_emotion(w.text, w.lemma)
            if emo and abs(pol) > abs(polarity):
                emotion, polarity = emo, pol

    if not emotion:
        return None, None, None, 0.0

    # 2) 원인절 찾기: root의 자식 중 advcl/ccomp
    cause_node = None
    if root_word:
        for w in words:
            if w.head == root_word.id and w.deprel in ('advcl','ccomp','obl'):
                cause_node = w
                break

    # 3) 인물 찾기
    person = None

    # 원인절 내 nsubj
    if cause_node:
        for w in words:
            if w.head == cause_node.id and w.deprel in ('nsubj','nsubj:pass'):
                cand = clean_person(w.text)
                if cand and (cand in name_map or cand in PERSON_COMMON):
                    person = name_map.get(cand, cand)
                    break
                elif cand and len(cand) >= 2:
                    person = cand
                    break

    # root의 nsubj에서도 탐색
    if not person and root_word:
        for w in words:
            if w.head == root_word.id and w.deprel in ('nsubj','nsubj:pass'):
                cand = clean_person(w.text)
                if cand and (cand in name_map or cand in PERSON_COMMON):
                    person = name_map.get(cand, cand)
                    break
                elif cand and len(cand) >= 2:
                    person = cand
                    break

    # name_map 전체 순회 (고유명사 보완)
    if not person:
        for w in words:
            cand = clean_person(w.text)
            if cand in name_map:
                person = name_map[cand]
                break
        if not person:
            for w in words:
                cand = clean_person(w.text)
                if cand in PERSON_COMMON:
                    person = cand
                    break

    # 4) 원인 레이블 추출
    cause_label = None
    if cause_node:
        # 인물 토큰 id 수집 (서브트리에서 제외)
        exclude = set()
        if person:
            for w in words:
                if clean_person(w.text) == person or w.text.startswith(person):
                    exclude.add(w.id)
                    # 인물 토큰의 자식도 제외
                    for w2 in words:
                        if w2.head == w.id:
                            exclude.add(w2.id)

        raw_cause = get_subtree_text(words, cause_node.id, exclude_ids=exclude)
        cause_label = clean_cause_label(raw_cause, person=person)

    return person, cause_label, emotion, polarity


# ================================================================
# 5) 텍스트 전처리 — 화자 태그 제거
# ================================================================
SPEAKER_RE = re.compile(r'^.{1,8}\s*[:：]\s*')

def preprocess_text(raw):
    lines = raw.strip().splitlines()
    has_tag = any(SPEAKER_RE.match(l.strip()) for l in lines if l.strip())
    result = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if has_tag:
            if line.startswith("내담자"):
                content = SPEAKER_RE.sub('', line).strip()
                if content: result.append(content)
        else:
            result.append(line)
    return "\n".join(result)


# ================================================================
# 6) 그래프 생성
# ================================================================
def create_graph(text_path, verbose=True):
    p = Path(text_path)
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {text_path}")

    raw = p.read_text(encoding="utf-8").strip()
    if not raw: raise ValueError("텍스트 비어있음")

    processed = preprocess_text(raw)
    if not processed: raise ValueError("분석할 발화 없음")

    # 줄바꿈 기준으로 한 줄씩 처리
    # 마침표가 없으면 자동으로 붙여서 Stanza가 문장 경계를 확실히 인식하게 함
    lines = [l.strip() for l in processed.splitlines() if l.strip()]

    G = nx.DiGraph()
    G.add_node("내담자", type="내담자")

    all_sents = []
    for line in lines:
        # 마침표/물음표/느낌표 없으면 마침표 추가
        if not line[-1] in '.?!':
            line = line + '.'
        doc = nlp(line)
        for sent in doc.sentences:
            all_sents.append(sent)

    for sent in all_sents:
        person, cause, emotion, pol = analyze_sentence(sent)

        if verbose:
            print(f"[문장] {sent.text.strip()}")
            print(f"  인물: {person}  |  원인: {cause}  |  감정: {emotion} ({pol:+.0f})")

        if not (person and emotion):
            if verbose: print("  → 인물 또는 감정 미탐지, 스킵")
            continue

        edge_color = "#6BAED6" if pol > 0 else "#FB6A4A"

        G.add_node(person,  type="인물")
        G.add_node(emotion, type="감정")
        G.add_edge("내담자", person,  color="#AAAAAA", style="solid")
        G.add_edge(person,  emotion, color=edge_color,  style="solid")

        if cause:
            G.add_node(cause, type="원인")
            G.add_edge(person,  cause,   color="#9E9AC8", style="dashed")
            G.add_edge(cause,   emotion, color=edge_color,  style="solid")

    return G


# ================================================================
# 7) 시각화
# ================================================================
def draw_graph(G):
    if G.number_of_nodes() <= 1:
        print("그래프 데이터가 부족합니다."); return

    pos = nx.spring_layout(G, k=3.0, seed=42, iterations=100)

    fig, ax = plt.subplots(figsize=(16, 11))

    for node_type, style in NODE_STYLE.items():
        nodelist = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
        if not nodelist: continue
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist,
            node_color=style["color"], node_shape=style["shape"],
            node_size=style["size"], alpha=0.92,
            edgecolors="#888888", linewidths=1.2, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=11, font_family="Malgun Gothic",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.75, ec="none"), ax=ax)

    # 실선 / 점선 구분
    solid_edges  = [(u,v) for u,v,d in G.edges(data=True) if d.get("style","solid") == "solid"]
    dashed_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("style") == "dashed"]

    solid_colors  = [G[u][v]["color"] for u,v in solid_edges]
    dashed_colors = [G[u][v]["color"] for u,v in dashed_edges]

    nx.draw_networkx_edges(G, pos, edgelist=solid_edges,
        width=2.0, edge_color=solid_colors, alpha=0.85,
        arrows=True, arrowsize=18,
        connectionstyle="arc3,rad=0.08", ax=ax)

    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges,
        width=1.5, edge_color=dashed_colors, alpha=0.75,
        style="dashed", arrows=True, arrowsize=14,
        connectionstyle="arc3,rad=0.08", ax=ax)

    legend_handles = [mpatches.Patch(color=s["color"], label=t) for t, s in NODE_STYLE.items()]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.8)

    ax.set_title("내담자 관계 감정 맵", fontsize=16, fontweight="bold", pad=16)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ================================================================
# 실행
# ================================================================
if __name__ == "__main__":
    G = create_graph(TEXT_PATH, verbose=True)
    print(f"\n[요약] 노드 {G.number_of_nodes()}개 / 엣지 {G.number_of_edges()}개")
    draw_graph(G)
