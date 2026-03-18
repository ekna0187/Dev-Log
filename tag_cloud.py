import json
import os
import re
import stanza
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict, Counter

FONT_PATH = r"C:/Windows/Fonts/malgun.ttf"

# ================================================================
# [모듈 1] 감정 사전 구축
# ================================================================
FUNCTIONAL_BLOCKLIST = {
    '것','수','때','곳','분','줄','데','바','지',
    '가','나','이','그','저','있','없','하','되',
    '않','못','안','더','잘','좀','이다','아니','말','쉬','뛰','쓰','써','봐',
    # 단독 한글자 어간 — 조사 오탐 위험 (화가→화, 빛이→빛)
    '화','빛','꽃','물','불','흙','산','강',
    # 한글자+하다 오탐 방지
    '화하다','나하다','이하다','그하다',
}

# 감정 수식어 — 사전에 등재되어 있어도 감정어가 아닌 명사
EMO_MODIFIERS = {
    '기분','마음','감정','느낌','생각','기색','표정','상태','분위기','심정','심리','감각',
}

def build_senti_dict(json_path):
    with open(json_path, encoding='utf-8') as f:
        raw = json.load(f)
    root_pols = defaultdict(list)
    for item in raw:
        pol = int(item['polarity'])
        if pol == 0:
            continue
        for r in item['word_root'].split():
            r = r.strip()
            if not r:
                continue
            if len(r) == 1:
                if r not in FUNCTIONAL_BLOCKLIST:
                    root_pols[r + '다'].append(pol)
                continue
            if r in FUNCTIONAL_BLOCKLIST:
                continue
            root_pols[r].append(pol)
            root_pols[r + '다'].append(pol)
            if not r.endswith('하') and not r.endswith('다'):
                root_pols[r + '하다'].append(pol)

    senti_dict = {}
    for root, pols in root_pols.items():
        sign_vote = Counter(1 if p > 0 else -1 for p in pols)
        dominant  = sign_vote.most_common(1)[0][0]
        same_sign = [p for p in pols if (p > 0) == (dominant > 0)]
        senti_dict[root] = max(same_sign, key=abs)
    return senti_dict


# ================================================================
# [모듈 2] 불용어 (기본 + name.json 자동 로딩)
# ================================================================
_BASE_STOPWORDS = {
    '못','안','더','잘','좀','매우','정말','너무','아주','꽤',
    '그냥','막','또','이미','항상','늘','자꾸','많이','조금',
    '별로','전혀','다시','계속','같이','함께','혼자','먼저',
    '하다','있다','되다','없다','같다','보다','오다','가다',
    '이다','아니다','말다','들다','나다','주다','받다','알다',
    '나오다','들어가다','나가다','올라가다','내려가다',
    '것','수','때','곳','분','줄','만큼','대로','뿐',
    '다','나','너','저','이','그',
    # 단독으로 오면 감정어 아닌 것들
    '화','나오','이오','가오',
}

def build_stopwords(name_json_path):
    stopwords = set(_BASE_STOPWORDS)
    if name_json_path and os.path.exists(name_json_path):
        with open(name_json_path, encoding='utf-8') as f:
            name_data = json.load(f)
        for role, variants in name_data.items():
            stopwords.add(role)
            stopwords.update(variants)
    return stopwords


# ================================================================
# [모듈 3] Stanza lemma → 어간 추출
#
# Stanza kaist 모델의 lemma는 두 가지 패턴으로 나옴:
#
# 패턴 A — '+' 형태소 분리:
#   '좋았어요'  → lemma: '좋+았+어요'  → split('+')[0] → '좋'
#   '힘들어요'  → lemma: '힘들+어+요'  → split('+')[0] → '힘들'
#   '슬퍼요'   → lemma: '슬프+어+요'  → split('+')[0] → '슬프'
#
# 패턴 B — 어미 미제거:
#   '짜증났어요' → lemma: '짜증나어요'  → 어미 제거 → '짜증나'
#   '걱정돼요'  → lemma: '걱정되어요'  → 어미 제거 → '걱정되'
#   '불안했어요' → lemma: '불안하었어요' → 어미 제거 → '불안'
# ================================================================

NEG_PREFIXES = {'안', '못', '안하', '못하'}

def extract_stem(lemma):
    """
    Stanza lemma → 순수 어간.

    패턴 A — '+' 형태소 분리:
      '좋+아+요'      → '좋'
      '안+좋+아+요'   → 첫 조각이 부정소('안') → 두 번째 조각 '좋' 반환
      '짜증나+아+요'  → '짜증나'

    패턴 B — 어미 미제거:
      '불안하었어요'  → 어미 제거 → '불안하'
      '짜증나어요'   → 어미 제거 → '짜증나'
    """
    if not lemma:
        return ''
    if '+' in lemma:
        parts = lemma.split('+')
        # 첫 조각이 부정소면 두 번째 조각을 어간으로
        stem = parts[1] if parts[0] in NEG_PREFIXES and len(parts) > 1 else parts[0]
        return stem
    for sfx in ('하었어요','하았어요','었어요','았어요','였어요',
                '어요','아요','여요','하였다','었다','았다',
                '었','았','여','하여','하다'):
        if lemma.endswith(sfx) and len(lemma) > len(sfx):
            return lemma[:-len(sfx)]
    return lemma


def lookup_emotion(stem, senti_dict, stopwords):
    """
    어간 → 감정사전 조회.
    시도 순서:
    1) stem 그대로
    2) stem + '다'
    3) stem + '하다'
    4) stem + '나다'
    5) 끝 '하' 제거 → 어근 + '하다'  (불안하 → 불안하다)
    6) 끝 '되' 제거 → 어근 + '하다'  (걱정되 → 걱정하다)
    7) 끝 '스럽/롭' 제거 → 어근 + '하다'  (당황스럽 → 당황하다)
    """
    if not stem or len(stem) < 1:
        return None, 0

    candidates = [stem, stem+'다', stem+'하다', stem+'나다']

    # '하' 제거: 불안하 → 불안 → 불안하다
    if stem.endswith('하') and len(stem) > 1:
        base = stem[:-1]
        candidates += [base, base+'하다', base+'다']

    # '되' 제거: 걱정되 → 걱정 → 걱정하다
    if stem.endswith('되') and len(stem) > 1:
        base = stem[:-1]
        candidates += [base, base+'하다', base+'다']

    # '스럽' 제거: 당황스럽 → 당황 → 당황하다
    # 사전에 '당황스럽다' 없고 '당황하다'만 있는 경우 커버
    if stem.endswith('스럽') and len(stem) > 2:
        base = stem[:-2]
        candidates += [base, base+'하다', base+'스럽다']

    # '롭' 제거: 외롭 → 외 (너무 짧음), 자유롭 → 자유 → 자유롭다
    if stem.endswith('롭') and len(stem) > 2:
        base = stem[:-1]  # '롭' 제거
        candidates += [base+'다', base[:-1]+'롭다']  # 외롭다, 자유롭다

    for cand in candidates:
        if not cand or len(cand) < 2:
            continue
        if cand in stopwords:
            continue
        base_check = cand.rstrip('다')
        if base_check in EMO_MODIFIERS:
            continue
        if base_check.endswith('하') and base_check[:-1] in EMO_MODIFIERS:
            continue
        pol = senti_dict.get(cand, 0)
        if pol != 0:
            return cand, pol

    return None, 0


# ================================================================
# [모듈 4] 메인 분석 파이프라인
# ================================================================

def analyze_data(json_path, text_path, name_json_path=None):
    senti_dict = build_senti_dict(json_path)
    stopwords  = build_stopwords(name_json_path)

    with open(text_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    # 화자 태그 처리: "내담자 : 발화" → 발화만 추출
    has_tag = any(':' in line for line in raw_lines)
    if has_tag:
        lines = [line.split(':', 1)[1].strip() for line in raw_lines if ':' in line]
    else:
        lines = [line.strip() for line in raw_lines if line.strip()]

    # Stanza: tokenize + pos + lemma 만 사용 (depparse 불필요)
    # stanza.download('ko')  # 처음 한 번만
    nlp = stanza.Pipeline(
        'ko',
        processors='tokenize,pos,lemma',
        use_gpu=False,
        tokenize_pretokenized=False,
    )

    pos_freq, neg_freq = {}, {}

    print(f"{'표면형':<14} {'lemma':<20} {'어간':<10} {'등재어':<14} {'극성'}")
    print('-' * 65)

    for line in lines:
        if not line:
            continue
        # 마침표 자동 추가 → Stanza 문장 분리 보조
        if line[-1] not in '.?!':
            line = line + '.'
        doc = nlp(line)

        for sent in doc.sentences:
            prev_stem = ''  # 직전 토큰 어간 (bigram 결합용)

            for word in sent.words:
                if word.upos not in ('ADJ', 'VERB', 'NOUN'):
                    prev_stem = ''
                    continue

                surface = re.sub(r'[^가-힣]', '', word.text).strip()
                lemma   = word.lemma or ''

                if len(surface) < 2:
                    prev_stem = ''
                    continue
                if surface in stopwords:
                    prev_stem = ''
                    continue

                stem = extract_stem(lemma)

                # NOUN + 한글자 stem = 명사+조사 패턴 (화가, 빛이 등)
                # 단독 감정 조회는 스킵하고 bigram용 prev_stem만 저장
                skip_solo = (word.upos == 'NOUN' and len(stem) == 1)

                # bigram 결합 시도: 직전 어간 + 현재 어간
                # 예: '화'(prev) + '나오'(curr) → '화나오' MISS
                #     '화'(prev) + '나' (나오의 첫음절) → '화나' → 화나다 ✓
                entry, polarity = None, 0
                if prev_stem:
                    # 시도1: prev + stem 전체
                    entry, polarity = lookup_emotion(prev_stem + stem, senti_dict, stopwords)
                    # 시도2: prev + stem 첫 음절 (나오→나, 들어→들)
                    if polarity == 0 and len(stem) > 1:
                        first_syl = stem[0]
                        entry, polarity = lookup_emotion(prev_stem + first_syl, senti_dict, stopwords)

                # 단독 어간 조회 (NOUN 한글자는 스킵)
                if polarity == 0 and not skip_solo:
                    entry, polarity = lookup_emotion(stem, senti_dict, stopwords)

                # surface 직접 조회 (fallback, NOUN 한글자는 스킵)
                if polarity == 0 and not skip_solo:
                    entry, polarity = lookup_emotion(surface, senti_dict, stopwords)

                if polarity != 0:
                    sign = '+' if polarity > 0 else '-'
                    print(f"{surface:<14} {lemma:<20} {stem:<10} {entry:<14} {sign}{abs(polarity)}")
                    if polarity > 0:
                        pos_freq[entry] = pos_freq.get(entry, 0) + 1
                    else:
                        neg_freq[entry] = neg_freq.get(entry, 0) + 1

                # 현재 어간을 다음 토큰의 prev_stem으로 저장
                prev_stem = stem if stem else ''  

    return pos_freq, neg_freq


# ================================================================
# [모듈 5] 워드클라우드 렌더링
# ================================================================

def render_result(pos, neg):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    configs = [(neg, 'Negative', 'Reds'), (pos, 'Positive', 'Blues')]

    for i, (data, title, cmap) in enumerate(configs):
        if data:
            wc = WordCloud(
                font_path=FONT_PATH,
                background_color='white',
                colormap=cmap,
                width=800, height=800,
                prefer_horizontal=1.0,
                min_word_length=2,
                collocations=False,
            ).generate_from_frequencies(data)
            axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(title, fontsize=24, fontweight='bold', pad=20)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# ================================================================
# 실행
# ================================================================

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    p, n = analyze_data(
        os.path.join(base_dir, 'SentiWord_info.json'),
        os.path.join(base_dir, 'ggtest.text'),
        os.path.join(base_dir, 'name.json'),
    )
    render_result(p, n)
