import bisect
import copy
import os
import re
import unicodedata
from typing import List, Any
import pyopenjtalk as pjt
from text.symbols import punctuation


def LIS_solver(seq: List[int]) -> int:
    n = len(seq)
    seq = [0] + seq
    dp = [0] * n
    i = 0
    for x in range(1, n + 1):
        if seq[x] > dp[i]:
            i += 1
            dp[i] = seq[x]
        else:
            ind = bisect.bisect(dp, x) - 1
            dp[ind] = min(dp[ind], seq[x])
    return i


def Permutation_LCS_solver(seq_a: List[int], seq_b: List[int]) -> int:
    # O(nlogn)
    rg = set(seq_a)
    n = len(seq_a)
    seq_a = seq_a
    usages = {}
    cor = {}
    for i, x in enumerate(rg):
        usages[x] = 0
        cor[x] = i + 1
    pos = []
    for x in range(len(seq_a)):
        pos.append([])
    for x in range(n):
        pos[cor[seq_a[x]]].append(x)
    tm = []
    for x in range(len(seq_b)):
        try:
            t_pos = pos[cor[seq_b[x]]][usages[seq_b[x]]]
            usages[seq_b[x]] += 1
        except (IndexError, KeyError):
            tm.append(-1)
        else:
            tm.append(t_pos)
    return LIS_solver(tm)


def LCS_solver(seq_a: List[Any], seq_b: List[Any]) -> (dict, int):
    n = len(seq_a)
    m = len(seq_b)
    seq_a = [0] + seq_a
    seq_b = [0] + seq_b
    dp = []
    trans = {}
    for _ in range(n + 1):
        dp.append([0] * (m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i] == seq_b[j]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                trans[(i, j)] = (i - 1, j - 1)
            else:
                # dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                if dp[i - 1][j] > dp[i][j - 1]:
                    dp[i][j] = dp[i - 1][j]
                    trans[(i, j)] = (i - 1, j)
                else:
                    dp[i][j] = dp[i][j - 1]
                    trans[(i, j)] = (i, j - 1)
    cur = (n, m)
    res = {}
    while cur[0] and cur[1]:
        nxt = trans[cur]
        if nxt == (cur[0] - 1, cur[1] - 1):
            res[cur] = nxt
        cur = nxt
    return res, dp[n][m]


def distribute_phonemes(pn, cn):
    resu = [pn // cn] * cn
    rem = pn - cn * (pn // cn)
    for x in range(rem):
        resu[cn - x - 1] += 1
    return resu


def numeric_feature_by_regex(regex, s):
    # https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html#10.2-Tacotron-2-%E3%82%92%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%81%AB%E9%81%A9%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AE%E5%A4%89%E6%9B%B4
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def g2p_with_accent_info(norm_text):
    # https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html#10.2-Tacotron-2-%E3%82%92%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%81%AB%E9%81%A9%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AE%E5%A4%89%E6%9B%B4
    # 以重音短语分组作为分割 ' '
    # a1为重音音节距离(降) ']'
    # 重音短语起始第二个音节组为升 '/'
    # e3:前一个音节组词性(1=疑问) '?'
    # 这个函数需要使用word2ph信息
    fl = pjt.extract_fullcontext(norm_text)
    n = len(fl)
    result = []
    word2ph = calculate_word2ph(norm_text)
    rcm = {}
    off = 0
    for cnt, x in enumerate(word2ph):
        for sui in range(x):
            rcm[off+sui] = cnt
        off += x
    new_word2ph = copy.deepcopy(word2ph)
    assert n == sum(new_word2ph) + 2
    gof = 0
    for current in range(n):
        lab_curr = fl[current]
        p3 = re.search(r"-(.*?)\+", lab_curr).group(1)
        if p3 == 'sil':
            if not current:
                gof += 1
            if current == n - 1:
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                gof += 1
                if e3:
                    result.append('?')
                    new_word2ph[rcm[current - gof]] += 1
                else:
                    result.append('*')
                    new_word2ph[rcm[current - gof]] += 1
            continue
        elif p3 == 'pau':
            e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
            if e3:
                result.append('?')
            else:
                result.append(',')
        else:
            result.append(p3)
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", fl[current + 1])

        if a3 == 1 and a2_next == 1:
            result.append("#")
            new_word2ph[rcm[current + 1 - gof]] += 1
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            result.append("]")
            new_word2ph[rcm[current + 1 - gof]] += 1
        elif a2 == 1 and a2_next == 2:
            result.append("/")
            new_word2ph[rcm[current + 1 - gof]] += 1
    assert len(result) == sum(new_word2ph)
    return result, new_word2ph


def get_item(key, dic, default):
    try:
        return dic[key]
    except KeyError:
        return default


def calculate_word2ph(norm_text: str) -> list:
    sc = [x.lower() for x in pjt.g2p(norm_text).split(' ')]
    ss = []
    cm = {}
    rcm = {}
    punc_num = 0
    for cnt, tm in enumerate(norm_text):
        if tm in punctuation:
            punc_num += 1
            cm[cnt] = (len(ss) + 1, len(ss) + 1)
            ss.append('pau')
        else:
            si = len(ss)
            subs = [tc.lower() for tc in pjt.g2p(tm).split(' ')]
            ss += subs
            cm[cnt] = (si + 1, si + len(subs))
            for ind in range(si, si + len(subs)):
                rcm[ind] = cnt + 1
    res, _ = LCS_solver(ss, sc)
    tres = {}
    for x in res.keys():
        tres[x[0]] = x[1]
    res = tres
    total = len(sc)
    word2ph = []
    match_failures = ()  # (start_char_id, stop_char_id)
    is_last_perfect_match = True
    last_ep = 1
    for x in range(len(norm_text)):
        start = cm[x][0]
        end = cm[x][1]  # phoneme range in per-processed sequence (index starts from 1)
        start_p = True
        stop_p = True
        while not get_item(start, res, False) and start <= end:
            start += 1
            start_p = False
        while not get_item(end, res, False) and start <= end:
            end -= 1
            stop_p = False
        if start > end:
            # All phonemes of the token cannot be matched.
            # This char will get the phoneme num depends on next successful match
            if match_failures:
                match_failures = (match_failures[0], match_failures[1] + 1)
            else:
                match_failures = (x, x)
            continue
        extra = 0
        if start_p and stop_p:
            # perfect match
            total -= res[end] - res[start] + 1
            if match_failures:
                # previous matches failed.Need to distribute available phonemes to them.
                available = res[start] - last_ep
                assert available >= 0, ("If you get this error and the input of this function is ALREADY "
                                        "normalized, this may be a logical bug, please "
                                        "consider to report it on github")
                if not is_last_perfect_match:
                    # if the last successful match is not perfect, give it up to 1
                    if available > match_failures[1] - match_failures[0] + 1:
                        word2ph[-1] += 1
                        available -= 1
                        total -= 1
                word2ph += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                total -= available
                match_failures = ()
                # check if there are any not used phonemes before res[start] and if the last match is perfect
            elif res[start] - last_ep > 0:
                if not is_last_perfect_match:
                    # if not, apply these phonemes to the last match
                    word2ph[-1] += res[start] - last_ep
                else:
                    plus = distribute_phonemes(res[start] - last_ep, 2)
                    extra += plus[1]
                    if len(word2ph) > 0:
                        word2ph[-1] += plus[0]
                    else:
                        extra += plus[0]
                total -= res[start] - last_ep
            word2ph.append(res[end] - res[start] + 1 + extra)
            last_ep = res[end] + 1
            is_last_perfect_match = True
        elif not start_p and stop_p:
            available = res[start] - last_ep
            assert available >= 0, ("If you get this error and the input of this function is ALREADY "
                                    "normalized, this may be a logical bug, please "
                                    "consider to report it on github")
            extra = 0
            if available or match_failures:
                if match_failures:
                    if is_last_perfect_match:
                        if available > match_failures[1] - match_failures[0] + 1:
                            available -= 1
                            extra = 1
                            total -= 1
                        word2ph += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                        total -= available
                    else:
                        if available > match_failures[1] - match_failures[0] + 2:
                            available -= 2
                            extra = 1
                            word2ph[-1] += 1
                            total -= 2
                        word2ph += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                        total -= available
                    match_failures = ()
                elif is_last_perfect_match:
                    # apply these to the current char
                    extra = available
                else:
                    plus = distribute_phonemes(res[start] - last_ep, 2)
                    extra += plus[1]
                    if len(word2ph) > 0:
                        word2ph[-1] += plus[0]
                    else:
                        extra += plus[0]
                    total -= res[start] - last_ep
            word2ph.append(res[end] - res[start] + 1 + extra)
            total -= res[end] - res[start] + 1
            is_last_perfect_match = True
            last_ep = res[end] + 1
        elif start_p and not stop_p:
            if match_failures:
                available = res[start] - last_ep
                assert available >= 0, ("If you get this error and the input of this function is ALREADY "
                                        "normalized, this may be a logical bug, please "
                                        "consider to report it on github")
                if not is_last_perfect_match:
                    if available > match_failures[1] - match_failures[0] + 1:
                        word2ph[-1] += 1
                        available -= 1
                        total -= 1
                word2ph += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                total -= available
                match_failures = ()
            elif res[start] - last_ep > 0:
                if not is_last_perfect_match:
                    word2ph[-1] += res[start] - last_ep
                else:
                    plus = distribute_phonemes(res[start] - last_ep, 2)
                    extra += plus[1]
                    if len(word2ph) > 0:
                        word2ph[-1] += plus[0]
                    else:
                        extra += plus[0]
                total -= res[start] - last_ep
            word2ph.append(res[end] - res[start] + 1 + extra)
            total -= res[end] - res[start] + 1
            is_last_perfect_match = False
            last_ep = res[end] + 1
        else:
            available = res[start] - last_ep
            assert available >= 0, ("If you get this error and the input of this function is ALREADY "
                                    "normalized, this may be a logical bug, please "
                                    "consider to report it on github")
            if available or match_failures:
                if match_failures:
                    if is_last_perfect_match:
                        if available > match_failures[1] - match_failures[0] + 1:
                            available -= 1
                            extra = 1
                            total -= 1
                        word2ph += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                        total -= available
                    else:
                        if available > match_failures[1] - match_failures[0] + 2:
                            available -= 2
                            extra = 1
                            word2ph[-1] += 1
                            total -= 2
                        word2ph += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                        total -= available
                    match_failures = ()
                elif is_last_perfect_match:
                    extra = available
                else:
                    plus = distribute_phonemes(available, 2)
                    if len(word2ph) > 0:
                        word2ph[-1] += plus[0]
                    else:
                        extra += plus[0]
                    extra += plus[1]
                    total -= available
            word2ph.append(res[end] - res[start] + 1 + extra)
            total -= res[end] - res[start] + 1
            last_ep = res[end] + 1
            is_last_perfect_match = False
        if word2ph[-1] < 0:
            raise AssertionError
    if match_failures:
        available = len(sc) - last_ep + 1
        assert available >= 0
        if not is_last_perfect_match:
            if match_failures[1] - match_failures[0] + 1 < available:
                word2ph[-1] += 1
                available -= 1
                total -= 1
        word2ph += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
    # print(word2ph, sum(word2ph), len(sc))
    assert sum(word2ph) == len(sc), ("If you get this error and the input of this function is ALREADY normalized, "
                                     "this may be a logical bug, please consider to report it on github")
    assert len(word2ph) == len(norm_text), ("If you get this error and the input of this function is ALREADY "
                                            "normalized, this may be a logical bug, please "
                                            "consider to report it on github")
    return word2ph


def swap(a, b):
    return b, a


_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}


def japanese_convert_numbers_to_words(text: str) -> str:
    res = text
    # res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    # res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    for x in _CURRENCY_MAP.keys():
        res = res.replace(x, _CURRENCY_MAP[x])
    # res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
}


def replace_punctuation(text):
    # pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = text
    # replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    for x in rep_map.keys():
        replaced_text = replaced_text.replace(x, rep_map[x])
    return replaced_text


def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = "".join([i for i in res if is_japanese_character(i)])
    res = replace_punctuation(res)
    return res


if __name__ == '__main__':
    # demo for matching, designed to be visual
    sentence = 'わたしはマスターの所有物ですので。勝手に売買するのは違法です'
    norm = text_normalize(sentence)
    print(g2p_with_accent_info(text_normalize(sentence)))
    print("word2ph:", calculate_word2ph(text_normalize(sentence)))
    sa, word2ph = g2p_with_accent_info(text_normalize(sentence))
    print('new word2ph:', word2ph)
    sb = []
    sa = [''] + sa
    g2p_r, _ = g2p_with_accent_info(text_normalize(sentence))
    for tm in text_normalize(sentence):
        if tm in ["!", "?", "…", ",", ".", "'", "-"]:
            sb.append('pau')
        else:
            sb += pjt.g2p(tm).split(' ')

    sb = [''] + sb
    if len(sa) < len(sb):
        sa, sb = swap(sa, sb)
    lge = 0
    for x in sa:
        lge = max(lge, len(x))
    for x in sb:
        lge = max(lge, len(x))
    res, ans = LCS_solver(sa, sb)
    cr = {}
    for x in res.keys():
        cr[x[1]] = x[0]
    tn = len(text_normalize(sentence))
    SP = 114  # if you get formatting issues while checking results, try setting this variable bigger
    sp = SP * len(sa)
    sat = list(sp * " ")
    opr = []
    for x in cr.keys():
        sat[SP * cr[x] - len(str(sb[x - 1])):SP * cr[x]] = list(str(sb[x - 1]))
        opr.append((x, cr[x]))
    opr.sort()
    for x in range(len(opr) - 1):
        sn = sb[opr[x][0]:opr[x + 1][0] - 1]
        if not sn:
            continue
        tl = 0
        for _ in sn:
            tl += len(str(_))
        rs = (SP * opr[x + 1][1] - len(str(sb[opr[x + 1][0] - 1]))) - SP * opr[x][1] - 1 - len(sn) - tl
        spn = rs // len(sn)
        up = "["
        for it in sn:
            up += " " * spn + str(it) + " "
        up = up[:-1] + (rs - spn * len(sn)) * " " + "]"
        sat[SP * opr[x][1]:SP * opr[x + 1][1] - len(str(sb[opr[x + 1][0] - 1]))] = list(up)
    sbt = ""
    for x in sa:
        sbt += " " * (SP - len(str(x))) + str(x)
    sbt = list(sbt)
    pf = sb[:opr[0][0] - 1]
    tlp = 0
    for x in pf:
        tlp += len(x)
    if len(pf) < 2:
        pf += (2 - len(pf)) * ['']
    sat[:SP * opr[0][1] - len(sb[opr[0][0] - 1])] = list(
        f"[{(' ' * ((SP * opr[0][1] - len(sb[opr[0][0] - 1]) - 2 - tlp) // (len(pf) - 1))).join(pf)}"
        f"{' ' * (SP * opr[0][1] - 2 - tlp - len(sb[opr[0][0] - 1]) - (((SP * opr[0][1] - len(sb[opr[0][0] - 1]) - 2 - tlp) // (len(pf) - 1)) * (len(pf) - 1)))}]"
    )
    sat = "".join(sat)
    # sbt = "".join(sbt)
    tbr = []
    x = 0
    while x < len(sat):
        if sat[x:x + SP - lge] == " " * (SP - lge):
            tbr.append((x, x + SP - lge - 1, SP - lge - 1))
            x += SP - lge
        else:
            x += 1
    sat = list(sat)
    sbt = list(sbt)
    mi = 0
    x = 0
    print(sat)
    print(sbt)
    while x < len(sat) - 1:
        if sat[x] == sbt[x] == ' ' and (sbt[x - 1] == sbt[x + 1] == ' ') and (sat[x - 1] == sat[x + 1] == ' '):
            sat.pop(x)
            sbt.pop(x)
        else:
            x += 1
    print(f"successfully matched {ans} group{'s' if ans > 1 else ''}, similarity:{round(ans / len(sa) * 100, 2)}%, "
          f"{round(ans / len(sb) * 100, 2)}%")
    print("".join(sat))
    print("".join(sbt))
    off_w = 0
    op1 = ""
    op2 = ""
    for x in range(tn):
        if not word2ph[x]:
            continue
        phonemes = g2p_r[off_w:off_w+word2ph[x]]
        st = " ".join(phonemes) + "    "
        op1 += st + "|"
        op2 += f"[{str(x+1).center(len(st)-2, ' ')}]|"
        off_w += word2ph[x]
    print(op1.replace('/', '↑').replace(']', '↓'))
    print(op2)
    for cnt, x in enumerate(norm):
        print((cnt+1, x), end=" ")
    print(f"\n{norm}")
