import bisect
import copy
import re
import unicodedata
from typing import List, Any

import pyopenjtalk as pjt

from text.symbols import punctuation, symbols


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
    for i, x2 in enumerate(rg):
        usages[x2] = 0
        cor[x2] = i + 1
    pos = []
    for x2 in range(len(seq_a)):
        pos.append([])
    for x2 in range(n):
        pos[cor[seq_a[x2]]].append(x2)
    tm2 = []
    for x2 in range(len(seq_b)):
        try:
            t_pos = pos[cor[seq_b[x2]]][usages[seq_b[x2]]]
            usages[seq_b[x2]] += 1
        except (IndexError, KeyError):
            tm2.append(-1)
        else:
            tm2.append(t_pos)
    return LIS_solver(tm2)


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
    result = {}
    while cur[0] and cur[1]:
        nxt = trans[cur]
        if nxt == (cur[0] - 1, cur[1] - 1):
            result[cur] = nxt
        cur = nxt
    return result, dp[n][m]


REVERSE = True


def distribute_phonemes(pn, cn):
    resu = [pn // cn] * cn
    rem = pn - cn * (pn // cn)
    for i in range(rem):
        resu[i] += 1
    if REVERSE:
        resu.reverse()
    return resu


def numeric_feature_by_regex(regex, s):
    # https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html#10.2-Tacotron-2-%E3%82%92%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%81%AB%E9%81%A9%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AE%E5%A4%89%E6%9B%B4
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def g2p_with_accent_info(norm_text: str):
    """
    Parameters:
        norm_text: The normalized text
    returns phonemes sequence with accent information and pause flags and the word2ph sequence of it
    """
    # https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html#10.2-Tacotron-2-%E3%82%92%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%81%AB%E9%81%A9%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AE%E5%A4%89%E6%9B%B4
    fl = pjt.extract_fullcontext(norm_text)
    n = len(fl)
    result = []
    word2phonemes = calculate_word2ph(norm_text)
    rcm = {}
    off = 0
    for count, x2 in enumerate(word2phonemes):
        for sui in range(x2):
            rcm[off + sui] = count
        off += x2
    new_word2ph = copy.deepcopy(word2phonemes)
    ed = '\n'
    if not fl:
        print(f"Can't extract any context from [{norm_text.strip(ed)}]!")
    assert n == sum(new_word2ph) + 2, (f"Preprocessing failed, can't proceed with this text![{norm_text.strip(ed)}]\n"
                                       f"Expect length of context to be {sum(new_word2ph) + 2}, found {len(fl)}\n"
                                       f"word2ph:{new_word2ph}")
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
                try:
                    _ = norm_text[rcm[current - gof + 1]]
                    gof -= 1
                except KeyError:
                    pass
                if norm_text[rcm[current - gof]] == '!':
                    result.append('!')
                elif e3:
                    result.append('?')
                    new_word2ph[rcm[current - gof]] += 1
                else:
                    result.append('*')
                    new_word2ph[rcm[current - gof]] += 1
            continue
        elif p3 == 'pau':
            e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
            if norm_text[rcm[current - gof]] == '!':
                result.append('!')
            elif e3:
                result.append('?')
            else:
                result.append(',')
        else:
            result.append(p3.replace('ch', 'ʧ').replace('sh', 'ʃ').replace('cl', 'Q').replace('ts', 'ʦ'))
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", fl[current + 1])

        if a3 == 1 and a2_next == 1:
            pass
            # result.append("#")
            # new_word2ph[rcm[current + 1 - gof]] += 1
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            result.append("]")
            new_word2ph[rcm[current + 1 - gof]] += 1
        elif a2 == 1 and a2_next == 2:
            result.append("/")
            new_word2ph[rcm[current + 1 - gof]] += 1
    if norm_text[-1] == '!' and not new_word2ph[-1] and result[-1] == '*':
        i = -2
        try:
            while new_word2ph[i] < 2:
                i -= 1
            new_word2ph[i] -= 1
            new_word2ph[-1] += 1
        except IndexError:
            pass
        result[-1] = '!'
    assert len(result) == sum(new_word2ph)
    assert min(new_word2ph) >= 0
    return result, new_word2ph


def get_item(key, dic, default):
    try:
        return dic[key]
    except KeyError:
        return default


def calculate_word2ph(norm_text: str) -> list:
    """
    Parameters:
        norm_text: The normalized text
    returns the word2ph(Number of phonemes in each character) sequence of the given input
    """
    sc = [x.lower().replace('ch', 'ʧ').replace('sh', 'ʃ').replace('cl', 'Q').replace('ts', 'ʦ') for x in
          pjt.g2p(norm_text).split(' ')]
    ss = []
    cm = {}
    rcm = {}
    punc_num = 0
    for count, tc in enumerate(norm_text):
        if tc in punctuation:
            punc_num += 1
            cm[count] = (len(ss) + 1, len(ss) + 1)
            ss.append('pau')
        else:
            si = len(ss)
            subs = [tc.lower().replace('ch', 'ʧ').replace('sh', 'ʃ').replace('cl', 'Q').replace('ts', 'ʦ') for tc in
                    pjt.g2p(tc).split(' ')]
            ss += subs
            cm[count] = (si + 1, si + len(subs))
            for ind in range(si, si + len(subs)):
                rcm[ind] = count + 1
    result, _ = LCS_solver(ss, sc)
    tres = {}
    for x in result.keys():
        tres[x[0]] = x[1]
    result = tres
    total = len(sc)
    word2phonemes = []
    match_failures = ()  # (start_char_id, stop_char_id)
    is_last_perfect_match = True
    last_ep = 1
    for x in range(len(norm_text)):
        start = cm[x][0]
        end = cm[x][1]  # phoneme range in per-processed sequence (index starts from 1)
        start_p = True
        stop_p = True
        while not get_item(start, result, False) and start <= end:
            start += 1
            start_p = False
        while not get_item(end, result, False) and start <= end:
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
            total -= result[end] - result[start] + 1
            if match_failures:
                # previous matches failed.Need to distribute available phonemes to them.
                available = result[start] - last_ep
                assert available >= 0, ("If you get this error and the input of this function is ALREADY "
                                        "normalized, this may be a logical bug, please "
                                        "consider to report it on github")
                if not is_last_perfect_match:
                    # if the last successful match is not perfect, give it up to 1
                    if available > match_failures[1] - match_failures[0] + 1:
                        word2phonemes[-1] += 1
                        available -= 1
                        total -= 1
                word2phonemes += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                total -= available
                match_failures = ()
                # check if there are any not used phonemes before res[start] and if the last match is perfect
            elif result[start] - last_ep > 0:
                if not is_last_perfect_match:
                    # if not, apply these phonemes to the last match
                    word2phonemes[-1] += result[start] - last_ep
                else:
                    plus = distribute_phonemes(result[start] - last_ep, 2)
                    extra += plus[1]
                    if len(word2phonemes) > 0:
                        word2phonemes[-1] += plus[0]
                    else:
                        extra += plus[0]
                total -= result[start] - last_ep
            word2phonemes.append(result[end] - result[start] + 1 + extra)
            last_ep = result[end] + 1
            is_last_perfect_match = True
        elif not start_p and stop_p:
            available = result[start] - last_ep
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
                        word2phonemes += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                        total -= available
                    else:
                        if available > match_failures[1] - match_failures[0] + 2:
                            available -= 2
                            extra = 1
                            word2phonemes[-1] += 1
                            total -= 2
                        word2phonemes += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                        total -= available
                    match_failures = ()
                elif is_last_perfect_match:
                    # apply these to the current char
                    extra = available
                else:
                    plus = distribute_phonemes(result[start] - last_ep, 2)
                    extra += plus[1]
                    if len(word2phonemes) > 0:
                        word2phonemes[-1] += plus[0]
                    else:
                        extra += plus[0]
                    total -= result[start] - last_ep
            word2phonemes.append(result[end] - result[start] + 1 + extra)
            total -= result[end] - result[start] + 1
            is_last_perfect_match = True
            last_ep = result[end] + 1
        elif start_p and not stop_p:
            if match_failures:
                available = result[start] - last_ep
                assert available >= 0, ("If you get this error and the input of this function is ALREADY "
                                        "normalized, this may be a logical bug, please "
                                        "consider to report it on github")
                if not is_last_perfect_match:
                    if available > match_failures[1] - match_failures[0] + 1:
                        word2phonemes[-1] += 1
                        available -= 1
                        total -= 1
                word2phonemes += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                total -= available
                match_failures = ()
            elif result[start] - last_ep > 0:
                if not is_last_perfect_match:
                    word2phonemes[-1] += result[start] - last_ep
                else:
                    plus = distribute_phonemes(result[start] - last_ep, 2)
                    extra += plus[1]
                    if len(word2phonemes) > 0:
                        word2phonemes[-1] += plus[0]
                    else:
                        extra += plus[0]
                total -= result[start] - last_ep
            word2phonemes.append(result[end] - result[start] + 1 + extra)
            total -= result[end] - result[start] + 1
            is_last_perfect_match = False
            last_ep = result[end] + 1
        else:
            available = result[start] - last_ep
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
                        word2phonemes += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                        total -= available
                    else:
                        if available > match_failures[1] - match_failures[0] + 2:
                            available -= 2
                            extra = 1
                            word2phonemes[-1] += 1
                            total -= 2
                        word2phonemes += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
                        total -= available
                    match_failures = ()
                elif is_last_perfect_match:
                    extra = available
                else:
                    plus = distribute_phonemes(available, 2)
                    if len(word2phonemes) > 0:
                        word2phonemes[-1] += plus[0]
                    else:
                        extra += plus[0]
                    extra += plus[1]
                    total -= available
            word2phonemes.append(result[end] - result[start] + 1 + extra)
            total -= result[end] - result[start] + 1
            last_ep = result[end] + 1
            is_last_perfect_match = False
        if word2phonemes[-1] < 0:
            raise AssertionError
    if match_failures:
        available = len(sc) - last_ep + 1
        assert available >= 0
        if not is_last_perfect_match:
            if match_failures[1] - match_failures[0] + 1 < available:
                word2phonemes[-1] += 1
                available -= 1
                total -= 1
        word2phonemes += distribute_phonemes(available, match_failures[1] - match_failures[0] + 1)
    assert sum(word2phonemes) == len(sc), (
        "If you get this error and the input of this function is ALREADY normalized, "
        "this may be a logical bug, please consider to report it on github\n"
        f"{word2phonemes}, {sc}\n"
        f"{sum(word2phonemes)}, {len(sc)}\n"
        f"{norm_text}"
        )
    assert len(word2phonemes) == len(norm_text), ("If you get this error and the input of this function is ALREADY "
                                                  "normalized, this may be a logical bug, please "
                                                  "consider to report it on github\n"
                                                  f"{word2phonemes}, {len(norm_text)}"
                                                  )
    assert min(word2phonemes) >= 0, ("ERR minimum less than 0.If you get this error and the "
                                     "input of this function is ALREADY "
                                     "normalized, this may be a logical bug, please "
                                     "consider to report it on github"
                                     )
    return word2phonemes


def swap(a, b):
    return b, a


_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}


def japanese_convert_numbers_to_words(text: str) -> str:
    result = text
    for x2 in _CURRENCY_MAP.keys():
        result = result.replace(x2, _CURRENCY_MAP[x2])
    return result


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
    "…": "...",
    "―": "-",
}


def replace_punctuation(text):
    replaced_text = text

    for x2 in rep_map.keys():
        replaced_text = replaced_text.replace(x2, rep_map[x2])
    return replaced_text


def text_normalize(text):
    result = unicodedata.normalize("NFKC", text)
    result = japanese_convert_numbers_to_words(result)
    result = replace_punctuation(result)
    return result.replace('\n', '').replace(" ", "")


if __name__ == '__main__':
    # demo for matching, designed to be visual
    # sentence = open('filelists/train.list', encoding='utf-8').readlines()[721].split('|')[3]
    # print(sentence)
    sentence = 'っ――!'
    # sentence = '1...!'
    norm = text_normalize(sentence)
    ers = []
    # for cnt, x in enumerate(open('fl.txt', encoding='utf-8').readlines()):
    #     try:
    #         g2p_with_accent_info(text_normalize(x.split('|')[3].strip('\n')))
    #     except AssertionError as err:
    #         ers.append((err, x.split("|")[3], cnt))
    # for x in ers:
    #     print(f"{x[1]} -> {str(x[0])}:\n{x[2]}")
    print(g2p_with_accent_info(text_normalize(sentence)))
    print("word2ph:", calculate_word2ph(text_normalize(sentence)))
    sa_, word2ph = g2p_with_accent_info(text_normalize(sentence))
    sa = []
    for x in sa_:
        sa.append(x.lower())
    print('new word2ph:', word2ph)
    sb = []
    sa = [''] + sa
    g2p_r, _ = g2p_with_accent_info(text_normalize(sentence))
    for tm in text_normalize(sentence):
        if tm in ["!", "?", "…", ",", ".", "'", "-"]:
            sb.append('pau')
        else:
            sb += [x.lower() for x in pjt.g2p(tm).split(' ')]

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
    sat = list(sat)
    sbt = list(sbt)
    mi = 0
    x = 0
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
        phonemes = g2p_r[off_w:off_w + word2ph[x]]
        st = " ".join(phonemes) + "    "
        op1 += st + "|"
        op2 += f"[{str(x + 1).center(len(st) - 2, ' ')}]|"
        off_w += word2ph[x]
    print(op1.replace('/', '↑').replace(']', '↓'))
    print(op2)
    for cnt, x in enumerate(norm):
        print((cnt + 1, x), end=" ")
    print(f"\n{norm}")
