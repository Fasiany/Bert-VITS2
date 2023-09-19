import bisect
from typing import List
import pyopenjtalk as pjt


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


def LCS_solver(seq_a: List[int], seq_b: List[int]) -> (dict, int):
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


def distribute_phonemes(cn, n):
    return [cn // n] * cn, cn - (n * (cn // n))


def calculate_word2ph_by_matching_char_and_phone(norm_text, tokenized) -> list:
    sc = pjt.g2p(norm_text).split(' ')
    ss = ['']
    for tm in norm_text:
        if tm in ["!", "?", "…", ",", ".", "'", "-"]:
            if ss[-1] != 'pau':
                ss.append('pau')
        else:
            ss.append(tm)
    ss.pop(0)
    word2ph = []
    # TODO:Complete the rest part (match known relationships of the sequence and distribute unknown phonemes)

    return word2ph


def swap(a, b):
    return b, a


if __name__ == '__main__':
    # demo for matching, designed to be visual
    # sb = [1, 11, 12, 13, 2, 7, 8, 9, 5, 3, 2, 9, 1, 4, 0]
    # sa = [1, 3, 3, 2, 4, 5, 8, 2, 5, 3, 7, 8, 5, 1, 3, 2, 1, 3, 4]
    sa = list("fUtokorokashiidarekaniyobibaretamitainasoNnakaifunoonakiwakedesu 114514".lower())
    sb = list('natsUkashiidarekaniyobaretamitainapausoNnakaisekIfunoonakibuNdesU 1919810'.lower())
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
        pf.append('')
    sat[:SP * opr[0][1] - len(sb[opr[0][0] - 1])] = list(
        f"[{(' ' * ((SP * opr[0][1] - len(sb[opr[0][0] - 1]) - 2 - tlp) // (len(pf) - 1))).join(pf)}{' ' * (SP * opr[0][1] - 2 - tlp - len(sb[opr[0][0] - 1]) - (((SP * opr[0][1] - len(sb[opr[0][0] - 1]) - 2 - tlp) // (len(pf) - 1)) * (len(pf) - 1)))}]")
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
    for x in tbr:
        sat[x[0] - mi:x[1] - mi] = []
        sbt[x[0] - mi:x[1] - mi] = []
        mi += x[2]
    x = 0
    while x < len(sat) - 1:
        if sat[x] == sbt[x] == ' ' and (sbt[x - 1] == sbt[x + 1] == ' ') and (sat[x - 1] == sat[x + 1] == ' '):
            sat.pop(x)
            sbt.pop(x)
        else:
            x += 1
    print(f"successfully matched {ans} group{'s' if ans > 1 else ''}")
    print("".join(sat))
    print("".join(sbt))
