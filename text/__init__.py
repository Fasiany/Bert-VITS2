import torch

from text.symbols import *
from text.character_phoneme_matching import LCS_solver

SHOW_EXTRA_INFO = True

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert_train(norm_text, bert, word2ph, tokenizer):
    nn = len(norm_text)
    w2p = []
    off = 1
    assert len(norm_text) == len(word2ph) - 2, f"{norm_text}, {len(word2ph)}\n{word2ph}"
    unk_exists = False
    tokenized = tokenizer.tokenize(norm_text)
    nt = len(tokenized)
    lre = {}
    cor_table = {}
    if "[UNK]" in tokenized:
        if SHOW_EXTRA_INFO:
            print(f"Warning:Sentence {norm_text} contains character(s) that is "
                  f"unknown to the tokenizer([UNK] tokens).If you get "
                  f"errors after this warning, try to remove this sentence from the dataset list."
                  f"The algorithm will ATTEMPT to fix this problem automatically, which may throw exceptions."
                  f"You can also use --ignore-if-unk-exists opinion while running"
                  f" preprocess_text.py to avoid this problem")
        sqt = []
        for tv in tokenized:
            if tv == "[UNK]":
                continue
            sqt += list(tv.replace("#", ""))
        lre, _ = LCS_solver(sqt, list(norm_text))

        for x in lre.keys():
            cor_table[x[0]] = x[1]
    off2 = 0
    for cnt, x in enumerate(tokenized):
        if x == '[UNK]':
            # print(lre)
            if off != 1:
                std = cor_table[max(off - 1 - off2, 1)]
                end = max(cor_table[off - off2], 1)
            else:
                std = 1
                end = cor_table[1]
            x = "-" * (end - std - 1 + (1 if off == 1 else 0))
            off2 += len(x)
            # print(list(norm_text))
            # print(sqt)
            # print(cor_table)
            # print(off, std, end)
            if SHOW_EXTRA_INFO:
                print(f"[UNK AUTOFIX]RAW:{norm_text[max(0, std - 5):min(nn, end + 7)]}\n\t\tTOKENIZED:"
                      f"{tokenized[max(0, cnt - 1)].replace('#', '').replace('[UNK]', '')}[UNRECOGNIZED]"
                      f"{tokenized[min(nt - 1, cnt + 1)].replace('#', '')}\n\t\tCalculated "
                      f"length:{end - std - 1 + (1 if off == 1 else 0)}"
                      f"(Unrecognized content starts at index {std + (1 if off > 1 else 0)} and ends at index {end - 1})")
            unk_exists = True
            assert end - std - 1 + (1 if off == 1 else 0)
        w2p.append(sum(word2ph[off:off + len(x.replace("#", ""))]))
        off += len(x.replace("#", ""))
    w2p = [word2ph[0]] + w2p + [word2ph[-1]]
    try:
        assert sum(w2p) == sum(word2ph), (w2p, word2ph, tokenizer.tokenize(norm_text), norm_text)
    except AssertionError as value:
        if not unk_exists:
            raise value
        raise RuntimeError(f"Word2ph length mismatch the target length, and at least one [UNK] has been detected in "
                           f"tokenized sequence.This means the auto fixing algorithm has failed(Which also means there"
                           f" might be bugs in the algorithm, please consider to report it on github)."
                           f"Look up to inspect the warning(before traceback, "
                           f"if SHOW_EXTRA_INFO is True) of this sentence "
                           f"to get potential solution:\n{repr(value)}")
    phone_level_feature = []
    for i in range(len(w2p)):
        if not w2p[i]:
            continue
        repeat_feature = bert[i].repeat(w2p[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def get_bert(norm_text, word2ph, language, device):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    return bert
