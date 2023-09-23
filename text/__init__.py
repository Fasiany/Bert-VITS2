import torch

from text.symbols import *

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = []  # _symbol_to_id[symbol] for symbol in cleaned_text
    for symbol in cleaned_text:
        try:
            phones.append(_symbol_to_id[symbol])
        except KeyError:
            phones.append(0)  # symbol not found in ID map, use 0('_') by default
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert_train(norm_text, bert, word2ph, tokenizer):
    w2p = []
    # word2ph.pop(0)
    # word2ph.pop(-1)
    off = 1
    assert len(norm_text) == len(word2ph) - 2, f"{norm_text}, {len(word2ph)}\n{word2ph}"
    for x in tokenizer.tokenize(norm_text):
        w2p.append(sum(word2ph[off:off+len(x.replace("#", ""))]))
        off += len(x.replace("#", ""))
    w2p = [word2ph[0]] + w2p + [word2ph[-1]]
    phone_level_feature = []
    # [1, 2, 3, 3, 2, 6, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 1, 3, 5, 2, 5, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 4, 2, 4, 2, 2, 3, 1, 2, 1, 1, 1, 3, 1]
    # [3, 4, 6, 6, 4, 12, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 4, 2, 6, 10, 4, 10, 4, 4, 4, 0, 0, 0, 0, 0, 0, 2, 4, 6, 4, 2, 8, 4, 8, 4, 4, 6, 2, 4, 2, 2, 2, 6, 2]
    # print(f'processing bert with:{w2p}\n{len(w2p)}\n{word2ph}')
    for i in range(len(w2p)):
        repeat_feature = bert[i].unsqueeze(0).repeat(w2p[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def get_bert(norm_text, word2ph, language, device, tokenizer):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    return bert
