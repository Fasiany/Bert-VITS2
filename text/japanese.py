# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import unicodedata

import pyopenjtalk
import torch
from transformers import AutoTokenizer

from text.character_phoneme_matching import g2p_with_accent_info, calculate_word2ph
from text.symbols import symbols
from text.japanese_bert import get_bert_feature, BERT  # BERT is the path to the model

_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}


def japanese_convert_numbers_to_words(text: str) -> str:
    res = text
    for x in _CURRENCY_MAP.keys():
        res = res.replace(x, _CURRENCY_MAP[x])
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
    "…": "...",
}


def replace_punctuation(text):
    replaced_text = text

    for x in rep_map.keys():
        replaced_text = replaced_text.replace(x, rep_map[x])

    return replaced_text


def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    res = replace_punctuation(res)
    return res.replace('\n', '').replace(" ", "")


tokenizer = AutoTokenizer.from_pretrained(BERT)


def g2p(norm_text, apply_accent_info=False):
    if apply_accent_info:
        return g2p_a(norm_text)
    phs = pyopenjtalk.g2p(norm_text).split(" ")
    word2ph = calculate_word2ph(norm_text)
    phs = ['_'] + phs + ['_']
    tns = [0 for _ in phs]
    word2ph = [1] + word2ph + [1]
    return phs, tns, word2ph


def g2p_a(norm_text):
    phs, word2ph = g2p_with_accent_info(norm_text)
    phonemes = ['_'] + phs + ['_']
    tones = [0 for _ in phonemes]
    word2ph = [1] + word2ph + [1]
    return phonemes, tones, word2ph


def process_bert(txt, file=None):
    txt = text_normalize(txt)
    bert_info = get_bert_feature(text_normalize(txt))
    if file is not None:
        torch.save(bert_info, file)
    return bert_info


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(BERT)
    text = "Hello,こんにちは、世界！……"

    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)
