_punctuation = ',.!?-~…'
punctuation = list(_punctuation)
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"
OOR = 'T'

num_zh_tones = 6

# japanese
ja_symbols = list('AEINOQUabdefghijklmnoprstuvwyzʃʧʦɯɹəɥ⁼ʰ`→↓↑/]*^#') + ['ky'] + ['gy'] + ['ry'] + ['hy'] + ['T']
num_ja_tones = 1


# combine all symbols
normal_symbols = sorted(set(ja_symbols))
symbols = [pad] + normal_symbols + pu_symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# combine all tones
num_tones = num_zh_tones + num_ja_tones

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
num_languages = len(language_id_map.keys())

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
}
