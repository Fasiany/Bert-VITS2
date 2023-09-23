import json
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
from text.cleaner import clean_text
from text.japanese import process_bert
from emotion_extract import preprocess_one
from text.japanese_bert import tokenizer


@click.command()
@click.option(
    "--transcription-path",
    default="filelists/genshin.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default="filelists/train.list")
@click.option("--val-path", default="filelists/val.list")
@click.option(
    "--config-path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    transcription_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
):
    if cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"
    errors = []
    if clean:
        out_file = open(cleaned_path, "w", encoding="utf-8")
        for line in tqdm(open(transcription_path, encoding="utf-8").readlines()):
            fae = False
            try:
                utt, spk, language, text = line.strip().split("|")
                process_bert(text, utt.replace(".wav", ".bert.pt"))
                preprocess_one(utt)
                norm_text, phones, tones, word2ph = clean_text(text, language)
                w2p = word2ph
                # due to unknown reasons, the value of tokenized sequence does not keep the same
                # assert sum(w2p) == sum(word2ph), f"{tokenizer.tokenize(text)}, {w2p}, {word2ph}, {norm_text}, {len(norm_text)}, {len(w2p)}, {len(word2ph)}"
                # fae = True
                # assert len(w2p) == len(tokenizer.tokenize(text))+2, f"{tokenizer.tokenize(text)}, {w2p}, {word2ph}, \n{norm_text}, {len(norm_text)}, {len(w2p)}, {len(word2ph)}"
                out_file.write(
                    "{}|{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        " ".join([str(i) for i in w2p]),
                    )
                )
            except Exception as error:
                if fae:
                    raise Exception
                print('error!')
                errors.append((error, line))

        out_file.close()

        transcription_path = cleaned_path
    if errors:
        print(f"{len(errors)} error{'s' if len(errors) > 1 else ''} occurred during cleaning:")
        for err in errors:
            print(f'{repr(err[0])}:{err[1]}')
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(transcription_path, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
