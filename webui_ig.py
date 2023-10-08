# flake8: noqa: E402

import sys, os
import logging
import IPython.display as ipd
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import argparse
import numpy as np
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert, get_bert_train
import gradio as gr
import webbrowser
from text.japanese import g2p, text_normalize, tokenizer

net_g = None

if sys.platform == "darwin" and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = "cuda"

# TODO:update webui for its code was tested in previous environment


def get_text(text, word2ph, phone, tone, language_str, wav_path):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if hps.data.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            nw = []
            for i in range(len(word2ph)):
                nw.append(word2ph[i] * 2)
            nw[0] += 1
            word2ph = nw
        emotion_path = wav_path.replace(".wav", ".emo.npy")
        bert = get_bert(text, word2ph, language_str, "cuda", tokenizer)
        bert = get_bert_train(text_normalize(text), bert, word2ph, tokenizer)
        assert bert.shape[-1] == len(phone), f"length of phonemes does not match input length of bert:{len(phone)}, {bert.shape}, {text}, {word2ph}"
        assert language_str == 'JP', "This project only supports Japanese for now."
        emotion = torch.FloatTensor(np.load(emotion_path))
        # emotion = torch.zeros(1024*[0])
        ja_bert = bert
        # dimension info of bert:[1024, len(phonemes)]
        assert ja_bert.shape[-1] == len(phone), f"""length of phonemes does not match input length of bert:{(
            ja_bert.shape,
            len(phone),
            len(word2ph),
            word2ph,
            text,
        )}"""
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return emotion, ja_bert, phone, tone, language


def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language, emo, emo_all_zero=False):
    global net_g
    phones, tones, word2ph = g2p(text_normalize(text))
    print(phones)
    emotion, ja_bert, phones, tones, lang_ids = get_text(text, word2ph, phones, tones, language, emo)
    print("infer:", phones)
    print(phones)
    print(tones)
    print(lang_ids)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        if emo_all_zero:
            emotion = torch.FloatTensor([0] * 1024)
        emotion = emotion.to(device).unsqueeze(0)
        # ja_bert = torch.zeros(ja_bert.shape)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        print("length of input：", x_tst.shape, ja_bert.shape, lang_ids
              .shape, emotion.shape, x_tst_lengths.shape, tones.shape)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                # emotion,
                tones,
                lang_ids,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0] #                 emotion, ja_bert,
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, emotion, x_tst_lengths, speakers
        return audio


def tts_fn(
    text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language
):
    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sid=speaker,
            language=language,
        )
        torch.cuda.empty_cache()
    return "Success", (hps.data.sampling_rate, audio)


if __name__ == "__main__":
    P = 'MUTI_BERT_ONLY'
    S = "8000"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default=f"./logs/{P}/G_{S}.pth", help="path of your model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/config_P.json",
        help="path of your config file",
    )
    parser.add_argument(
        "--share", default=False, help="make link public", action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log"
    )

    args = parser.parse_args()
    if args.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(args.config)

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "mps"
            if sys.platform == "darwin" and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print("length of symbols:", len(symbols))
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None, skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP"]
    FILL_EMO_WITH_ZEROS = True
    # FILL_EMO_WITH_ZEROS = False
    ad = infer("1, 2, 3, それは?", 0.2, 0.667, 0.8, 1, 'PM', 'JP', 'ATRI_VD_WAV_48K/ATR_b102_051.wav', FILL_EMO_WITH_ZEROS)
    open("temp.wav", "wb").write(ipd.Audio(ad, rate=hps.data.sampling_rate, normalize=False).data)
    exit()
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(
                    label="Text",
                    placeholder="Input Text Here",
                    value="吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。",
                )
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="Speaker"
                )
                sdp_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.2, step=0.1, label="SDP Ratio"
                )
                noise_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise Scale"
                )
                noise_scale_w = gr.Slider(
                    minimum=0.1, maximum=2, value=0.8, step=0.1, label="Noise Scale W"
                )
                length_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=1, step=0.1, label="Length Scale"
                )
                language = gr.Dropdown(
                    choices=languages, value=languages[0], label="Language"
                )
                btn = gr.Button("Generate!", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio")

        btn.click(
            tts_fn,
            inputs=[
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
            ],
            outputs=[text_output, audio_output],
        )

    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share)