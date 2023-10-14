import sys

import torch
from text import get_bert_train
from transformers import BertJapaneseTokenizer, AutoModelForMaskedLM as BertModel

# set this variable to the path of bert model
BERT = "/home/zhang/PycharmProjects/Bert-VITS2_E/bert/bert-large-japanese-v2"

tokenizer = BertJapaneseTokenizer.from_pretrained(BERT)
BERT_LAYER = -3
if torch.cuda.is_available():
    device_g = "cuda"
elif (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
):
    device_g = "mps"
else:
    device_g = "cpu"
print(f'loading bert on device {device_g}')
model = BertModel.from_pretrained(BERT).to(device_g)


def get_bert_feature(text, word2ph=None, device=None):  # arg device is actually not used.Keep it here for compatibility
    if word2ph is None:
        word2ph = []
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        inputs = inputs.to(device_g)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res['hidden_states'][BERT_LAYER:BERT_LAYER+1], -1)
    return get_bert_train(text, res[0].cpu(), word2ph, tokenizer)
