import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys

# set this variable to the path of bert model
BERT = '/home/zhang/PycharmProjects/Bert-VITS2_E/bert/bert-large-japanese-v2'

tokenizer = AutoTokenizer.from_pretrained(BERT)
BERT_LAYER = 20
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
model = AutoModelForMaskedLM.from_pretrained(BERT).to(device_g)


def get_bert_feature(text, word2ph, device=None):  # arg device is actually not used.Keep it here for compatibility
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device_g)
        res = model(**inputs, output_hidden_states=True)
        # we will get 25 layers in bert large
        # print(f"res shape:{res['hidden_states'].shape}")
        res = res['hidden_states'][BERT_LAYER]
    # print(len(word2ph), inputs['input_ids'].shape)
    assert inputs['input_ids'].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[0][i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
