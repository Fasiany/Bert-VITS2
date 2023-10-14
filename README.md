# 暂时搁置中

由于马上就要csp-s复赛，~~在教练的威光下~~我决定搁置项目至2023-10-21 18:00 (GMT+8)

同步了一次最新进度，**理论上**项目目前是可以使用与训练的(从最新的可用测试进度直接复制来的，可能要把ja_bert维度信息改回768)。最新进度正在进行测试与实验，慢速推进中

仍在开发中，目前项目应用了新g2p算法与来自**HTSEngine**的基本语调/停顿辅助信息。彻底重写后的g2p算法大幅提升数据利用率，鲁棒性与准确度(具体表现为g2p报错概率极大幅度地降低并提升bert特征信息计算流程准确性)

请注意训练需要load底模，项目只支持日语训练，但是推理的时候可以加

目前已经实测确认的事情：

1.new g2p 可行性:是

2.new g2p+emotion embeddings 可行性:是

3.new g2p+bert feature :成功 

4.new g2p+bert feature+emotion embedddings ....失败(TE.fw eeb aout)

5.???? 实验中

(估计这个项目还得花上不少时间吧)

一定要确保句子包含的所有特殊符号都在symbols.py内，否则匹配算法可能会炸。特别是各种空格与全角符号(一般来说只影响预处理，因为炸了的不会进训练数据集)

如果句子里面只有一堆标点符号没有可发音的文字匹配算法也有可能会炸

---

# Bert-VITS2

VITS2 Backbone with bert
## 成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应该参阅代码自己学习如何训练。
### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
### 严禁用于任何政治相关用途
#### Video:https://www.bilibili.com/video/BV1hp4y1K78E
#### Demo:https://www.bilibili.com/video/BV1TF411k78w
## References
+ [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS)
+ [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
+ [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
+ [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
+ [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
