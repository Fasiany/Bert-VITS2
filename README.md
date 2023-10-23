此项目是某人推完 **ATRI -My Dear Moments-** 后一拍脑袋搞出的东西。这意味着此分支不倾向于将项目做成production-ready的样子。同时，也不会刻意支持与优化日语之外的语言，也不会跟进原仓库此方面的更新。

如果您希望使用成熟可靠的项目，请使用原仓库。

另外，也不需要对这个东西有什么期望，我不能保证未来的自己某一天不会突然删库(如果是我自作多情请忽略这句话)

# 搁置中

备考NOIP

同步了一次最新进度，**理论上**项目目前是可以使用与训练的(从最新的可用测试进度直接复制来的。最新进度正在进行测试与实验，慢速推进中

~~仍在开发中~~是否放弃开发待定。目前项目应用了新g2p算法与来自**HTSEngine**的基本语调/停顿辅助信息。彻底重写后的g2p算法(与1.1.1master分支相比)大幅提升数据利用率，鲁棒性与准确度(具体表现为g2p报错概率极大幅度地降低并提升bert特征信息计算流程准确性)

训练需要load底模，只支持日语训练(阉割了中文部分，因为中文部分没有任何改动~~而且处理起来还需要花额外的精力~~。需要中文直接使用原项目即可)，推理时可以放简单的英语单词(复杂的会逐个念字母)


(估计这个项目还得花上不少时间吧)

一定要确保句子包含的所有特殊符号都在symbols.py内，否则匹配算法可能会炸。特别是各种空格与全角符号(一般来说只影响预处理，因为炸了的不会进训练数据集)

如果句子里面只有一堆标点符号没有可发音的文字匹配算法也有可能会炸


# Usage 

训练集列表格式请参照filelists/esd.list文件格式

1.预处理音素列表等信息：

~~~bash
python preprocess.py --transcription-path your_dataset_list [--train-path training_dataset_path] [--val-path evaluating_dataset_path] [--config-path config_file_path] [--ignore-if-unk-exists]
~~~

注：默认值为filelists/train.list, filelists/val.list, configs/config.json，下同

指定--ignore-if-unk-exists选项将丢弃tokenize后含有未知字符的训练数据，可防止训练报错。训练时遇到此类数据算法将发出警告并尝试自行修正问题

2.生成bert特征信息(你也许应该把训练集目录下的*.pt文件先删干净)：

~~~bash
python bert_gen.py [-c your_config_file]
~~~

3.启动训练

开始前请确保已经设置完RANK, WORLD_SIZE, MASTER_ADDR与MASTER_PORT环境变量

~~~bash
python train_ms.py -c config_file -m model_name
~~~

4.图形推理webui界面

~~~bash
python webui.py -c config_file -m model_file
~~~
---

# Bert-VITS2

VITS2 Backbone with bert
## ~~成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应该参阅代码自己学习如何训练。~~
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
+ [CjangCjengh/japanese_g2p](https://github.com/CjangCjengh/japanese_g2p)
+ [https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html](https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html)
+ [innnky/emotional-vits](https://github.com/innnky/emotional-vits)
