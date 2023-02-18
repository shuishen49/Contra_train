# Contra_train
#视频地址

[![Watch the video](https://i1.hdslb.com/bfs/archive/575686852210ead66e4ae41bc2a420526c7e4125.jpg@320w_200h_1c_!web-space-index-myvideo.avif
)](https://www.bilibili.com/video/BV1JA41127yF/?share_source=copy_web&vd_source=ffb7da622ce658d7366a91f7aeff01b2)

#关于模型
| 关卡 | 最好的模型 |
| --- | --- |
| `第一关` | best_model_1430000.zip |
| `第二关` | 正再训练 |

Contra_train
魂斗罗训练模型
需要自己安装

gym 

retro

等库

需要先导入游戏。包含rom文件。

并且你需要先修改下面几个文件
C:\ProgramData\Anaconda3\Lib\site-packages\retro\data\stable\Contra-Nes

包含
data.json

Level1-99.state

scenario.json

因为原版这几个文件是不正确的。

我并没有训练出第一关可以一命通关的模型。可能超参设置有问题导致的。

请使用 train.py开始训练模型

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/shuishen49/Contra_train/blob/main/train.py)
