# 版本：  
基本平台：pycharm2019/anaconda5.2.0/ubuntu18.04  
## 核心库文件  
keras==2.1.5  
tensorflow==1.4  
python==3.6  
# 文件说明：  
data/H5/:训练模型文件  
result/:测试集文件  
SuperResolution/:其他超分辨率模型文件夹
# 已训练模型
本项目提供一个基于DIV2K剪裁96x96共计5W张图像训练400Epoch的模型，设置了选用最佳模型  
# 测试
测试，直接predict即可，更改相应路径  
# 训练
训练，调用train函数，需要更改训练图像文件夹  

