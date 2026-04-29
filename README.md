
# BERT-CRF-Intent-Slot-Recognition
BERT-CRF 中文意图识别与槽位填充

基于 BERT + CRF 实现智能客服领域意图识别与槽位填充联合任务，采用预训练语言模型提取上下文语义特征，结合CRF完成序列标签约束，适用于对话系统、智能客服语义理解场景。

## 项目功能
1. 中文用户意图分类
2. 文本关键信息槽位抽取
3. 数据集预处理、模型训练、评估与离线推理

## 技术栈
Python、PyTorch、Transformers、BERT、CRF、序列标注

## 项目结构
data/    存放训练、验证、测试数据集
model/   BERT-CRF 模型定义
utils/   数据加载、评价指标工具
train.py 模型训练脚本
predict.py 单句预测推理脚本

## 运行方式
1. 安装依赖
pip install -r requirements.txt

2. 模型训练
python train.py

3. 模型推理
python predict.py --text "我想修改手机号"
