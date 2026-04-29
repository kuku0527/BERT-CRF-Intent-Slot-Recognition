import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model.bert_crf import BertCRF
from utils.data_loader import IntentSlotDataset

# 配置
bert_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集
train_set = IntentSlotDataset('data/train.txt', tokenizer)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# 模型初始化
model = BertCRF(bert_name, num_intents=2, num_slots=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# 训练循环
model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, intent_labels, slot_labels = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        loss, _, _ = model(input_ids, attention_mask, intent_labels, slot_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
