import torch
from transformers import BertTokenizer
from model.bert_crf import BertCRF

bert_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertCRF(bert_name, num_intents=2, num_slots=5).to(device)
model.eval()

def predict(text):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors='pt').to(device)
    with torch.no_grad():
        intent_logits, slot_preds = model(**encoding)
    intent = torch.argmax(intent_logits, dim=-1).item()
    return intent, slot_preds[0]

if __name__ == "__main__":
    print(predict("我想修改手机号"))
