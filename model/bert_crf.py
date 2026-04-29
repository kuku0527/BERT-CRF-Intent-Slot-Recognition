import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class BertCRF(nn.Module):
    def __init__(self, bert_name, num_intents, num_slots, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_linear = nn.Linear(hidden_size, num_intents)
        self.slot_linear = nn.Linear(hidden_size, num_slots)
        self.crf = CRF(num_slots, batch_first=True)

    def forward(self, input_ids, attention_mask, intent_labels=None, slot_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        seq_out = self.dropout(outputs.last_hidden_state)

        intent_logits = self.intent_linear(pooled)
        slot_logits = self.slot_linear(seq_out)

        if intent_labels is not None and slot_labels is not None:
            intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_labels)
            slot_loss = -self.crf(slot_logits, slot_labels, mask=attention_mask.bool(), reduction='mean')
            return intent_loss + slot_loss, intent_logits, slot_logits
        else:
            return intent_logits, self.crf.decode(slot_logits, mask=attention_mask.bool())
