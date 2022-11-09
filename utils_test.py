import numpy as np
from transformers import AutoModel, AutoTokenizer, XLMRobertaTokenizer
import torch


def to_cuda(inputs, is_tensor=True):
    for e in inputs:
        if not is_tensor:
            inputs[e] = torch.tensor(inputs[e])
        inputs[e] = inputs[e].cuda()
    return inputs


class wrapper():
    def __init__(self, args):
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.model = self.model.cuda()
        if 'roberta' in args.model_name_or_path:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.args = args

    def _encode_batch(self, batch):
        args = self.args
        with torch.no_grad():
            outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
        if args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()

    def encode_texts(self, texts, batch_size=128, max_length=32):
        self.model.eval()
        with torch.no_grad():
            embeddings = []
            text_ids = []
            for i in range(len(texts)//batch_size + 1):
                s = i*batch_size
                e = (i+1)*batch_size
                if s >= len(texts):
                    break
                inputs = self.tokenizer(texts[s:e], padding=True, truncation=True, return_tensors="pt", max_length=max_length)
                inputs = to_cuda(inputs)
                outputs = self._encode_batch(inputs)
                for emb in outputs:
                    embeddings.append(emb)
                for one_id in inputs['input_ids']:
                    text_ids.append(one_id)
            return torch.stack(embeddings,dim=0)