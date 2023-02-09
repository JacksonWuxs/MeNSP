import random
import torch as tc
import numpy as np
import tqdm
import transformers as trf


def batch_pad(idx, pad):
    if not isinstance(idx, tc.Tensor):
        maxlen = max(map(len, idx))
        idx = tc.tensor([_ + [pad] * (maxlen - len(_)) for _ in idx])
    if len(idx.shape) == 1:
        idx = idx.unsqueeze(0)
    return idx


class PretrainedLanguageModel(tc.nn.Module):
    def __init__(self, name, head, device):
        tc.nn.Module.__init__(self)
        assert head in ("mean", "cls", "max")
        self.head = head
        self.encoder = trf.AutoModelForPreTraining.from_pretrained(name, output_hidden_states=True)
        self.tokenizer = trf.AutoTokenizer.from_pretrained(name)
        self.pad = self.tokenizer.pad_token
        self.mask = self.tokenizer.mask_token
        self.cls = self.tokenizer.cls_token
        self.sep = self.tokenizer.sep_token
        self.padID = self.w2i(self.pad)
        self.device = device
        self.to(device)

    def w2i(self, tokens):
        if len(tokens) == 0:
            return []
        if isinstance(tokens, str):
            return self.tokenizer.convert_tokens_to_ids([tokens])[0]
        if not isinstance(tokens[0], str):
            return [self.w2i(_) for _ in tokens]
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def i2w(self, ids):
        if len(ids) == 0:
            return []
        if isinstance(ids, int):
            return self.tokenizer.convert_ids_to_tokens([ids])[0]
        if not isinstance(ids[0], int):
            return [self.i2w(_) for _ in ids]
        return self.tokenizer.convert_ids_to_tokens(ids)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def forward(self, ids, masks=None, segs=None, *args, **kwrds):
        ids = batch_pad(ids, self.padID)
        if ids.device != self.device:
            ids = ids.to(dtype=tc.long, device=self.device, non_blocking=True)
        if masks is None:
            masks = tc.where(ids != self.padID, 1.0, 0.0)
        masks = masks.to(dtype=tc.long, device=self.device, non_blocking=True)
        if segs is not None:
            segs = batch_pad(segs, 0).to(dtype=tc.long, device=self.device, non_blocking=True)
        outputs = dict(self.encoder(input_ids=ids, token_type_ids=segs, attention_mask=masks))
        outputs["mask"] = masks
        return outputs
    
    def _get_embedding(self, outputs):
        if self.head == "cls":
            return outputs['hidden_states'][-1][:, 0, :].cpu()
        if self.head == "mean":
            embed = outputs['hidden_states'][-1].sum(axis=1)
            return (embed / outputs["mask"].sum(axis=1, keepdims=True)).cpu()
        if self.head == "max":
            return outputs['hidden_states'][-1].max(axis=1).values.cpu()
        raise NotImplementedError("Does not support head %s for generating embeddings" % (self.head,))

    def _get_pairproba(self, outputs, choices=1):
        assert choices >= 1
        logits = outputs['seq_relationship_logits']
        if choices == 1:
            return logits.cpu()
        return logits[:, 0].reshape(-1, choices).cpu()

    def encode(self, texts, maxlen=None, batchsize=None):
        batch, embeds = [], []
        for tokens in texts:
            if isinstance(tokens, str):
                tokens = self.tokenize(tokens) 
            batch.append(self.w2i([self.cls] + tokens[:maxlen] + [self.sep]))
            if len(batch) == batchsize:
                embeds.append(self._get_embedding(self(batch)))
                batch.clear()
        if len(batch) > 0:
            embeds.extend(self._get_embedding(self(batch)))
        return tc.nn.functional.normalize(tc.vstack(embeds))

    def pairwise(self, text_A, text_B, maxlen=499, batchsize=1):
        text_B = [self.tokenize(each) if isinstance(each, str) else each for each in text_B]

        probas, batch_ids, batch_segs = [], [], []
        for tokens_A in text_A:
            if isinstance(tokens_A, str):
                tokens_A = self.tokenize(tokens_A)
            for tokens_B in text_B:
                ids, seg = self._concate_pair(tokens_A, tokens_B, 499)
                batch_ids.append(ids)
                batch_segs.append(seg)
            if len(batch_ids) // len(text_B) == batchsize:
                probas.append(self._get_pairproba(self(ids=batch_ids, segs=batch_segs), len(text_B)))
                batch_ids.clear()
                batch_segs.clear()
        if len(batch_ids) > 0:
            probas.extend(self._get_pairproba(self(ids=batch_ids, segs=batch_segs), len(text_B)))
        return tc.vstack(probas)

    def _concate_pair(self, texta, textb, maxlen):
        if maxlen is not None:
            texta, textb = texta.copy(), textb.copy()
            while len(texta) + len(textb) > maxlen:
                if len(texta) > len(textb):
                    texta.pop(-1)
                else:
                    textb.pop(-1)
        ids = [self.cls] + textb + [self.sep] + texta 
        seg = [0] * (len(textb) + 2) + [1] * len(texta)
        return self.w2i(ids), seg
        


class TwoStages(tc.nn.Module):
    def __init__(self, plmname, device, *args, **kwrds):
        tc.nn.Module.__init__(self)
        self.plm = PretrainedLanguageModel(plmname, "max", device)
        self.rubrics = self.context = self.threshold = None
        self.device = device

    def set_rubrics(self, rubrics):
        self.rubrics = [self.plm.tokenize(_) for _ in rubrics]
        self.context = self.plm.encode(self.rubrics, maxlen=510, batchsize=None)
        self.threshold = ((self.context[-1:] @ self.context[:2].T) * tc.tensor([0.35, 0.65])).sum().item()
        self.threshold = ((self.context[-1:] @ self.context[:2].T) * tc.tensor([0.5, 0.5])).sum().item()
##        print(self.context @ self.context.T, self.threshold)
        self.probas = self.plm.pairwise(self.rubrics, self.rubrics, batchsize=4).mean(axis=0, keepdims=True)

    def predict(self, data, batchsize=4, **args):
        with tc.no_grad():
            self.plm.eval()
            scores = self.plm.encode(data, maxlen=510, batchsize=batchsize) @ self.context[-1:].T # (bs, )
            similar = self.plm.pairwise(data, self.rubrics, batchsize=batchsize) # (bs, labels)
        assert scores.shape[0] == similar.shape[0] == len(data)
        assert similar.shape[1] == len(self.rubrics)
        condition = (scores <= self.threshold).squeeze()
        similar[condition, 0] = 1e5
        similar[~condition, 0] = -1e5
        similar[condition, 1:] = -1e5
        proba = tc.softmax(similar, -1)
        return proba

    def _fit_batch(self, optm, lossfn, batchX, batchY, bar):
        optm.zero_grad()
        predict = self.plm.pairwise(batchX, self.rubrics[1:], batchsize=None)
        loss = lossfn(tc.softmax(predict, -1), tc.tensor(batchY).long())
        loss.backward()
        optm.step()
        batchX.clear()
        batchY.clear()
        bar.update(1)
        return loss.cpu().item()

    def fit(self, data, labels, batchsize=8, epochs=1000,
            learnrate=3e-3, dropout=0.1, smoothing=0.1):
        no_decay = {'bias', 'LayerNorm.weight'}
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.plm.named_parameters() if "seq_relationship" in n and not any(nd in n for nd in no_decay)],'weight_decay': 0.0},
            {'params': [p for n, p in self.plm.named_parameters() if "seq_relationship" in n and any(nd in n for nd in no_decay)],'weight_decay': 0.0}
        ]
        batchsize = min(batchsize, len(data))
        data = [self.plm.tokenize(_) for _ in data]
        optm = tc.optim.AdamW(optimizer_grouped_parameters, lr=learnrate)
        lossfn = tc.nn.CrossEntropyLoss(label_smoothing=smoothing)
        bar = tqdm.tqdm(total=len(data) // batchsize * epochs)
        predicts, reals = [], []
        for epoch in range(epochs):
            self.train()
            samples = list(range(len(data)))
            random.shuffle(samples)
            batchX, batchY = [], []
            for idx in samples:
                label, sample = labels[idx], data[idx]
                if label == 0:
                    continue
                batchX.append([_ for _ in sample if random.random() >= dropout])
                batchY.append(label - 1)
                if len(batchX) == batchsize:
                    self._fit_batch(optm, lossfn, batchX, batchY, bar)
            if len(batchX) > 0:
                self._fit_batch(optm, lossfn, batchX, batchY, bar)

                
                
        
        
        
