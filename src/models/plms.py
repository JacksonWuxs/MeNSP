import tqdm
import numpy as np
import torch as tc
import transformers as trf


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = tc.argmax(tc.abs(u), 0)
    i = tc.arange(u.shape[1]).to(u.device)
    signs = tc.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PLM(tc.nn.Module):
    def __init__(self, name, head, tokenizer=None, device="cpu"):
        super(PLM, self).__init__()
        assert head in ("mlm", "clm", "pretrain", "max", "mean", "pooler", "cls", "s2s")
        self.device = device
        self.head = head
        self._name = name + head
        auto_model = {"mlm": trf.AutoModelForMaskedLM,
                      "clm": trf.AutoModelForCausalLM,
                      "s2s": trf.AutoModelForSeq2SeqLM,
                      "pretrain": trf.AutoModelForPreTraining,\
                      }.get(head, trf.AutoModel)
        self.encoder = auto_model.from_pretrained(name, return_dict=False)
        self.dim = self.encoder.config.hidden_size
        self.tokenizer = tokenizer if tokenizer else PLM.load_tokenizer(name)
        for names in [("cls", "bos"), ("sep", "eos"), ("mask",)]:
            token = ""
            for name in names:
                if hasattr(self.tokenizer, name + "_token"):
                    token = getattr(self.tokenizer, name + "_token")
                    break
            setattr(self, name + "_token", token)
        self.to(self.device)

    @classmethod
    def load_tokenizer(cls, name):
        return trf.AutoTokenizer.from_pretrained(name)

    def disable_training(self):
        self.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            param.requires_grad_(False)

    def w2i(self, tokens):
        if len(tokens) == 0:
            return []
        if not isinstance(tokens[0], str):
            return [self.w2i(_) for _ in tokens]
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def i2w(self, ids):
        if len(ids) == 0:
            return []
        if not isinstance(ids[0], int):
            return [self.i2w(_) for _ in ids]
        return self.tokenizer.convert_ids_to_tokens(ids)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def _prepare(self, ids, embs, segs, masks):
        if ids is not None and ids.device != self.device:
            ids = ids.to(dtype=tc.long, device=self.device, non_blocking=True)
        if embs is not None and embs.device != self.device:
            embs = embs.to(dtype=tc.float32, device=self.device, non_blocking=True)
        if masks is not None and masks.device != self.device:
            masks = masks.to(dtype=tc.long, device=self.device, non_blocking=True)
        elif ids is not None:
            masks = tc.where(ids >= 1, 1.0, 0.0).to(dtype=tc.long,device=self.device, non_blocking=True)
        elif embs is not None:
            masks = tc.where(tc.sum(embs.abs(), -1) > 0, 1.0, 0.0).to(dtype=tc.long, device=self.device, non_blocking=True)
        if self.head in ("s2s",):
            decoder_ids, decoder_embs = None, None
            if ids is not None:
                decoder_ids = self.encoder._shift_right(ids)
            if embs is not None:
                decoder_embs = embs.new_zeros(embs.shape)
                decoder_embs[:, 1:] = embs[:, :-1].clone()
                decoder_embs[:, 0] = self.encoder.get_input_embeddings()(tc.LongTensor([self.encoder.config.decoder_start_token_id]).to(self.device))
            return {"input_ids": ids,
                    "decoder_input_ids": decoder_ids, 
                    "inputs_embeds": embs, 
                    "decoder_inputs_embeds": decoder_embs,
                    "attention_mask": masks}
        if segs is not None and segs.device != self.device:
            segs = segs.to(dtype=tc.long, device=self.device, non_blocking=True)
        return {"input_ids": ids,
                "inputs_embeds": embs,
                "token_type_ids": segs,
                "attention_mask": masks}

    def _postprocess(self, inputs, outputs, index, return_logits):
        if self.head == "pretrain":
            assert len(outputs) == 2
            return outputs

        if self.head == "pooler":
            return outputs[1]

        if self.head == "cls":
            return outputs[0][:, 0, :]

        if self.head == "mean":
            embs, mask = outputs[0], inputs["attention_mask"]
            embs = embs * mask.unsqueeze(-1)
            return tc.sum(embs, 1) / (mask.sum(1, keepdim=True) + 1e-9)

        if self.head == "max":
            return tc.max(outputs[0], 1).values

        if self.head == "mlm":
            if index is None:
                assert inputs["input_ids"] is not None
                index = tc.nonzero(inputs["input_ids"] == self.tokenizer.mask_token_id, as_tuple=False).to(self.device)
            logits = outputs[0][index[:, 0], index[:, 1]].squeeze()
            
        elif self.head in ("clm", "s2s"):
            logits = outputs[0][:, -1, :]
        
        if return_logits:
            return logits
        return tc.softmax(logits, -1)
                
    def forward(self, ids=None, embs=None, segs=None, masks=None, index=None, return_logits=True):
        inputs = self._prepare(ids, embs, segs, masks)
        outputs = self.encoder(**inputs)
        return self._postprocess(inputs, outputs, index, return_logits)

    def encode(self, texts, padding=32, batchsize=64, total=None):
        bar = tqdm.tqdm(total=total // batchsize)
        embs, batch = [], []
        bs = 0
        with tc.no_grad():
            self.eval()
            for _, ids in enumerate(texts):
                ids = ids[:padding]
                ids.extend([0] * (padding - len(ids)))
                batch.append(ids)
                if len(batch) == batchsize:
                    inputs = tc.tensor(batch, device=self.device).reshape((len(batch), -1)).long()
                    embs.append(self(inputs))
                    batch.clear()
                    bar.update(1)
                    bs += 1
            if len(batch) > 0:
                inputs = tc.tensor(batch, device=self.device)
                embs.append(self(inputs.reshape((len(batch), -1))))
            return self._whitening(tc.cat(embs, axis=0))
        
    def _whitening(self, X):
        # PCA
##        mu = tc.mean(X, dim=0, keepdim=True)
##        X = X - mu
##        U, S, V = tc.svd(X)
##        #U, Vt = svd_flip(U, V)
##        accumulate, sum_S = 0.0, sum(S.detach().cpu().tolist())
##        for idx, s in enumerate(S.detach().cpu().tolist(), 1):
##            accumulate += s / sum_S
##            if accumulate > 0.85:
##                break
##        X = X[:, idx:] @ tc.diag(S)[idx:, idx:] @ V[idx:, idx:]
        
        # whitening
##        u, s, vt = tc.svd(tc.mm(X.T, X) / (X.shape[0] - 1.0))
##        W = tc.mm(u, tc.diag(1.0 / tc.sqrt(s)))
##        X = tc.mm(X, W)
        return X

    
    

if __name__ == "__main__":
    sentences = ["this is a test case [MASK]", "this is the user case [MASK]"]
    plm = PLM("gpt2", head=None)
    ids = [plm.tokenizer.convert_tokens_to_ids(plm.tokenizer.tokenize(_)) for _ in sentences]
    
    print(plm.encode(ids, total=2).shape)

