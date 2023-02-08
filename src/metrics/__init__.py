

import numpy as np
import torch as tc
from sklearn import metrics


class Evaluator:
    def __init__(self, data, batch_size, num_class=4):
        self.dataset = tc.utils.data.DataLoader(data, batch_size=batch_size,
                                                num_workers=2, shuffle=False)
        self.preds, self.reals = [], []
        self.pids, self.rids = [], []

    def evaluate(self, model):
        torch_model = isinstance(model, tc.nn.Module)
        self.clear()
        if torch_model:
            model.eval()
        with tc.no_grad():
            bar = tqdm.tqdm(total=len(self.dataset))
            for batch in self.dataset:
                if len(batch["problem"]) == 1:
                    for key, val in batch.items():
                        batch[key] = val.unsqueeze(0)
                
                pred = model(**batch).reshape(-1,)
