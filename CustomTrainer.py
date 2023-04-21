import torch
from transformers import Trainer
from builtins import AttributeError
import math as m

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        if labels is not None:
            print('Labels found')
        else:
            print(inputs)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 4 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0, 4.0]))
        #try:
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        print(loss)
        #print(loss.backward())
        #except AttributeError as err:
        #    N = self.model.config.num_labels
        #    loss = torch.tensor(-m.log(1/N), requires_grad=True)
        return (loss, outputs) if return_outputs else loss