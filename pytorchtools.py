"""
Taken from https://github.com/Bjarten/early-stopping-pytorch
"""
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.max_epochs = 0

    def __call__(self, val_score, model, epoch):

        score = val_score
        assert epoch >= self.max_epochs
        self.max_epochs = epoch

        if self.best_score is None:
            print(">>>> First epoch: best_score is None")
            self.best_score = score
            self.save_checkpoint(val_score, model)
            return True
        elif score < self.best_score + self.delta: # If the score didn't improve
            self.counter += 1
            self.trace_func(
                f"  - EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
            return True

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.path))
        return model

    def save_checkpoint(self, val_score, model):
        if self.path is None:
            return

        """Saves model when metric increases."""
        if self.verbose:
            msg = "   > Score increased"
            msg = msg + f"({self.best_score:.4f} --> {val_score:.4f})"
            msg = msg + " - Saving model to " + self.path + "..."
            self.trace_func(msg)
        
        torch.save(model.state_dict(), self.path)
