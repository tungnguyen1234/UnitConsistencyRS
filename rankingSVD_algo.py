import torch as t
import gc

class rankingSVD():
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def train(self, r_train, percent):
        n_factors = int(min(self.rows, self.cols) * percent)
        if r_train.is_sparse:
            U, S, Q = t.svd_lowrank(r_train, q = n_factors)
            Q_remain = Q @ Q.t()
            del U, S, Q  # Free up memory
            t.cuda.empty_cache()
            gc.collect()
            return Q_remain
        else:
            R = t.tensor(r_train, dtype = t.double)
            U, S, Q = t.svd_lowrank(R, q = n_factors)
            pred = R @ Q @ Q.t()
            return pred