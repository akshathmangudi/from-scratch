import numpy as np 

class SGD:
    def __init__(self, lr: float = 0.01, m: float = 0.2, 
                 b: float = 0.2, epoch: int = 10, batch_size: int = 1):
        self.lr = lr
        self.m = m
        self.b = b
        self.epoch = epoch
        self.batch_size = batch_size
        
        self.log = []
        self.mse = []
        
    def update(self, X, y): 
        for _ in range(self.epoch):
            indexes = np.random.randint(0, len(X), self.batch_size)
            Xs = np.take(X, indexes)
            ys = np.take(y, indexes)
            N = len(Xs)

            f = ys - (self.m * Xs + self.b)

            self.m -= self.lr * (-2 * Xs.dot(f).sum() / N)
            self.b -= self.lr * (-2 * f.sum() / N)

            self.log.append((self.m, self.b))
            mse = np.mean((y - (self.m * X + self.b)) ** 2)
            self.mse.append(mse)


        return self.m, self.b, self.log, self.mse