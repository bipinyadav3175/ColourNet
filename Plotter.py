import matplotlib.pyplot as plt
from queue import Queue

class Plotter():
    def __init__(self, mode='train'):
        if mode not in ('train', 'test'):
            raise ValueError("Please specify a valid plotting mode")
        self.mode = mode
        self.gen_loss = []
        self.i = 0
    
    def add_loss(self, gen_loss):
        self.i += 1
        self.gen_loss.append(gen_loss)

        self.gen_loss = self.gen_loss[-1000:]


        if self.i%50 == 0:
            self.save_graph()
    

    def save_graph(self):
        x = range(1, len(self.gen_loss) + 1)
        plt.plot(x, self.gen_loss, 'r-', label='Generator')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.savefig("losses_train.png" if self.mode == 'train' else 'losses_test.png')

    