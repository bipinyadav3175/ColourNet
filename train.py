from ColourNet import ColourNet
import torch
import os
import time
from Plotter import Plotter
from DataHandler import DataHandler

def train_model(batch_size=8, reset=False, device='cpu', iter=20000):
    data_handler = DataHandler(batch_size, split='train', device=device)
    colourNet = ColourNet(device)
    plotter = Plotter(mode='train')

    if 'ColourNet_G.pth' in os.listdir(os.curdir) and not reset:
        colourNet.load()

    best_loss = float('inf')

    for i in range(iter):

        grayscale, color = data_handler.get_batch()

        # Training the Generator
        loss = colourNet.train_G(grayscale, color)


        # Save the model if loss is decresed
        if loss < best_loss:
            colourNet.save()

        plotter.add_loss(loss)
        print(i, (i*data_handler.batch_size*100)/len(data_handler.files), "%  ", "Total Loss: ", loss)

    print('Training finished')


if __name__ == '__main__':
    train_model()
    