from ColourNet import ColourNet
import torch
import os
from Plotter import Plotter
from DataHandler import DataHandler
from argparse import ArgumentParser
from train import train_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
           

if __name__ == '__main__':
    parser = ArgumentParser(prog='b2c', description='Transforms black and white images to color using ColourNet 2.35M parameters')
    parser.add_argument('mode', choices=('train', 'convert'))
    parser.add_argument('device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('-p', '--path')
    parser.add_argument('-d', '--destination')
    parser.add_argument('-b', '--batch_size', default=8)
    parser.add_argument('-i', '--iterations', default=20000)
    parser.add_argument('-r', '--reset', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu'
    print('Running on', 'gpu...' if device is 'cuda' else 'cpu...')
    

    if args.mode == 'convert':
        assert args.path is not None
        assert args.destination is not None

        data_handler = DataHandler(split='convert', device=device)
        colourNet = ColourNet(device)
        colourNet.load()

        x = data_handler.path_to_file_to_tensor(args.path)
        img_tensor = colourNet.b2c(x)
        data_handler.save_tensor_as_image(img_tensor, args.destination)
        print("Image saved to", args.destination)

    else:
        print("Trainning model from", "scratch..." if args.reset else "last checkpoint...")
        train_model(int(args.batch_size), args.reset, device=device, iter=int(args.iterations))
