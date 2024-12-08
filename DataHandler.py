import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import random
from matplotlib.image import imsave

class DataHandler():
    def __init__(self, batch_size=32, device=None, split='convert'):
        self.device = device
        if device == None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if split not in ('train', 'test', 'convert'):
            raise ValueError("Please Specify a valid data split to use")
        
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((512, 512)),
        ])
        
        if split == 'convert':
            return

        self.files = os.listdir('grayscale')
        self.files = self.files[ : int(len(self.files) * 0.8)] if split == 'train' else self.files[int(len(self.files) * 0.8) : ]
        random.Random(29).shuffle(self.files)
        self.batch_size = batch_size
        self.used = 0

    
    def all_done(self):
        return self.used >= len(self.files)
        

    def get_batch(self):

        batch = random.sample(self.files, k=self.batch_size)
        y = torch.stack([self.path_to_file_to_tensor(os.path.join('original', file)) for file in batch])
        # y = None
        x = torch.stack([self.path_to_file_to_tensor(os.path.join('grayscale', file)) for file in batch])
        return x, y


    def path_to_file_to_tensor(self, path):
        img = Image.open(path)

        img = self.transform(img).to(self.device).type(torch.float) / 127.5 - 1# [0, 255] -> [0, 1]
        # img = transforms.Normalize(0, 1)(img)
        return img
    
    def save_tensor_as_image(self, img, dest):
        if dest is None:
            raise ValueError("Incorrect path to save image")
        transform = transforms.ToPILImage()
        img = transform((img + 1)/2)
        img.save(dest)
    


if __name__ == '__main__':
    handler = DataHandler()
    img = handler.path_to_file_to_tensor(os.path.join('grayscale', 'COCO_train2014_000000000009.jpg'))
    print(img.shape)
    print(img)