import torch
from Autoencoder import Autoencoder
import torch.nn.functional as F
import torch.nn as nn

class ColourNet():
    def __init__(self, device=None):
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device


        self.generator = Autoencoder().to(device)
        self.generator.apply(self.weights_init)
        self.optimizer_generator = torch.optim.AdamW(self.generator.parameters(), lr=1e-3)


    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    

    def train_G(self, grayscale, color):
        self.generator.train()
        
        
        generated = self.generator(grayscale)

        mse = F.mse_loss(generated.flatten(1), color.flatten(1))
        loss = mse

        self.optimizer_generator.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1)
        self.optimizer_generator.step() # update weights

        return loss.item()
    
    def test(self, x, y):
        """
        Returns Accuracy of predictions
        """
        if not isinstance(x, torch.Tensor):
            raise f"x is not a torch.Tensor. x = {type(x)}"
        
        if not isinstance(y, torch.Tensor):
            raise f"y is not a torch.Tensor. x = {type(y)}"
        
        # y_pred = self.model(x)
        # loss = F.mse_loss(y_pred, y)

        # return  (1 - loss.item()) * 100
    
    def b2c(self, x):
        """
        Returns a tensor of coloured image from the input black&white image
        """
        self.generator.eval()
        if not isinstance(x, torch.Tensor):
            raise f"x is not a torch.Tensor. x = {type(x)}"

        with torch.no_grad():
            return self.generator(x.unsqueeze(0))[0]

    
    def save(self):
        torch.save(self.generator.state_dict(), 'ColourNet_G.pth')

    def load(self):
        self.generator.load_state_dict(torch.load('ColourNet_G.pth'))