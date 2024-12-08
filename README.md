# ColourNet
Convert black and white (grayscale) images to colour images

### Installation
```
pip install -r requirements.txt
```

### Usage
```
b2c.py convert <device> -p <path> -d <destination>
```
+ device -> 'gpu' or 'cpu'
+ path -> location of grayscale image
+ destination -> location of color image to be saved

### Training
```
b2c.py train <device> -b <batch_size> -i <iterations>
```
+ device -> 'gpu' or 'cpu'
+ batch_size -> (default 8)
+ iterations -> (default 20000)
+ use -r flag to start the training from beginning instead of continuation

For training make sure you have the grayscale images in './grayscale/' directory and the respective colour images with the same name in './original/' directory
