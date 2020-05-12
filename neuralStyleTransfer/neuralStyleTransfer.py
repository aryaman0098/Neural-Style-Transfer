import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.io
import scipy.misc
from torchvision import transforms , models
import torch
 
#Hiding unwanted warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Displaying the content images and styling images
contentImage = scipy.misc.imread("input/vegeto.jpg")
plt.imshow(contentImage)
plt.show()
styleImage = scipy.misc.imread("input/gogeta.jpg")
plt.imshow(styleImage)
plt.show()

#Loading VGG19 convolutional neural network
model = models.vgg19(pretrained=True).features

for p in model.parameters():
    p.requires_grad = False

#Selecting the layers whose activations will be taken in neural style network, and storing it in a dictionary
def modelActivations(input, model):
    layers = {
    '0' : 'conv1_1',
    '5' : 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '21': 'conv4_2',
    '28': 'conv5_1'
    }
    features = {}
    x = input
    x = x.unsqueeze(0)
    for name,layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x 
    return features

#Resizing the image
print(contentImage.shape)
transform = transforms.Compose([transforms.Resize(450),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


#Loading the content and style image
contentImage = Image.open("input/vegeto.jpg").convert("RGB")
contentImage = transform(contentImage)
styleImage = Image.open("input/gogeta.jpg").convert("RGB")
styleImage = transform(styleImage)
print("Content Image Shape = ", contentImage.shape)


#Converting the tensor image into a displayable form
def tensorPrint(image):
    x = image.clone().detach().numpy().squeeze()
    x = x.transpose(1,2,0)
    x = x*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    return x

#Calculating the gram matrix, which computes the correlation between features
def gramMatrix(imageFeature):
    _, c, h, w = imageFeature.size()
    imageFeature = imageFeature.view(c, h*w)
    gramMat = torch.mm(imageFeature, imageFeature.t())
    return gramMat

#Target image
target = contentImage.clone().requires_grad_(True)


#Calculating the the activations of style image and content image
styleActivations = modelActivations(styleImage, model)
contentActivations = modelActivations(contentImage, model)


#Weights for the computing styleimage loss
styleWeigth = {"conv1_1" : 1.0, 
               "conv2_1" : 0.8,
               "conv3_1" : 0.4,
               "conv4_1" : 0.2,
               "conv5_1" : 0.1}

#Computing the gram matrix of all the  activations of the style image activations
styleGrams = {layer:gramMatrix(styleActivations[layer]) for layer in styleActivations}

#Weights for content loss and style loss
alpha = 100
beta = 1e8
iterations = 4000

#sess = tf.Session()
#Loading the adam optimizer
optimizer = torch.optim.Adam([target],lr = 0.007)

for i in range(1, iterations+1):
    #Computing the activations of the target image
    targetActivations = modelActivations(target, model)
    #Computing the style loss
    contentLoss = torch.mean((contentActivations['conv4_2'] - targetActivations['conv4_2'])**2)

    styleLoss = 0
    #Comptuting the style loss
    for layer in styleWeigth:
        stylegram = styleGrams[layer]
        targetGrams = targetActivations[layer]
        _, c, h, w = targetGrams.shape
        targetgram = gramMatrix(targetGrams)

        styleLoss += (styleWeigth[layer] * torch.mean((targetgram - stylegram)**2))/(c* h * w)

    #Total loss     
    totalLoss = alpha * contentLoss + beta * styleLoss

            
        
    if i % 50 == 0:
        print("Loss = ", totalLoss)
        
    optimizer.zero_grad()
    totalLoss.backward()
    optimizer.step()

    if i % 250 == 0:
        plt.imshow(tensorPrint(target), label = "Iteration "+str(i))
        plt.show()
        plt.imsave("output/" + str(i)+'.png',tensorPrint(target),format='png')




