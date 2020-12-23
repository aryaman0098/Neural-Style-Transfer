import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import vgg19
from IPython.display import display,clear_output
import math

VGG_BIASES = vgg19.preprocess_input((np.zeros((3))).astype("float32"))

# Defining the layers to be used for the style transfer
CONTENT_LAYER = ["block5_conv2"]
STYLE_LAYER = ["block1_conv1","block2_conv2","block3_conv3","block4_conv4"]


# Function for displaying the VGG19 model
def displayVGG19():
    model = vgg19.VGG19(include_top=False,weights="imagenet")
    model.summary()


# Function that converts the images suitable for VGG19 model
def processImages(contentImage, styleImage):
    processedContentImage = vgg19.preprocess_input(np.expand_dims(contentImage, axis = 0))
    processedStyleImage = vgg19.preprocess_input(np.expand_dims(styleImage, axis = 0))

    return processedContentImage, processedStyleImage


# Function that converts the images suitable for VGG19 model for multple style images transfer
def processImagesMultiple(contentImage, styleImage1, styleImage2):
    processedContentImage = vgg19.preprocess_input(np.expand_dims(contentImage, axis = 0))
    processedStyleImage1 = vgg19.preprocess_input(np.expand_dims(styleImage1, axis = 0))
    processedStyleImage2 = vgg19.preprocess_input(np.expand_dims(styleImage2, axis = 0))

    return processedContentImage, processedStyleImage1, processedStyleImage2


    
# Function that creates a VGG19 model and return the output of the above defined CONTENT_LAYER and SYLE_LAYER
def makeModel(CONTENT_LAYER, STYLE_LAYER):
    baseModel=vgg19.VGG19(include_top=False,weights="imagenet")
    baseModel.trainable = False
    contentLayers = CONTENT_LAYER
    styleLayers = STYLE_LAYER
    outputLayers = [baseModel.get_layer(layer).output for layer in (contentLayers + styleLayers)]
    return tf.keras.models.Model(baseModel.input,outputLayers)


# Deprocess the images from BGR to RGB
def deprocess(processedImg):
    unprocessedImg = processedImg - VGG_BIASES
    unprocessedImg = tf.unstack(unprocessedImg,axis=-1)
    unprocessedImg = tf.stack([unprocessedImg[2],unprocessedImg[1],unprocessedImg[0]],axis=-1)
    return unprocessedImg


# Function that return the content loss
def getContentLoss(newImageContent, baseImageContent):
    return np.mean(np.square(newImageContent - baseImageContent))


# Function that returns the gram matrix
def getGramMatrix(output):
    firstStyleLayer = output
    A = tf.reshape(firstStyleLayer,(-1,firstStyleLayer.shape[-1])) 
    n = A.shape[0]
    #taking the product of the transpose of the matrix and itself
    gramMatrix = tf.matmul(A,A,transpose_a=True)
    n=gramMatrix.shape[0]
    return gramMatrix/tf.cast(n,"float32"), n


# Function for style loss
def getStyleLoss(newImageStyle, baseStyle):
    newStyleGram, gramNumHt1 = getGramMatrix(newImageStyle)
    baseStyleGram, gramNumHt2 = getGramMatrix(baseStyle)
    assert gramNumHt1 == gramNumHt2
    loss = tf.reduce_sum(tf.square((baseStyleGram - newStyleGram))/(4*(gramNumHt1**2)*(gramNumHt2**2)))
    return loss



# Function that uses the content loss and style loss to generate the total loss (alpha and beta are hyper parameters)
def getTotalLoss(newImageOutput, baseContentImageOuput, baseStyleImageOuput, alpha = 0.0001, beta = .9999):
    newImageStyles = newImageOutput[len(CONTENT_LAYER):]
    baseImageStyles = baseStyleImageOuput[len(CONTENT_LAYER):]
    styleLoss = 0
    n = len(newImageStyles)
    for i in range(n):
        styleLoss += getStyleLoss(newImageStyles[i], baseImageStyles[i])
        
    newImageContents = newImageOutput[:len(CONTENT_LAYER)]
    baseImageContents = baseContentImageOuput[:len(CONTENT_LAYER)]
    contentLoss = 0
    n = len(newImageContents)
    for i in range(n):
        contentLoss += getContentLoss(newImageContents[i], baseImageContents[i]) / n
    
    return alpha * contentLoss + beta * styleLoss


# Using the adam optimizer
optimizer = tf.optimizers.Adam(5,beta_1=.99,epsilon=1e-3)


# Driver function for Neural Style Transfer
def neuralStyleTransfer(processedContentVar, baseContentOutput, baseStyleOutputs, numIterations, baseModel):
    images = []
    losses =[]
    i=0
    bestLoss = math.inf
    minVals = VGG_BIASES
    maxVals = 255 + VGG_BIASES
    for i in range(numIterations):   
        with tf.GradientTape() as tape:
            tape.watch(processedContentVar)
            contentVarOutputs = baseModel(processedContentVar)
            loss = getTotalLoss(contentVarOutputs, baseContentOutput, baseStyleOutputs)
            grad = tape.gradient(loss, processedContentVar)
            losses.append(loss)
            optimizer.apply_gradients(zip([grad],[processedContentVar]))
            clipped = tf.clip_by_value(processedContentVar, minVals, maxVals)
            processedContentVar.assign(clipped)
            if i % 5 == 0:
                images.append(deprocess(processedContentVar))
            if loss < bestLoss:
                bestImage = processedContentVar
                bestLoss = loss
            display(i)
            display(loss)
            clear_output(wait=True)
    return images, losses, bestImage



# Fucntion that computes the total loss for multiple style image transfer
def getTotalLossForMultipleImages(newImageOutput, baseContentImageOuput, baseStyleImageOuput1, baseStyleImageOuput2, beta1, beta2, alpha = 0.0001):
    newImageStyles = newImageOutput[len(CONTENT_LAYER):]
    baseImageStyles1 = baseStyleImageOuput1[len(CONTENT_LAYER):]
    baseImageStyles2 = baseStyleImageOuput2[len(CONTENT_LAYER):]
    styleLoss1 = 0
    n = len(newImageStyles)
    for i in range(n):
        styleLoss1 += getStyleLoss(newImageStyles[i], baseImageStyles1[i])
    styleLoss2 = 0
    for i in range(n):
        styleLoss2 += getStyleLoss(newImageStyles[i], baseImageStyles2[i])
    
    newImageContents = newImageOutput[:len(CONTENT_LAYER)]
    baseImageContents = baseContentImageOuput[:len(CONTENT_LAYER)]
    contentLoss = 0
    n = len(newImageContents)
    for i in range(n):
        contentLoss += getContentLoss(newImageContents[i], baseImageContents[i]) / n
    
    return alpha * contentLoss + beta1 * styleLoss1 + beta2 * styleLoss2



# Driver function for multiple style image Neural Style Transfer
def neuralStyleTransferForMultipleImages(processedContentVar, baseContentOutput, baseStyleOutputs1, baseStyleOutputs2, numIterations, baseModel, beta1, beta2):
    images = []
    losses =[]
    i=0
    bestLoss = math.inf
    minVals = VGG_BIASES
    maxVals = 255 + VGG_BIASES
    for i in range(numIterations):   
        with tf.GradientTape() as tape:
            tape.watch(processedContentVar)
            contentVarOutputs = baseModel(processedContentVar)
            loss = getTotalLossForMultipleImages(contentVarOutputs, baseContentOutput, baseStyleOutputs1, baseStyleOutputs2, beta1, beta2)
            grad = tape.gradient(loss, processedContentVar)
            losses.append(loss)
            optimizer.apply_gradients(zip([grad],[processedContentVar]))
            clipped = tf.clip_by_value(processedContentVar, minVals, maxVals)
            processedContentVar.assign(clipped)
            if i % 2 == 0:
                images.append(deprocess(processedContentVar))
            if loss < bestLoss:
                bestImage = processedContentVar
                bestLoss = loss
            display(i)
            display(loss)
            clear_output(wait=True)
    return images, losses, bestImage



