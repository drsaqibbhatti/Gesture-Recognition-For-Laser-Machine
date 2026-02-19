import torch
import numpy as np
import os
import json
import random
import cv2

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageOps
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class ActionDataset(Dataset):
    def __init__(self, path="", transform=None):
        self.path = path
        self.transform = transform

        self.datasetPair = []
        self.classNames = os.listdir(path)
        for className in self.classNames:
            classPath = os.path.join(path, className)
            videoFiles = os.listdir(classPath)
            for videoFile in videoFiles:
                videoFilePath = os.path.join(classPath, videoFile)
                self.datasetPair.append((className, videoFilePath))
        
    def __len__(self):
        return len(self.datasetPair)
    
    def __getitem__(self, index):

        className = self.datasetPair[index][0]
        classID = self.classNames.index(className)

        videoFilePath = self.datasetPair[index][1]

        tensorImages = []
        for i in range(16):
            imagePath = os.path.join(videoFilePath, str(i) + ".jpg")
            image = Image.open(imagePath).convert('RGB')
            torch_image = self.transform(image)
            ############################################
            torch_image = torch_image[[2, 1, 0], :, :]
            #############################################
            tensorImages.append(torch_image)

            
        videoFrames = torch.zeros(16, 3, 128, 171)
        for j in range(len(tensorImages)):
            videoFrames[j] = tensorImages[j]

        videoFrames = videoFrames.permute(1, 0, 2, 3)
        return videoFrames, classID

        
