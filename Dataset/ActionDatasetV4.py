import torch
import numpy as np
import os
import json
import random
import cv2
import json
import random

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageOps
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class ActionDatasetV4(Dataset):
    def __init__(self, 
                 path="", 
                 transform=None, 
                 frameNumber=16, 
                 imageWidth=128, 
                 imageHeight=128,
                 resizeWidth=128,
                 resizeHeight=128,
                 minThreshold= 200,
                 maxThreshold= 2000
                 ):
        
        self.path = path
        self.transform = transform
        self.labelPath = os.path.join(path, "Labels.json")
        self.frameNumber = frameNumber
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.resizeWidth = resizeWidth
        self.resizeHeight = resizeHeight
        self.minThreshold = minThreshold
        self.maxThreshold = maxThreshold


        with open(self.labelPath) as f:
            labelsInfo = json.load(f)

        self.classNames = []
        for label in labelsInfo:
            name = label["Name"]
            self.classNames.append(name)

        frameGroup = []
        for jsonFile in os.listdir(path):
            if jsonFile == "Labels.json" : continue
            if jsonFile.endswith(".json") != True : continue

            jsonFilePath = os.path.join(path, jsonFile)
            

            with open(jsonFilePath) as f:
                frameJson = json.load(f)

            keyName = os.path.splitext(jsonFile)[0]
            frames = []
            for frame in frameJson["Frames"]:
                labelName = frame["LabelName"]
                folderName = frame["FolderName"]
                frameIndex = frame["Index"]
                frames.append((labelName, frameIndex))

            frameInfo = {"name":keyName, "frames":frames}
            frameGroup.append(frameInfo)



        frameLabels = []
        for frameInfo in frameGroup:
            keyName = frameInfo["name"]
       
            frames = frameInfo["frames"]

            frameIndexTable = []
            tempLabelName = frames[0][0]  #Name
            for frame in frames:
                labelName = frame[0]
                frameIndex = frame[1]
                frameIndexTable.append(frameIndex)

                if tempLabelName != labelName:
                    frameLabel = {
                        "LabelName" : labelName,
                        "FolderName" : keyName,
                        "Frames" : frameIndexTable
                    }

                    frameLabels.append(frameLabel)
                    frameIndexTable = []
                    tempLabelName = labelName
                
            frameLabel = {
                "LabelName" : tempLabelName,
                "FolderName" : keyName,
                "Frames" : frameIndexTable
            }
            frameLabels.append(frameLabel)

            
        finalFrameLabels = []
        for frameLabel in frameLabels:
            #if len(frameLabel["Frames"]) < self.frameNumber: continue
            finalFrameLabels.append(frameLabel)
      
        self.frameLabelsInfo = finalFrameLabels
        random.shuffle(self.frameLabelsInfo)

    def summary(self):
        for className in self.classNames:
            count = 0
            for label in self.frameLabelsInfo:
                if label["LabelName"] == className:
                    count += 1

            print('className = ', className, " count=", count)


    def __len__(self):
        return len(self.frameLabelsInfo)
    
    def __getitem__(self, index):

        className = self.frameLabelsInfo[index]["LabelName"]
        folderName = self.frameLabelsInfo[index]["FolderName"]
        frames = self.frameLabelsInfo[index]["Frames"]
        classID = self.classNames.index(className)



        originFrameCount = len(frames)
        unitPriod = originFrameCount / self.frameNumber


        prob_mirror = random.random()
        tensorImages = []
        for i in range(0, self.frameNumber):
            frameIndex = int(round(unitPriod * i))
            if frameIndex >= originFrameCount:
                frameIndex = originFrameCount - 1

            fileName = frames[frameIndex]
            imagePath = os.path.join(self.path, folderName, str(fileName) + ".raw")


            ################################## Raw ############
            with open(imagePath, "rb") as f:
                raw_data = np.frombuffer(f.read(), dtype=np.uint16)  # Read the raw bytes into an array

            # Expected size for the raw depth data (16-bit depth = 2 bytes per pixel)
            expected_size = self.imageWidth * self.imageHeight
            
            if raw_data.size != expected_size:
                raise ValueError(f"Unexpected raw data size: {raw_data.size}, expected: {expected_size}")

            # Reshape the data into the correct dimensions (height, width)
            raw_image = raw_data.reshape((self.imageHeight, self.imageWidth)).astype(np.float32)
            #raw_image = np.uint8(raw_image / 256) #converting to 8-bit image
            
            # Apply clipping: set values outside [200, 2000] to 2000
            raw_image[(raw_image < self.minThreshold) | (raw_image > self.maxThreshold)] = self.maxThreshold
            ######################################################
    


            #image = Image.fromarray(raw_image, mode='L') # For 8 bit image
            #image = Image.fromarray(raw_image.astype(np.uint16), mode='I;16') # For 16-bit image (PIL cannot work with 16-bit. Saqib)
            image = Image.fromarray(raw_image.astype(np.float32))  # PIL auto-sets mode='F'


            if prob_mirror > 0.5:
                image =  ImageOps.mirror(image)

            torch_image = self.transform(image)

            tensorImages.append(torch_image)

            
        videoFrames = torch.zeros(self.frameNumber, 1, self.resizeHeight, self.resizeWidth)
        for j in range(len(tensorImages)):
            videoFrames[j] = tensorImages[j]

        videoFrames = videoFrames.permute(1, 0, 2, 3)
        return videoFrames, classID
