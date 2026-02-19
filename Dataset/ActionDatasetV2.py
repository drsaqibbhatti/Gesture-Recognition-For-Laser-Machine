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

class ActionDatasetV2(Dataset):
    def __init__(self, path="", transform=None, frameNumber=16, imageWidth=128, imageHeight=128):
        self.path = path
        self.transform = transform
        
        self.labelPath = os.path.join(path, "Labels.json")
        self.frameNumber = frameNumber
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight


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
            if len(frameLabel["Frames"]) < self.frameNumber: continue
            finalFrameLabels.append(frameLabel)
      
        # slidingWindowFrames = []
        # for frameLabel in finalFrameLabels:
        #     labelName = frameLabel["LabelName"]
        #     folderName = frameLabel["FolderName"]

        #     frameMaxCount = len(frameLabel["Frames"])

        #     if frameMaxCount == self.frameNumber:
        #         frames = []
        #         for frameIndex in range(frameMaxCount):
        #             frames.append(frameLabel["Frames"][frameIndex])
                
        #         slidingLabel = {
        #             "LabelName" : labelName,
        #             "FolderName" : folderName,
        #             "Frames" : frames
        #         }
        #         slidingWindowFrames.append(slidingLabel)
        #         continue



        #     for startIndex in range(0, frameMaxCount - self.frameNumber):
        #         frames = []
        #         for frameIndex in range(startIndex, startIndex + self.frameNumber):
        #             frames.append(frameLabel["Frames"][frameIndex])
                
        #         slidingLabel = {
        #             "LabelName" : labelName,
        #             "FolderName" : folderName,
        #             "Frames" : frames
        #         }
        #         slidingWindowFrames.append(slidingLabel)

        
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
            imagePath = os.path.join(self.path, folderName, str(fileName) + ".jpg")
            image = Image.open(imagePath).convert('RGB')

            if prob_mirror > 0.5 and className != "LightON" and className != "LightOFF" and className != "PDLC_UP" and className != "PDLC_DOWN":
                image =  ImageOps.mirror(image)

            torch_image = self.transform(image)
            ############################################
            torch_image = torch_image[[2, 1, 0], :, :]
            #############################################
            tensorImages.append(torch_image)

            
        videoFrames = torch.zeros(self.frameNumber, 3, self.imageHeight, self.imageWidth)
        for j in range(len(tensorImages)):
            videoFrames[j] = tensorImages[j]

        videoFrames = videoFrames.permute(1, 0, 2, 3)
        return videoFrames, classID
