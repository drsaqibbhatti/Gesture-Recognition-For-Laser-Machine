
import torch
import cv2
import numpy as np


from torch.utils.data import DataLoader
from torchvision.transforms import transforms




from Dataset.ActionDatasetV2 import ActionDatasetV2


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# configuration
imageHeight = 96
imageWidth = 96
epochs = 1
batchSize = 15
frameNumber = 30
true_labels = ["NoGesture", "DoOtherThings", "StopMachine", "ResumeMachine", "PDLC_UP", "PDLC_DOWN", "LightON", "LightOFF"] 
# configuration


transformCollection = []
transformCollection.append(transforms.CenterCrop((imageHeight, imageWidth)))
transformCollection.append(transforms.ToTensor()) 
transProcess = transforms.Compose(transformCollection)

dataset = trainDataset = ActionDatasetV2(path="C:\\ProgramData\\ActionLabelMaker\\DataFrames\\", transform=transProcess, frameNumber=frameNumber, imageWidth=imageWidth, imageHeight=imageHeight)
dataLoader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

totalBatches = len(dataLoader)
print('total batch = ', totalBatches)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataLoader):
        gpu_inputs = inputs.to(device)
        gpu_labels = labels.to(device)

        index = gpu_labels[0].item()

        for frameIndex in range(0, frameNumber):
            frame = gpu_inputs[0].permute(1, 0, 2, 3)
            frame = frame[frameIndex].permute(1, 2, 0)
            numpy_frame = frame.detach().cpu().numpy()* 255
            img = numpy_frame.astype(np.uint8).copy() 
            img = cv2.resize(img, (900,900))

            cv2.putText(img, text=true_labels[index], org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=4)
            cv2.putText(img, text=str(i), org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=4)
            cv2.imshow('image', img)
            cv2.waitKey(60)

    print('epoch = ', epoch)