
import torch
import cv2
import numpy as np


from torch.utils.data import DataLoader
from torchvision.transforms import transforms




from Dataset.ActionDatasetV3 import ActionDatasetV3


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# configuration
imageHeight = 480
imageWidth = 640
cropHeight = 400
cropWidth = 560
resizeWidth = 160
resizeHeight = 120
epochs = 500
batchSize = 1
frameNumber = 30
true_labels = ["NoGesture", "DoOtherThings", "StopMachine", "ResumeMachine", "PDLC_UP", "PDLC_DOWN", "LightON", "LightOFF"] 
# configuration


transformCollection = []
transformCollection.append(transforms.Resize((imageHeight, imageWidth)))
transformCollection.append(transforms.CenterCrop((cropHeight, cropWidth)))
transformCollection.append(transforms.Resize((resizeHeight, resizeWidth)))
transformCollection.append(transforms.ToTensor()) 
transProcess = transforms.Compose(transformCollection)

dataset = trainDataset = ActionDatasetV3(path="/mnt/d/hvs/Hyvsion_Projects/Actions_project/LabelTool/DataFrames", 
                                         bgPath="/mnt/d/hvs/Hyvsion_Projects/Actions_project/Bg_images",
                                         transform=transProcess, 
                                         frameNumber=frameNumber, 
                                         imageWidth=imageWidth, 
                                         imageHeight=imageHeight,
                                         resizeWidth=resizeWidth,
                                         resizeHeight=resizeHeight,
                                         minTh=(50,50,50), 
                                         maxTh=(100, 255, 255))
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
            img = cv2.resize(img, (500,500))

            cv2.putText(img, text=true_labels[index], org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=4)
            cv2.putText(img, text=str(i), org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=4)
            cv2.imshow('image', img)
            cv2.waitKey(12)

    print('epoch = ', epoch)