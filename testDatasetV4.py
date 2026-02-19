
import torch
import cv2
import numpy as np


from torch.utils.data import DataLoader
from torchvision.transforms import transforms




from Dataset.ActionDatasetV4 import ActionDatasetV4


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

dataset = trainDataset = ActionDatasetV4(path="D:/hvs//Hyvsion_Projects//Actions_project//Raw_test//ToF_cam", 
                                         transform=transProcess, 
                                         frameNumber=frameNumber, 
                                         imageWidth=imageWidth, 
                                         imageHeight=imageHeight,
                                         resizeWidth=resizeWidth,
                                         resizeHeight=resizeHeight)
dataLoader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)

totalBatches = len(dataLoader)
print('total batch = ', totalBatches)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataLoader):
        gpu_inputs = inputs.to(device)
        gpu_labels = labels.to(device)
        index = gpu_labels[0].item()
        sequence = gpu_inputs[0].permute(1, 0, 2, 3)  # [30, 1, H, W]

        for frameIndex in range(frameNumber):
            frame = sequence[frameIndex].squeeze(0)  # [H, W]
            numpy_frame = frame.detach().cpu().numpy() * 255.0
            img = numpy_frame.astype(np.uint8)
            img = cv2.resize(img, (500, 500))

            # Normalize for display (before adding text)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

            # Add text after normalization
            cv2.putText(img, text=true_labels[index], org=(30, 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255), thickness=2)
            cv2.putText(img, text=f"Frame {frameIndex}", org=(30, 90),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(255), thickness=2)

            print("Image shape:", img.shape, "Min:", img.min(), "Max:", img.max())

            cv2.imshow("ToF Sequence Viewer", img)
            if cv2.waitKey(12) == 27:
                break


    print(f"Epoch {epoch + 1}/{epochs}")