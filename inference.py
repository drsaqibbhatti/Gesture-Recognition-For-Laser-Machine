
import torch
import cv2
import time
import numpy as np

from datetime import datetime

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from Model.Mobile3DV2 import Mobile3DV2
from Model.SoftmaxModel import SoftmaxModel


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)



#configuration
labels = ["NoGesture", "DoOtherThings", "StopMachine", "ResumeMachine", "PDLC_UP", "PDLC_DOWN", "LightON", "LightOFF"] 
imageHeight = 480
imageWidth = 640
cropHeight = 400
cropWidth = 560
resizeWidth = 320
resizeHeight = 240
frameNumber = 16
classNum = 8
#configuration


model = Mobile3DV2(num_classes=classNum, width_mult=0.5).to(device)
state_dict = torch.load("D:\hvs\\Hyvsion_Projects\\Actions_project\\action_recognition_git\\actionclassification\\trained\\run_9\\Best_E_36_Acc_0.9938_loss_0.0327_W320_H240_D16_No_DP_State_dict.pt", map_location=device, weights_only=True)
weights = model.state_dict()

model.load_state_dict(state_dict)

weights2 = model.state_dict()
model.eval()

softmax_model = SoftmaxModel(backbone=model).to(device)
softmax_model.eval()

desired_exposure_value=-5
capture = cv2.VideoCapture(0)
#capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # auto mode

transformCollection = []
transformCollection.append(transforms.Resize((imageHeight, imageWidth)))
transformCollection.append(transforms.CenterCrop((cropHeight, cropWidth)))
transformCollection.append(transforms.Resize((resizeHeight, resizeWidth)))
transformCollection.append(transforms.ToTensor()) 
transProcess = transforms.Compose(transformCollection)

tempResult = ""
tempScore = 0.0
accScore = []
frames = []
outputs = []
scoreTh =75
frameCount = 0



start = datetime.now()
while True:
    ret, original = capture.read()
    # capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
    # capture.set(cv2.CAP_PROP_EXPOSURE, desired_exposure_value)

    
    
    frameCount += 1

    # frame = cv2.resize(original, dsize=(resizeWidth, resizeHeight))
    pimImage = Image.fromarray(original.astype('uint8'), 'RGB')
    tensor_image = transProcess(pimImage)
    #tensor_image = tensor_image[[2, 1, 0], :, :]
    if frameCount %2 ==0:
        frames.append(torch.unsqueeze(tensor_image, 0))

    
    if len(frames) == frameNumber:
        tensor_frames = torch.cat(frames)
        tensor_frames = tensor_frames.unsqueeze(0)
        tensor_frames = tensor_frames.to(device)
        tensor_frames = tensor_frames.permute(0, 2, 1, 3, 4) 

        
        frames.pop(0)

        output = softmax_model(tensor_frames)
        output[0][0] = 0
        output[0][1] = 0

        #numpy_output = output.detach().cpu().numpy()

        index = torch.argmax(output).item()
        score = output[0][index].item() * 100

        if  score > scoreTh:
            tempResult = labels[index]
            tempScore = score

        # accScore.append(numpy_output)
        # if len(accScore) > 2:
        #     mean_score = np.mean(accScore, axis=0)
            
        #     accScore.clear()
    

    cv2.putText(original, text=tempResult, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=4)
    cv2.putText(original, str(tempScore), org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=4)

    cv2.imshow("VideoFrame", original)
    cv2.waitKey(1)




    
        
