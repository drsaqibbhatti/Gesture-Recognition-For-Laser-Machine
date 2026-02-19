
import torch
import cv2
import time


from torch.utils.data import DataLoader
from torchvision.transforms import transforms



from Model.Mobile3DV2 import Mobile3DV2



USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


model = Mobile3DV2(num_classes=8).to(device)
model.eval()
dummy = torch.rand([1, 3, 30, 147, 196]).to(device)

start = time.time()
fps = 0
while True:
    out = model(dummy)
    fps += 1
    end = time.time()
    if (end - start) >= 1.0:
        start = time.time()
        print('fps = ', fps)
        fps = 0





    
        
