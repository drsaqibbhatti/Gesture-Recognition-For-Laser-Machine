
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from Dataset.ActionDatasetV4_FolderName import ActionDatasetV4_FolderName
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
USE_CUDA = torch.cuda.is_available() 
device = torch.device("cuda" if USE_CUDA else "cpu") 
print("Device:", device)


# configuration
imageHeight = 480
imageWidth = 640
cropHeight = 400
cropWidth = 560
resizeWidth = 320
resizeHeight = 240
bgMinThreshold= 200
bgMaxThreshold = 2000
learningRate = 0.001
momentum=0.9
weight_decay=1e-3
dampening=0.9
epochs = 10
classNum = 8
targetAcc = 0.999

frameNumber = 16
base_folder= "D:/hvs//Hyvsion_Projects//Actions_project//Sanity_check//12_DataFrames_250514"

preLoad = False



# configuration


transformCollection = []
transformCollection.append(transforms.Resize((imageHeight, imageWidth)))
transformCollection.append(transforms.CenterCrop((cropHeight, cropWidth)))
transformCollection.append(transforms.Resize((resizeHeight, resizeWidth)))
transformCollection.append(transforms.Lambda(lambda x: torch.from_numpy(np.array(x).astype(np.float32) / 4095.0).unsqueeze(0)))
transProcess = transforms.Compose(transformCollection)



dataset = trainDataset = ActionDatasetV4_FolderName(path="D:/hvs//Hyvsion_Projects//Actions_project//ToF_DataFrames//12_DataFrames_250514",
                                         transform=transProcess, 
                                         frameNumber=frameNumber, 
                                         imageWidth=imageWidth, 
                                         imageHeight=imageHeight,
                                         resizeWidth=resizeWidth,
                                         resizeHeight=resizeHeight,
                                         minThreshold=bgMinThreshold)
tarinLoader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)


for i, (inputs, labels, metas) in enumerate(tqdm(tarinLoader, desc="Saving Sanity Frames")):
    batch_size = inputs.shape[0]
    for sample_idx in range(batch_size):
        sequence = inputs[sample_idx, 0]  # [T, H, W]

        class_id = labels[sample_idx].item()
        label_name = dataset.classNames[class_id]
    
        # Get the original folder name from meta info
        folder_name = metas[sample_idx]

        # Save path
        save_dir = os.path.join(base_folder, f"minTh{bgMinThreshold}_maxTh{bgMaxThreshold}", label_name, f"{folder_name}_sanity")

        os.makedirs(save_dir, exist_ok=True)

        for frameIndex in range(sequence.shape[0]):
            frame_np = sequence[frameIndex].cpu().numpy() * 4095
            frame_disp = cv2.normalize(frame_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            save_path = os.path.join(save_dir, f"{frameIndex + 1}.png")
            cv2.imwrite(save_path, frame_disp)


