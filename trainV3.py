
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode

import numpy as np
from Dataset.ActionDatasetV4 import ActionDatasetV4
from Model.Mobile3DV2 import Mobile3DV2
from torch.nn import DataParallel
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("Training on:", device)


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
epochs = 200
classNum = 8
targetAcc = 0.99999
batchSize = 160
frameNumber = 16
base_folder= "/home/saqib/Projects/action_recognition/action_recognition_git/actionclassification/trained/ToF_Cam"
# Create unique run folder
run_id = 1
while os.path.exists(os.path.join(base_folder, f"run_{run_id}")):
    run_id += 1
run_folder = os.path.join(base_folder, f"run_{run_id}")
os.makedirs(run_folder)


preLoad = False



# configuration


transformCollection = []
transformCollection.append(transforms.Resize((imageHeight, imageWidth), interpolation=InterpolationMode.BICUBIC))
transformCollection.append(transforms.CenterCrop((cropHeight, cropWidth)))
transformCollection.append(transforms.Resize((resizeHeight, resizeWidth), interpolation=InterpolationMode.BICUBIC))
# transformCollection.append(transforms.RandomAutocontrast(p=0.4))
#transformCollection.append(transforms.ToTensor()) 
transformCollection.append(transforms.Lambda(lambda x: torch.from_numpy(np.array(x).astype(np.float32) / 4095.0).unsqueeze(0)))
transProcess = transforms.Compose(transformCollection)



dataset = trainDataset = ActionDatasetV4(path="/home/saqib/Projects/action_recognition/ToF_DataFrames//DataFrames_upto_250512",
                                         transform=transProcess, 
                                         frameNumber=frameNumber, 
                                         imageWidth=imageWidth, 
                                         imageHeight=imageHeight,
                                         resizeWidth=resizeWidth,
                                         resizeHeight=resizeHeight,
                                         minThreshold=bgMinThreshold,
                                         maxThreshold=bgMaxThreshold)
tarinLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True, drop_last=True)



model = Mobile3DV2(num_classes=classNum, width_mult=0.5)


######################### Using Multiple GPUs ############################
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DataParallel(model)

model = model.to(device)  
###########################################################################

if preLoad == True:
    state_dict = torch.load("/home/saqib/Projects/action_recognition/action_recognition_git/actionclassification/trained/run_8_W320_H240_D16/Epoch_7_Acc_0.6853_loss_0.8452_best.pt", map_location=device, weights_only=True)
    weights = model.state_dict()
    model.load_state_dict(state_dict)

crossEntropy = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RAdam(params=model.parameters(), lr=learningRate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-3)
totalBatches = len(tarinLoader)
print('total batch = ', totalBatches)

trainDataset.summary()

for i, (inputs, labels) in enumerate(tarinLoader):
    print("Tensor shape:", inputs.shape)   # [B, 1, T, H, W]
    print("Tensor dtype:", inputs.dtype)   # torch.float32
    print("Tensor min/max:", inputs.min().item(), inputs.max().item())
    break


max_acc = 0
min_loss= float('inf')
metrics=[]
for epoch in range(epochs):
    avg_loss = 0
    avg_acc = 0
    
    print(f"Epoch {epoch + 1}/{epochs}")
    
    for i, (inputs, labels) in enumerate(tqdm(tarinLoader, desc="Training")):
        gpu_inputs = inputs.to(device)
        gpu_labels = labels.to(device)


        # ###############Visualize all frames in the batch (all or first sample only)##################
        # with torch.no_grad():
        #     # sequence = inputs[0, 0]  # first sample only
        #     batch_size = inputs.shape[0]
        #     for sample_idx in range(batch_size):
        #         sequence = inputs[sample_idx, 0]  # shape: [T, H, W]
        #         for frameIndex in range(sequence.shape[0]):
        #             frame_np = sequence[frameIndex].cpu().numpy() * 4095  # Denormalize
        #             frame_disp = cv2.normalize(frame_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        #             cv2.putText(frame_disp, f"Batch {i}, Frame {frameIndex}", (20, 30),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120), 2)
        #             cv2.imshow("Training Frames", frame_disp)
        #             key = cv2.waitKey(10)
        #             if key == 27:  # ESC to break
        #                 break
        # ##############################################################################################
        
        optimizer.zero_grad()
        model.train()
        output = model(gpu_inputs)
        loss = crossEntropy(output, gpu_labels.long())
        loss.backward()

        optimizer.step()
        avg_loss += (loss.item() / totalBatches)

    for i, (inputs, labels) in enumerate(tqdm(tarinLoader, desc="Evaluation")):
        gpu_inputs = inputs.to(device)
        gpu_labels = labels.to(device)

        model.eval()
        output = model(gpu_inputs)

        _, predicted = torch.max(output.data, 1)
        correct = (predicted == gpu_labels.long().to(device)).sum()
        total = labels.size(0)
        avg_acc += ((correct / total).item() / totalBatches)
    #print('current learning rate=', scheduler.get_last_lr())
    # scheduler.step(avg_loss)



    savePath_last = os.path.join(run_folder, f"Last_E_{epoch}_Acc_{avg_acc:.4f}_loss_{avg_loss:.4f}_W{resizeWidth}_H{resizeHeight}_D{frameNumber}.pt")
    savePath_best = os.path.join(run_folder, f"Best_E_{epoch}_Acc_{avg_acc:.4f}_loss_{avg_loss:.4f}_W{resizeWidth}_H{resizeHeight}_D{frameNumber}.pt")

    # if max_acc < avg_acc:
    #     max_acc = avg_acc
    if avg_loss < min_loss:
        min_loss = avg_loss    
        model.eval()
        torch.save(model.state_dict(), savePath_best)


    print('epoch = ', epoch, ', acc=', avg_acc, ', loss=', avg_loss)




    # Log metrics
    metrics.append({
        'Epoch': epoch + 1,
        'Train Loss': avg_loss,
        'Train Accuracy': avg_acc,
    })
    

    # Save metrics and plots every epoch
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(run_folder, "metrics.csv"), index=False)

    plt.figure()
    plt.plot(metrics_df['Epoch'], metrics_df['Train Accuracy'], label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(run_folder, f"accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(metrics_df['Epoch'], metrics_df['Train Loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(run_folder, f"loss.png"))
    plt.close()

    if targetAcc <= avg_acc:
        print('training complete')
        break

model.eval()
torch.save(model.state_dict(), savePath_last)
