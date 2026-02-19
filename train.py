
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from Dataset.ActionDatasetV3 import ActionDatasetV3
from Model.Mobile3DV2 import Mobile3DV2
from torch.nn import DataParallel

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

learningRate = 0.04
momentum=0.9
weight_decay=1e-3
dampening=0.9
epochs = 100
classNum = 8
targetAcc = 0.999
batchSize = 271
savePath = "/home/saqib/Projects/action_recognition/action_recognition_git/actionclassification/trained/actionV2.pt"
preLoad = False
frameNumber = 30


# configuration


transformCollection = []
transformCollection.append(transforms.Resize((imageHeight, imageWidth)))
transformCollection.append(transforms.CenterCrop((cropHeight, cropWidth)))
transformCollection.append(transforms.Resize((resizeHeight, resizeWidth)))
transformCollection.append(transforms.ToTensor()) 
transProcess = transforms.Compose(transformCollection)



dataset = trainDataset = ActionDatasetV3(path="/home/saqib/Projects/action_recognition/DataFrames", 
                                         bgPath="/home/saqib/Projects/action_recognition/Bg_images",
                                         transform=transProcess, 
                                         frameNumber=frameNumber, 
                                         imageWidth=imageWidth, 
                                         imageHeight=imageHeight,
                                         resizeWidth=resizeWidth,
                                         resizeHeight=resizeHeight,
                                         minTh=(50,50,50), 
                                         maxTh=(100, 255, 255))
tarinLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True, drop_last=True)



model = Mobile3DV2(num_classes=classNum, width_mult=0.5).to(device)



if preLoad == True:
    state_dict = torch.load("E:\\Github\\actionclassification\\trained\\actionV1.pt", map_location=device, weights_only=True)
    weights = model.state_dict()
    model.load_state_dict(state_dict)

crossEntropy = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weight_decay, dampening=dampening)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-3)
totalBatches = len(tarinLoader)
print('total batch = ', totalBatches)

trainDataset.summary()

max_acc = 0

for epoch in range(epochs):
    avg_loss = 0
    avg_acc = 0
    
    print(f"Epoch {epoch + 1}/{epochs}")
    
    for i, (inputs, labels) in enumerate(tqdm(tarinLoader, desc="Training")):
        gpu_inputs = inputs.to(device)
        gpu_labels = labels.to(device)

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
        correct = (predicted == gpu_labels.long().cuda(0)).sum()
        total = labels.size(0)
        avg_acc += ((correct / total).item() / totalBatches)
    print('current learning rate=', scheduler.get_last_lr())
    scheduler.step(avg_loss)
    if max_acc < avg_acc:
        max_acc = avg_acc
        model.eval()
        torch.save(model.state_dict(), savePath)


    print('epoch = ', epoch, ', acc=', avg_acc, ', loss=', avg_loss)
    if targetAcc <= avg_acc:
        print('training complete')
        break


model.eval()
torch.save(model.state_dict(), savePath)
