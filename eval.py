
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import transforms




from Dataset.ActionDatasetV2 import ActionDatasetV2
from Model.Mobile3DV2 import Mobile3DV2


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


# configuration
imageHeight = 147
imageWidth = 196
learningRate = 0.001
epochs = 100
classNum = 8
targetAcc = 0.99
batchSize = 1
savePath = "E:\\Github\\actionclassification\\trained\\actionV1.pt"
preLoad = True
frameNumber = 18
weightDecay = 0.001
# configuration


transformCollection = []
transformCollection.append(transforms.Resize((imageHeight, imageWidth)))
transformCollection.append(transforms.ToTensor()) 
transProcess = transforms.Compose(transformCollection)



trainDataset = ActionDatasetV2(path="C:\\ProgramData\\ActionLabelMaker\\DataFrames\\", transform=transProcess, frameNumber=frameNumber, imageWidth=imageWidth, imageHeight=imageHeight)
tarinLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True)



model = Mobile3DV2(num_classes=classNum, width_mult=0.5).to(device)
if preLoad == True:
    state_dict = torch.load("E:\\Github\\actionclassification\\trained\\actionV1.pt", map_location=device, weights_only=True)
    weights = model.state_dict()
    model.load_state_dict(state_dict)


totalBatches = len(tarinLoader)
print('total batch = ', totalBatches)

trainDataset.summary()


avg_acc = 0
for i, (inputs, labels) in enumerate(tarinLoader):
    gpu_inputs = inputs.to(device)
    gpu_labels = labels.to(device)


    model.eval()
    output = model(gpu_inputs)


    _, predicted = torch.max(output.data, 1)
    correct = (predicted == gpu_labels.long().cuda(0)).sum()
    total = labels.size(0)
    cur_acc = (correct / total).item()
    avg_acc += (cur_acc / totalBatches)
    print("current accuracy=", cur_acc)

print("accuracy = ", avg_acc)


