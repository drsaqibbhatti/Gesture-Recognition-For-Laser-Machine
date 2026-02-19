
from torchvision.transforms import transforms
from PIL import Image




from Dataset.ActionDatasetV2 import ActionDatasetV2


transformCollection = []
transformCollection.append(transforms.Resize((128,128), interpolation=Image.NEAREST)) 
transformCollection.append(transforms.ToTensor()) 
transProcess = transforms.Compose(transformCollection)



trainDataset = ActionDatasetV2(path="/home/saqib/Projects/action_recognition/ToF_DataFrames//DataFrames_upto_250425", transform=transProcess, frameNumber=16, imageWidth=640, imageHeight=480)
trainDataset.summary()
