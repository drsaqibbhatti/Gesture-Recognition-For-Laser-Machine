

import torch



class ActionClassifier(torch.nn.Module):
    def __init__(self, classNum=6):
        super(ActionClassifier, self).__init__()
        self.classNum = classNum

        self.layers = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=64),
            torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            torch.nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=128),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),


            torch.nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=256),
            torch.nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=256),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),



            torch.nn.Conv3d(256, 512, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=512),
            torch.nn.Conv3d(512, 512, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=512),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),


            
            torch.nn.Conv3d(512, 512, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=512),
            torch.nn.Conv3d(512, 512, kernel_size=3, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=512),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.fn = torch.nn.Sequential(
            torch.nn.Linear(4608, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, self.classNum),
        )



    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.fn(out)
        return out