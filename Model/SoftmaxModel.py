

import torch



class SoftmaxModel(torch.nn.Module):
    def __init__(self, backbone):
        super(SoftmaxModel, self).__init__()
        self.backbone = backbone

        
    def forward(self, x):
        output = self.backbone(x)
        output = torch.softmax(output, dim=1)
        return output