import torch
import torch.nn as nn
from feat.utils import euclidean_metric

class ProtoNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model_type == 'ConvNet':
            from feat.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.model_type == 'ResNet':
            from feat.networks.resnet import ResNet
            self.encoder = ResNet()
        elif args.model_type == 'AmdimNet':
            from feat.networks.amdimnet import AmdimNet
            self.encoder = AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
        else:
            raise ValueError('')

    # n_nclass
    def forward(self, data_shot, data_query, class_lens=None):
        proto = self.encoder(data_shot) 
        if class_lens: 
            indices = torch.cumsum(class_lens) 
            indices = torch.cat([torch.tensor[0],indices])
            proto = torch([ torch.mean(proto[indices[i]:indices[i+1]]) for i in range(len(indices)-1)])
        else: proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        logits = euclidean_metric(self.encoder(data_query), proto) / self.args.temperature # distance here is negative --> similarity 
        return logits