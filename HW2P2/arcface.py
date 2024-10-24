import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class ArcFace(nn.Module):
#     def __init__(self, cin, cout, s=8.0, m=0.5, easy_margin=False):
#         super().__init__()
#         self.s = s
#         self.m = m
#         self.easy_margin = easy_margin
#         self.cout = cout
#         self.fc = nn.Linear(cin, cout, bias=False)
#         nn.init.xavier_uniform_(self.fc.weight)

#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, x, label=None):
#         # Normalize features and weights
#         x_norm = F.normalize(x, p=2, dim=1)
#         w_norm = F.normalize(self.fc.weight, p=2, dim=1)
#         cos = F.linear(x_norm, w_norm)  # Cosine similarity

#         if label is not None:
#             # Convert label to one-hot
#             one_hot = F.one_hot(label, num_classes=self.cout).float()

#             # Compute sine of theta
#             sin = torch.sqrt(1.0 - torch.pow(cos, 2) + 1e-6)

#             # Compute phi = cos(theta + m)
#             phi = cos * self.cos_m - sin * self.sin_m

#             if self.easy_margin:
#                 phi = torch.where(cos > 0, phi, cos)
#             else:
#                 phi = torch.where(cos > self.th, phi, cos - self.mm)

#             # Combine target and non-target logits
#             output = (one_hot * phi) + ((1.0 - one_hot) * cos)
#             output *= self.s
#             return output
#         else:
#             return cos * self.s


# class ArcFaceLoss(nn.Module):
#     def __init__(self, embedding_size, num_classes, s=30.0, m=0.50):
#         super(ArcFaceLoss, self).__init__()
#         self.s = s
#         self.m = m
#         self.W = nn.Parameter(torch.randn(embedding_size, num_classes))
#         nn.init.xavier_uniform_(self.W)

#     def forward(self, x, labels):
#         x = F.normalize(x, p=2, dim=1)  # 确保特征向量归一化
#         W = F.normalize(self.W, p=2, dim=0)  # 确保类别权重归一化
#         cosine = torch.matmul(x, W)

#         theta = torch.acos(cosine.clamp(-1.0, 1.0))  # 计算角度
#         target_logits = torch.cos(theta + self.m)
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, labels.view(-1, 1), 1)

#         logits = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)
#         loss = F.cross_entropy(logits, labels)
#         return loss

# load from pytorch https://kevinmusgrave.github.io/pytorch-metric-learning/losses/
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, s=30.0, m=0.50):  # Use double underscores for __init__
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(embedding_size, num_classes))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, labels):
        x = F.normalize(x, p=2, dim=1)  

        W = F.normalize(self.W, p=2, dim=0)  
 
        cosine = torch.matmul(x, W)

        theta = torch.acos(cosine.clamp(-1.0, 1.0))  
       
        target_logits = torch.cos(theta + self.m)


        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

   
        logits = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)
        loss = F.cross_entropy(logits, labels)
        return loss


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits