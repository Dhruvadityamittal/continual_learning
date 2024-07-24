import torch
from models.tinyhar import tinyhar
rand = torch.rand(1,3,200)


model = tinyhar(3,300, 728)
print(model)
out = model(rand)
# print(out.shape)



print(model.model.prediction.parameters())