import torch
import matplotlib.pyplot as plt

from reproduction_model import IAMPolicy, IAMBase
# path = 'trained_models\ppo\Warehouse_03-31-2021-23-15-04.pt'
path = 'trained_models\ppo\Warehouse_03-31-2021-23-16-25.pt'
iam_policy: IAMPolicy = torch.load(path)[0]
base: IAMBase = iam_policy.base
# base.static_A_matrix.weight

plt.figure()
plt.imshow(base.static_A_matrix.weight.cpu().detach().numpy())
plt.show()