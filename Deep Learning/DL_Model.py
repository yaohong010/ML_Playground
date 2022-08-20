import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
  def __init__(self, INPUT_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, HIDDEN_DIM_3, CLUSTERS_DIM, OUTPUUT_DIM):
    super(AutoEncoder, self).__init__()

    self.encoder = nn.Sequential(
        nn.Linear(INPUT_DIM, HIDDEN_DIM_1),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_3),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM_3, CLUSTERS_DIM), # compress to 3 features which can be visualized in plt
    )
    self.decoder = nn.Sequential(
        nn.Linear(CLUSTERS_DIM, HIDDEN_DIM_3),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM_3, HIDDEN_DIM_2),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_1),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM_1, OUTPUUT_DIM),
        nn.Sigmoid(),   # compress to range (0, 1)
    )

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return encoded, decoded