import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {DEVICE} device")
DEVICE = "cpu"

###########################################################################################
def trainmodel():
# Model
    # Number of layers
    # Type of layers
    # Number of nodes in each layer
# Optimizer
    # Learning rate
    # Momentum

    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 8 * 15, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 1)
    )

    """ model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8 * 8 * 15, 1)) """

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

    # ... and if you do, this initialization might not be relevant any more ...
    # model[1].weight.data = initialize_weights()
    # model[1].bias.data = torch.zeros((1000, 1))

    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    # 1000 - element tensor, each element is 15 (pieces), 8 x 8
    for epoch in range(20000):
        for x, y in trainloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(x)
            loss = torch.nn.functional.mse_loss(y_pred.reshape(1000, 1), y)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch ", epoch, " complete")

    torch.save(model, 'model.pkl')

###########################################################################################
if __name__=="__main__":
    trainmodel()
    
