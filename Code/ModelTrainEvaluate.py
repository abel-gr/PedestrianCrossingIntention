import numpy as np
from sklearn.metrics import accuracy_score

from torch import no_grad
from torch import argmax
from torch import reshape

def evaluate(model, loader, device):
    model.eval()

    predictions = []
    labels = []

    with no_grad():
        for data in loader:

            data = data.to(device)

            pred = model(data).detach().cpu()
            pred = argmax(pred, dim=-1).tolist()
            predictions.extend(pred)

            label = data.label.detach().cpu().float().numpy().tolist()
            labels.extend(label)

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    
    return accuracy_score(labels.reshape(-1), predictions.reshape(-1))



def train(model, train_loader, device, optimizer, crit):

    model.train()

    loss_all = 0
    samples = 0

    for data in train_loader:

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        label = data.y.to(device)

        label = reshape(label, (output.shape[0], output.shape[1]))

        loss = crit(output, label)

        loss.backward()
        loss_all = loss_all + data.num_graphs * loss.item()
        samples = samples + 1
        optimizer.step()

    return loss_all / samples