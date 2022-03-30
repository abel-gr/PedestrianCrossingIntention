import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

from torch import no_grad
from torch import argmax
from torch import reshape

def print_evaluation_train_val(epoch, train_metrics, val_metrics):

    train_metrics_items = list(train_metrics.items())
    val_metrics_items = list(val_metrics.items())
                
    evaluation = 'Epoch:'.ljust(33) + '{:03d}\n'.format(epoch + 1)
    
        
    output_p = []
        
    for metricName, metricValue in train_metrics_items:
        ev = ('Train ' + metricName + ':').ljust(26) + '{:10.4f}  -  '.format(metricValue)
        output_p.append(ev)
        
    i = 0
    for metricName, metricValue in val_metrics_items:
        ev = ('Validation ' + metricName + ':').ljust(31) + '{:10.4f}\n'.format(metricValue)
        
        if i < len(output_p):
            output_p[i] = output_p[i] + ev
        else:
            output_p.append(''.ljust(36) + '  -  ' + ev)
            
        i = i + 1
       
    for o in output_p:
        evaluation = evaluation + o
            
    print(evaluation, '\n\n')
    

def print_evaluation_test(test_metrics):
    
    evaluation = 'Testing subset results:\n\n'

    for metricName, metricValue in test_metrics.items():
        evaluation = evaluation + (metricName + ':').ljust(26) + '{:.4f}\n'.format(metricValue)

    print(evaluation, '\n\n')
    

"""
model: Model to be evaluated
loader: DataLoader with the data to the processed by the model
device: Device that will execute the model
computed_loss: If you already computed the loss you can include it to the dict of the metrics by passing its value here (float).
loss_crit: Loss function to be used to calculate the loss (e.g. torch.nn.BCELoss()). If not None, it will override the computed_loss parameter value.
"""
def evaluate(model, loader, device, computed_loss=None, loss_crit=None):
    
    model.eval()

    predictions = []
    labels = []

    loss_all = 0
    samples = 0
    
    with no_grad():
        for data in loader:

            data = data.to(device)
            
            # Compute predictions for results metrics
            output = model(data)
            
            pred = output.detach().cpu()
            pred = argmax(pred, dim=-1).tolist()
            predictions.extend(pred)

            label = data.label.detach().cpu().float().numpy().tolist()
            labels.extend(label)
            
            # Compute loss
            if loss_crit is not None:
                y = data.y.to(device)
                y = reshape(y, (output.shape[0], output.shape[1]))
                loss = loss_crit(output, y)
                
                #print('output:', output, '\n\ny:', y, '\n\nloss:', loss)

                loss_all = loss_all + loss.item() # * data.num_graphs
                samples = samples + 1

    
    metrics = {}
    
    if loss_crit is not None:
        av_loss = loss_all / samples
        metrics['Loss'] = av_loss
    elif computed_loss is not None:
        metrics['Loss'] = computed_loss
    
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    
    accuracy = accuracy_score(labels.reshape(-1), predictions.reshape(-1))
    metrics['Accuracy'] = accuracy
    
    balanced_accuracy = balanced_accuracy_score(labels.reshape(-1), predictions.reshape(-1))
    metrics['Balanced accuracy'] = balanced_accuracy

    precision = precision_score(labels.reshape(-1), predictions.reshape(-1))
    metrics['Precision'] = precision
    
    recall = recall_score(labels.reshape(-1), predictions.reshape(-1))
    metrics['Recall'] = recall
    
    f1 = f1_score(labels.reshape(-1), predictions.reshape(-1))
    metrics['f1-score'] = f1
    
    return metrics



"""
model: Model to be evaluated
loader: DataLoader with the data to the processed by the model
device: Device that will execute the model
"""
def predict(model, loader, device):
    
    model.eval()

    predictions = []
    labels = []
    
    with no_grad():
        for data in loader:

            data = data.to(device)
            
            # Compute predictions
            output = model(data)
            
            pred = output.detach().cpu()
            pred = argmax(pred, dim=-1).tolist()
            predictions.extend(pred)

            label = data.label.detach().cpu().float().numpy().tolist()
            labels.extend(label)
            
            
    return [predictions, labels] # Returns the predictions and the groundtruth


def ROC(model, loader, device, numberOfClasses):
    
    model.eval()

    predictions = np.zeros((loader.batch_size * len(loader), 2))
    labels = np.zeros((loader.batch_size * len(loader), 2))

    with no_grad():
        i = 0
        for data in loader:
            
            batch_s = data.label.shape[0]

            data = data.to(device)

            pred = model(data).detach().cpu().numpy()
            predictions[i:i+batch_s] = pred

            label = data.y.detach().cpu().float()
            label = reshape(label, (pred.shape[0], pred.shape[1])).numpy()
            
            labels[i:i+batch_s] = label
            
            i = i + batch_s

        
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0, numberOfClasses):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    return [fpr, tpr, roc_auc]
    
    
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
        loss_all = loss_all + loss.item() # * data.num_graphs
        samples = samples + 1
        optimizer.step()

    return loss_all / samples