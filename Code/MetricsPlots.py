import numpy as np
import matplotlib.pyplot as plt
import os

def plot_loss_v1(num_epochs, loss_values, figsize=10, textsize=15):
    
    plt.figure(figsize=(figsize, figsize))
    plt.title("Training loss", size=textsize)
    plt.plot(np.arange(1, num_epochs+1), loss_values)
    plt.xlabel("Epoch", size=textsize)
    plt.ylabel("Loss", size=textsize)
    plt.xticks(size=textsize)
    plt.yticks(size=textsize)
    plt.show()
    
def plot_loss(num_epochs, metrics_train, metrics_val, figsize=10, dpi=150, textsize=15, epochs_step=20):
        
    plt.figure(figsize=(figsize, figsize), dpi=dpi)
    plt.plot(np.arange(1, num_epochs+1, epochs_step),
             [m['Loss'] for m in metrics_train[::epochs_step]],
             marker='o', color='purple', label='Train set')
    plt.plot(np.arange(1, num_epochs+1, epochs_step),
             [m['Loss'] for m in metrics_val[::epochs_step]],
             marker='s', color='tomato', label='Validation set')
    plt.grid(color='0.75', linestyle='-', linewidth=0.5)
    plt.title('Training loss', fontsize=25)
    plt.xticks(np.arange(1, num_epochs + epochs_step, epochs_step), fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Loss', fontsize=17)
    plt.xlabel('Epoch', fontsize=17)
    plt.legend(loc='best', fontsize=14)
    plt.show()
    
def plot_classification_metrics_train_val(num_epochs, metrics_train, metrics_val, figsize=10, 
                                          dpi=150, textsize=15, epochs_step=5, show=True, folderName='plots'):
    
    if folderName is not None:
        os.makedirs(folderName, exist_ok=True)
    
    for metricName in metrics_train[0].keys():
        
        if metricName.lower() == "epoch":
            continue
    
        metricValuesPerEpoch_train = []
        metricValuesPerEpoch_val = []

        plt.figure(figsize=(figsize, figsize), dpi=dpi)
        plt.title(metricName, size=textsize)

        for m in metrics_train:
            metricValuesPerEpoch_train.append(m[metricName])

        #plt.plot(np.arange(1, num_epochs+1), metricValuesPerEpoch_train, color='blue', label='Training set')
        plt.plot(np.arange(1, num_epochs+1, epochs_step),
                 metricValuesPerEpoch_train[::epochs_step],
                 marker='o', color='purple', label='Train set')

        if metricName in metrics_val[0].keys():
            for m in metrics_val:
                metricValuesPerEpoch_val.append(m[metricName])

            #plt.plot(np.arange(1, num_epochs+1), metricValuesPerEpoch_val, color='darkgreen', label='Validation set')
            plt.plot(np.arange(1, num_epochs+1, epochs_step),
                     metricValuesPerEpoch_val[::epochs_step],
                     marker='s', color='tomato', label='Validation set')
        
        plt.grid(color='0.75', linestyle='-', linewidth=0.5)
        plt.ylabel(metricName, size=textsize)
        plt.xlabel("Epoch", size=textsize)
        plt.xticks(np.arange(1, num_epochs + epochs_step, epochs_step), fontsize=textsize)
        plt.yticks(fontsize=textsize)
        plt.legend(loc='best', fontsize=textsize)
        
        if folderName is not None:
            plt.savefig(folderName + '/' + metricName + '.png', transparent=False)
        
        if show:
            plt.show()
        else:        
            plt.close()


def plot_ROC(classToPlot, fpr, tpr, roc_auc, figsize=10, dpi=150, textsize=15):
    
    className = 'Crossing' if classToPlot == 1 else 'Not-crossing'
    
    plt.figure(figsize=(figsize, figsize), dpi=dpi)
    plt.plot(fpr[classToPlot], tpr[classToPlot], color="darkorange", lw=2,
             label="ROC curve (area = {:.4f})".format(roc_auc[classToPlot]))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(fontsize=textsize)
    plt.yticks(fontsize=textsize)
    plt.grid(color='0.75', linestyle='-', linewidth=0.5)
    plt.ylabel("True Positive Rate", size=textsize)
    plt.xlabel("False Positive Rate", size=textsize)
    plt.title("Receiver operating characteristic (ROC) for class " + className, size=textsize)
    plt.legend(loc='lower right', fontsize=textsize)
    plt.show()