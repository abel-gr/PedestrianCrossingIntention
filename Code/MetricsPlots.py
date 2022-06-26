import numpy as np
import matplotlib.pyplot as plt

def plot_loss(num_epochs, loss_values, figsize=10, textsize=15):
    
    plt.figure(figsize=(figsize,figsize))
    plt.title("Training loss", size=textsize)
    plt.plot(np.arange(1, num_epochs+1), loss_values)
    plt.xlabel("Epoch", size=textsize)
    plt.ylabel("Loss", size=textsize)
    plt.xticks(size=textsize)
    plt.yticks(size=textsize)
    plt.show()
    
    
def plot_classification_metrics_train_val(num_epochs, metrics_train, metrics_val, figsize=10, textsize=15):
    
    for metricName in metrics_train[0].keys():
        
        if metricName.lower() == "epoch":
            continue
    
        metricValuesPerEpoch_train = []
        metricValuesPerEpoch_val = []

        plt.figure(figsize=(figsize,figsize))
        plt.title(metricName, size=textsize)

        for m in metrics_train:
            metricValuesPerEpoch_train.append(m[metricName])

        plt.plot(np.arange(1, num_epochs+1), metricValuesPerEpoch_train, color='blue', label='Training set')

        if metricName in metrics_val[0].keys():
            for m in metrics_val:
                metricValuesPerEpoch_val.append(m[metricName])

            plt.plot(np.arange(1, num_epochs+1), metricValuesPerEpoch_val, color='darkgreen', label='Validation set')
        
        plt.xlabel("Epoch", size=textsize)
        plt.ylabel(metricName, size=textsize)
        plt.xticks(size=textsize)
        plt.yticks(size=textsize)
        plt.legend(loc='best', prop={'size': textsize})
        plt.show()


def plot_ROC(classToPlot, fpr, tpr, roc_auc, figsize=10, textsize=15):
    
    plt.figure(figsize=(figsize,figsize))
    plt.plot(fpr[classToPlot], tpr[classToPlot], color="darkorange", lw=2,
             label="ROC curve (area = {:.4f})".format(roc_auc[classToPlot]))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(size=textsize)
    plt.yticks(size=textsize)
    plt.xlabel("False Positive Rate", size=textsize)
    plt.ylabel("True Positive Rate", size=textsize)
    plt.title("Receiver operating characteristic (ROC) for class " + str(classToPlot), size=textsize)
    plt.legend(loc="lower right", prop={'size': textsize})
    plt.show()