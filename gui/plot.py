import matplotlib.pyplot as plt
import numpy as np
import itertools


# noinspection PyUnresolvedReferences
def confusion_matrix(cm,
                     classes,
                     normalize=True,
                     title='Confusion matrix',
                     cmap=plt.cm.Blues,
                     image_address='.\\images\\tmp.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(16, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 size=5)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(image_address, dpi=100)


def normal(x, image_address='.\\images\\tmp.png'):
    plt.figure(figsize=(16, 9))
    mu = np.mean(x)
    sigma = np.std(x)
    count, bins, ignored = plt.hist(x, 10, normed=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.axvline(x=mu, color='r')

    plt.ylabel('Count / Probability')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.savefig(image_address, dpi=100)


def trend(trends, image_address='.\\images\\tmp.png'):
    plt.figure(figsize=(16, 9))
    plt.plot(trends[0], color='blue')
    plt.plot(trends[1], color='r')

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig(image_address, dpi=100)
