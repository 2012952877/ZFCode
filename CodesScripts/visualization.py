from matplotlib import pyplot as plt


def plot_loss(train_loss, eval_loss):
    plt.plot(train_loss, '-o')
    plt.plot(eval_loss, '-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Loss')

    plt.show()
