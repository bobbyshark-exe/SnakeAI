import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(scores, mean_scores, save_path='training.png'):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.tight_layout()
    plt.savefig(save_path)
