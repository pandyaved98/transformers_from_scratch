import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_weights(attention_weights, src_tokens, tgt_tokens):
    plt.figure(figsize=(10, 10))
    sns.heatmap(attention_weights, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap="YlGnBu")
    plt.xlabel("Source Tokens")
    plt.ylabel("Target Tokens")
    plt.title("Attention Weights")
    plt.show()
