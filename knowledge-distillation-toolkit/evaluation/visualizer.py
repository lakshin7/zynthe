import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, metrics, save_path):
    if not train_losses and not val_losses and not metrics:
        print("No data to plot.")
        return

    epochs = range(1, max(len(train_losses), len(val_losses), *(len(v) for v in metrics.values())) + 1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot loss curves
    axs[0].set_title("Training and Validation Loss")
    if train_losses:
        axs[0].plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    if val_losses:
        axs[0].plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Plot metrics curves
    axs[1].set_title("Metrics over Epochs")
    for metric_name, values in metrics.items():
        if values:
            axs[1].plot(range(1, len(values) + 1), values, label=metric_name.capitalize())
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Metric Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
