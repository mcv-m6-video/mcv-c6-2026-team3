import re
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def parse_log(text):
    epochs, train_losses, val_losses, best_epochs = [], [], [], []

    for line in text.splitlines():
        match = re.match(r"\[Epoch (\d+)\] Train loss: ([\d.]+) Val loss: ([\d.]+)", line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))
        if "New best mAP epoch!" in line and epochs:
            best_epochs.append(epochs[-1])

    return epochs, train_losses, val_losses, best_epochs


def plot_losses(epochs, train_losses, val_losses, best_epochs, output_path="loss_plot.png"):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epochs, train_losses, color="#378ADD", linewidth=2,
            marker="o", markersize=4, label="Train loss")
    ax.plot(epochs, val_losses, color="#D85A30", linewidth=2, linestyle="--",
            marker="s", markersize=4, label="Val loss")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Train loss vs Validation loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(left=min(epochs) - 0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_losses.py <log_file> [output.png]")
        print("       cat training.log | python plot_losses.py -")
        sys.exit(1)

    input_arg = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "loss_plot.png"

    if input_arg == "-":
        text = sys.stdin.read()
    else:
        with open(input_arg, "r") as f:
            text = f.read()

    epochs, train_losses, val_losses, best_epochs = parse_log(text)

    if not epochs:
        print("No epoch data found. Check the log format.")
        sys.exit(1)

    print(f"Parsed {len(epochs)} epochs. Best mAP epochs: {best_epochs}")
    plot_losses(epochs, train_losses, val_losses, best_epochs, output_path)


if __name__ == "__main__":
    main()