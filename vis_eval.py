import pandas as pd
import matplotlib.pyplot as plt

file_prefix = 'srdtcn_metr-la_6'
df = pd.read_csv(file_prefix + '.csv')

metrics = {
    'RSE': df.iloc[:, 1].tolist(),
    'R2': df.iloc[:, 2].tolist(),
    'Loss': df.iloc[:, 3].tolist()
}


def plot_metric(metric_name, values, file_prefix):
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    plt.plot(values, color='red')
    ax.set_ylabel(metric_name)
    ax.set_xlabel('Epoch')
    plt.savefig(f"{metric_name.lower()}_{file_prefix}.png", bbox_inches='tight')
    plt.clf()


for metric_name, values in metrics.items():
    plot_metric(metric_name, values, file_prefix)

print("save successfully")
