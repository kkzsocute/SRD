import pandas as pd
import matplotlib.pyplot as plt

# file_prefixes = ['srdgru_pems-bay_6', 'srdtcn_pems-bay_6']
file_prefixes = ['srdgru_metr-la_6', 'srdtcn_metr-la_6']
colors = ['red', 'blue']  # 预定义的颜色列表

all_metrics = {'RSE': [], 'R2': [], 'Loss': []}

for file_prefix in file_prefixes:
    df = pd.read_csv(file_prefix + '.csv')
    all_metrics['RSE'].append(df.iloc[:, 1].tolist())
    all_metrics['R2'].append(df.iloc[:, 2].tolist())
    all_metrics['Loss'].append(df.iloc[:, 3].tolist())

def plot_metric(metric_name, values_list, file_prefixes, colors):
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    for values, file_prefix, color in zip(values_list, file_prefixes, colors):
        plt.plot(values, label=file_prefix, color=color)
    ax.set_ylabel(metric_name)
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.savefig(f"{metric_name.lower()}_metr-la_6.png", bbox_inches='tight')
    plt.clf()

# 使用循环绘制所有度量
for metric_name, values_list in all_metrics.items():
    plot_metric(metric_name, values_list, file_prefixes, colors)

print("Save successfully")
