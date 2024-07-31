import csv
import re

log_file_path = 'srdgru_metr-la_6.log'
csv_file_path = 'srdgru_metr-la_6.csv'

columns = ['epoch', 'rrse', 'r2', 'final_loss']

data = []

# pattern = re.compile(r"(\d+)\s+'(?:train|valid)'.*?rrse: ([\d.]+).*?r2: ([\d.]+).*?final_loss: ([\d.]+)")
pattern = re.compile(r"(\d+)\s+'valid'.*?rrse: ([\d.]+).*?r2: ([\d.]+).*?final_loss: ([\d.]+)")

with open(log_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            epoch, rrse, r2, final_loss = match.groups()
            data.append([epoch, rrse, r2, final_loss])

with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)
    writer.writerows(data)

print(f"save successfully to {csv_file_path}")
