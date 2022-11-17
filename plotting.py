import matplotlib.pyplot as plt

with open('nohup.out', 'r') as f:
    lines = f.readlines()
x_axis = []
y_axis = []
for line in lines:
    if line.startswith("Finish Epoch: "):
        line = line.replace("\n", "")
        start = line.index("[")
        end = line.index("]")
        x_axis.append(float(line[start + 1:end]))
        index = line.index("Loss: ")
        index += 6
        y_axis.append(float(line[index:].replace("\t", "")))

plt.plot(x_axis, y_axis, '-', color='#4169E1', alpha=0.8, linewidth=1, label='MSE Loss')
plt.legend(loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()
