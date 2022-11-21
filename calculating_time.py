file_name = input("File Name:")

with open(file_name, 'r') as f:
    lines = f.readlines()
total_num = 0
cnt = 0
for line in lines:
    if line.startswith("Time used in one epoch:  "):
        idx = line.index("Time used in one epoch:  ")
        length = len("Time used in one epoch:  ")
        num = float(line[length:])
        total_num += num
        cnt += 1

print(total_num)
print(total_num / cnt)
