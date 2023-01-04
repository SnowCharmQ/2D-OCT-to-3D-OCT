import os
import math
import matplotlib.pyplot as plt


def get_val(line: str, idx: int):
    i = idx
    while True:
        if i == len(line) or line[i] == ' ':
            break
        else:
            i += 1
    return line[idx:i]


def get_error_metrics(line: str):
    line = line.replace("\n", "").replace("\t", "")
    idx = line.index("mse: ") + 5
    mse = float(get_val(line, idx))
    idx = line.index("mae: ") + 5
    mae = float(get_val(line, idx))
    idx = line.index("rmse: ") + 6
    rmse = float(get_val(line, idx))
    idx = line.index("psnr: ") + 6
    psnr = float(get_val(line, idx))
    idx = line.index("ssim: ") + 6
    ssim = float(get_val(line, idx))
    return mse, mae, rmse, psnr, ssim


for file_name in os.listdir('out'):
    if os.path.isdir('out/' + file_name):
        continue
    f = open('out/' + file_name, 'r')
    lines = f.readlines()
    epoch = 0
    x_axis = []
    train_loss = []
    best_mse, best_mae, best_rmse, best_psnr, best_ssim = math.inf, math.inf, math.inf, -math.inf, -math.inf
    mses, maes, rmses, psnrs, ssims = [], [], [], [], []
    for line in lines:
        if line.startswith("Start training in epoch"):
            epoch += 1
            x_axis.append(epoch)
        if line.startswith("Finish Epoch: "):
            index = line.index("Loss:")
            index += 6
            train_loss.append(float(line[index:].replace("\t", "")))
        if line.startswith("Average"):
            mse, mae, rmse, psnr, ssim = get_error_metrics(line)
            mses.append(mse)
            maes.append(mae)
            rmses.append(rmse)
            psnrs.append(psnr)
            ssims.append(ssim)
            if mse < best_mse:
                best_mse = mse
            if mae < best_mae:
                best_mae = mae
            if rmse < best_rmse:
                best_rmse = rmse
            if psnr > best_psnr:
                best_psnr = psnr
            if ssim > best_ssim:
                best_ssim = ssim
        if line.startswith("Start testing"):
            break
    print("{} | mse: {} | mae: {} | rmse: {} | psnr: {} | ssim: {}"
          .format(file_name, best_mse, best_mae, best_rmse, best_psnr, best_ssim))

    dir_name = 'out/' + file_name.split(".")[0]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    plt.figure()
    plt.plot(x_axis, train_loss, '-', color='#4169E1', alpha=0.8, linewidth=1, label="Train Loss")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss Chart")
    plt.savefig(dir_name + "/loss.png")
    plt.close()

    plt.figure()
    plt.plot(x_axis, mses, '-', color='#4169E1', alpha=0.8, linewidth=1, label="MSE")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Validation MSE Chart")
    plt.savefig(dir_name + "/mse.png")
    plt.close()

    plt.figure()
    plt.plot(x_axis, maes, '-', color='#4169E1', alpha=0.8, linewidth=1, label="MAE")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Validation MAE Chart")
    plt.savefig(dir_name + "/mae.png")
    plt.close()

    plt.figure()
    plt.plot(x_axis, rmses, '-', color='#4169E1', alpha=0.8, linewidth=1, label="RMSE")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE Chart")
    plt.savefig(dir_name + "/rmse.png")
    plt.close()

    plt.figure()
    plt.plot(x_axis, psnrs, '-', color='#4169E1', alpha=0.8, linewidth=1, label="PSNR")
    plt.legend(loc="lower right")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("Validation PSNR Chart")
    plt.savefig(dir_name + "/psnr.png")
    plt.close()

    plt.figure()
    plt.plot(x_axis, ssims, '-', color='#4169E1', alpha=0.8, linewidth=1, label="SSIM")
    plt.legend(loc="lower right")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM Chart")
    plt.savefig(dir_name + "/ssim.png")
    plt.close()
