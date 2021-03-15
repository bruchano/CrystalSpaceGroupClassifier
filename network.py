import torch
import matplotlib.pyplot as plt


def train_one_epoch(device, model, optimizer, criterion, train_loader):
    print("Training starts")
    model.train()
    loss_epoch = 0.
    for b, (batch_input, batch_label) in enumerate(train_loader):
        for i in range(len(batch_input)):
            # reset gradient history
            optimizer.zero_grad()
            # read data
            data_input, data_label = batch_input[i], batch_label[i]
            data_input, data_label = data_input.to(device), data_label.to(device)
            # feed
            output = model(data_input)
            loss = criterion(output, data_label)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        print("\r\ttrain batch:{}/{}".format(b, len(train_loader)), end="")
    return round(loss_epoch, 4)


def validate_one_epoch(device, model, criterion, valid_loader):
    print("Evaluating starts")
    model.eval()
    num_valid = len(valid_loader.sampler.indices)
    if num_valid == 0:
        raise FileNotFoundError("length of data is 0")
    val_loss = 0.
    num_correct = 0
    for b, (batch_input, batch_label) in enumerate(valid_loader):
        for i in range(len(batch_input)):
            # read data
            data_input, data_label = batch_input[i], batch_label[i]
            data_input, data_label = data_input.to(device), data_label.to(device)
            # feed
            output = model(data_input)
            # record fitness
            val_loss += criterion(output, data_label).item()
            if torch.max(output, 1)[1] == data_label:
                num_correct += 1
        print("\r\tvalid batch:{}/{}".format(b, len(valid_loader)), end="")

    val_loss /= num_valid
    num_correct /= num_valid
    return round(val_loss, 4), round(num_correct*100, 4)


def validate_train_loop(device, model, optimizer, scheduler, criterion, valid_loader, train_loader, num_epoch, save_path, fig_path):
    result = validate_one_epoch(device, model, criterion, valid_loader)
    print("\rvalid loss:{} accuracy:{}%".format(*result))

    # used to plot
    loss_list = []
    epoch_list = []
    pred_res = []

    for epoch in range(num_epoch):
        result = train_one_epoch(device, model, optimizer, criterion, train_loader)
        print("\rtrain epoch:{} loss:{}".format(epoch, result))

        # plot loss Vs. epoch
        loss_list.append(result)
        epoch_list.append(epoch + 1)

        result = validate_one_epoch(device, model, criterion, valid_loader)
        print("\rvalid loss:{} accuracy:{}%".format(*result))

        pred_res.append(result[1])  # plot prediction result Vs. epoch

        scheduler.step(epoch)
        torch.save(model.state_dict(), save_path)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(epoch_list, loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0, 6)
    plt.title("Loss vs Epoch")

    plt.subplot(2, 1, 2)
    plt.plot(epoch_list, pred_res)
    plt.xlabel("epoch")
    plt.ylabel("prediction result (%)")
    plt.title("prediction Vs. epoch")
    plt.ylim(0, 100)
    plt.savefig(fig_path)
    plt.show()


