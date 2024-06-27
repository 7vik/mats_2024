def train(model, train_loader, loss_fn, optimizer, device):
    train_losses = []
    model.train()
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_idx % 400 == 0:
                print(f'.', end='')
    return train_losses