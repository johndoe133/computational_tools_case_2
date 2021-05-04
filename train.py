from tqdm import tqdm_notebook as tqdm
import torch
device = "cuda"

def train(epochs, model, opt, loss_fn, val_loader, train_loader):
    for epoch in tqdm(range(epochs)):

        val_avg_loss = 0
        model.eval()
        val_correct= 0
        val_total = 0
        with torch.no_grad():
            for X_val_batch, Y_val_batch in val_loader:
                #X_val_batch = X_val_batch.type(torch.FloatTensor)
                #Y_val_batch = torch.tensor(Y_val_batch).long()
                # The next to lines allows the model to run on the gpu if cuda is available
                X_val_batch = X_val_batch.to(device)
                Y_val_batch = Y_val_batch.to(device)

                # forward pass
                Y_val_pred = model(X_val_batch)
                #Y_val_batch = Y_val_batch.view(-1,1)
                val_loss = loss_fn(Y_val_pred,Y_val_batch)  # compute loss

                # Compute average epoch loss
                val_avg_loss += val_loss
                _, predicted = torch.max(Y_val_pred.data, 1)
                val_total += Y_val_batch.size(0)
                val_correct += (predicted == Y_val_batch).sum().item()
            
        val_avg_loss = val_avg_loss / len(val_loader)


        train_avg_loss = 0
        model.train()  # Set the model to train mode. This will enable dropout and so on if implemented
        correct= 0
        total = 0
        for X_batch, Y_batch in train_loader:
            #X_batch = X_batch.type(torch.FloatTensor)
            #Y_batch = torch.tensor(Y_batch).long()
            # The next to lines allows the model to run on the gpu if cuda is available
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward pass
            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)  # compute loss
            loss.backward()  # backward-pass to compute gradients
            opt.step()  # update weights

            # Compute average epoch loss
            train_avg_loss += loss

            _, predicted = torch.max(Y_pred.data, 1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

        train_avg_loss = train_avg_loss / len(train_loader)
            
        if epoch %10 == 0:
            print(f"Epoch: {epoch} Training loss: {train_avg_loss} Training acc: {(correct/total)* 100}. Validation loss: {val_avg_loss} Validation acc: {(correct/total)* 100}" )