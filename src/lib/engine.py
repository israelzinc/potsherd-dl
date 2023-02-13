from tqdm import tqdm
import torch


def train_fn(model, data_loader, optimizer, device):
    fin_loss = 0
    correct = 0
    total = 0
    model.train()
    tk = tqdm(data_loader, total=len(data_loader))    
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        outputs, loss = model(**data)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        labels = data["targets"]        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()        
        fin_loss += loss.item()    
    return ((correct/total), fin_loss / len(data_loader))


def evaluate(model, data_loader, device):
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(device)

            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            batch_preds = batch_preds.cpu()
            fin_preds.append(batch_preds)
        return fin_preds, fin_loss / len(data_loader)


def predict(model, data_loader, device):
    model.eval()
    final_predictions = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for data in tk0:
            for key, value in data.items():
                data[key] = value.to(device)
            predictions, _ = model(**data)
            predictions = predictions.cpu()
            final_predictions.append(predictions)
    return final_predictions


def salient(model, data_loader, device):
    model.eval()
    final_preds = []
    grads = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            # print(f'K={key},V={value}')
            data[key] = value.to(device)
            # print("DATA KEY",data[key])
            # data[key].require_grads()
        # predictions, _ = model(**data)
        output, _ = model(**data)
        final_preds.append(output)
        # print("OUTPUT SALIENCY",output)
        # Catch the output
        # output_idx = output.argmax()
        # output_max = output[0, output_idx]
        
        # Do backpropagation to get the derivative of the output based on the image
        # output_max.backward()

        # # Retireve the saliency map and also pick the maximum value from channels on each pixel.
        # # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
        # saliency, _ = torch.max(image.grad.data.abs(), dim=1) 
        # saliency = saliency.reshape(IMAGE_SIZE, IMAGE_SIZE)

        # predictions = predictions.cpu()
        # final_preds.append(predictions)
    
    return final_preds

