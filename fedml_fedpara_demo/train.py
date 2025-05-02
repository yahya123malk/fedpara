import torch
import torch.nn as nn
import torch.optim as optim

from model.simple_cnn import SimpleCNN
from data.mnist_loader import get_mnist_loaders
from fedpara.compress import fedpara_trainable_compress  # <- updated import


def train_one_epoch(model, dataloader, device):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()


def get_model_delta(model_before, model_after):
    delta = {}
    for name, param_before in model_before.state_dict().items():
        delta[name] = model_after.state_dict()[name] - param_before
    return delta


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def cosine_similarity(t1, t2):
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    return torch.dot(t1_flat, t2_flat) / (t1_flat.norm() * t2_flat.norm() + 1e-8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_loaders()

    global_model = SimpleCNN().to(device)
    client_model = SimpleCNN().to(device)
    client_model.load_state_dict(global_model.state_dict())  # sync with global

    train_one_epoch(client_model, train_loader, device)

    delta = get_model_delta(global_model, client_model)
    print("Delta computed for layers:", list(delta.keys()))

    # Compress and show cosine similarity
    compressed_deltas = {}
    for name, d in delta.items():
        if len(d.shape) == 2:
            W_hat, A, B = fedpara_trainable_compress(d, rank=8)
            compressed_deltas[name] = W_hat
            cos_sim = cosine_similarity(d, W_hat)
            print(f"[FedPara-Trained] {name}: cos_sim = {cos_sim:.4f}")
            print(f"Norm of delta: {d.norm():.4f}, Norm of approx: {W_hat.norm():.4f}")
            mse = torch.nn.functional.mse_loss(W_hat, d)
            relative_mse = mse / (d.norm()**2 + 1e-8)
            print(f"[FedPara-Trained] {name}: relative MSE = {relative_mse:.6f}")
            print(f"[FedPara-Trained] {name}: cos_sim = {cos_sim:.4f}")
        else:
            print(f"[Skip] {name}: not a 2D weight")

    # Apply compressed update to global model
    with torch.no_grad():
        for name, d in delta.items():
            if name in compressed_deltas:
                global_model.state_dict()[name].add_(compressed_deltas[name])

    acc = evaluate_model(global_model, test_loader, device)
    print(f"✅ Test accuracy after applying FedPara update: {acc:.4f}")


if __name__ == "__main__":
    main()
