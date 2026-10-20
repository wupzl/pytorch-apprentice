"""CPU-friendly classification demo: python examples/train.py"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(256, 2)
    y = (x[:, 0] * x[:, 1] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for epoch in range(60):
        model.train()
        total_loss = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(features)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}: loss={total_loss / len(x):.4f}')
    model.eval()
    with torch.no_grad():
        accuracy = (model(x.to(device)).argmax(dim=-1) == y.to(device)).float().mean()
    print(f'Training accuracy (not a test-set metric): {accuracy.item():.1%}')


if __name__ == '__main__':
    main()
