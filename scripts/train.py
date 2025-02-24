import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataloader import DatasetLoader
from vgg import VGG 

MODEL_NAME = "vgg_trained3.pth"
BATCH_SIZE = 256
EPOCHS = 80
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {DEVICE}")

def train():
    loader = DatasetLoader()

    train_dataset = TensorDataset(loader.train_images, loader.train_labels)
    val_dataset = TensorDataset(loader.val_images, loader.val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VGG()
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    # training loop
    for epoch in range(EPOCHS):
        print(f"starting epoch: {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"epoch [{epoch+1}/{EPOCHS}], loss: {running_loss/len(train_loader):.4f}, train accuracy: {train_accuracy:.2f}%")

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"validation accuracy: {val_accuracy:.2f}%")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_NAME)  # Save best model
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")


    # Save trained model
    torch.save(model.state_dict(), MODEL_NAME)
    print("training is complete")

if __name__ == "__main__":
    train()
