from dataloader import DatasetLoader

def test_data_loader(): # function to test the dataloader. prints amount of data in the datasets
    loader = DatasetLoader()

    train_images, train_labels = loader.train_images, loader.train_labels
    val_images, val_labels = loader.val_images, loader.val_labels

    print(f"Train dataset: {len(train_images)} samples, {len(train_labels)} labels")
    print(f"Validation dataset: {len(val_images)} samples, {len(val_labels)} labels")

    print(f"Train images shape: {train_images.shape}")
    print(f"Validation images shape: {val_images.shape}")

if __name__ == "__main__":
    test_data_loader()
