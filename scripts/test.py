import torch
import time
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from dataloader import DatasetLoader
from vgg import VGG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {DEVICE}")

# function to "warn-up" GPU to get accurate result on first latency
def warm_up(model, device, input_shape=(1, 3, 32, 32)):
    if device.type == "cuda":
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            model(dummy_input)
        torch.cuda.synchronize()
        print("GPU warm-up completed.")

# function to test the trained model on test data set, record results and print them to the concole, create the submission file "score.csv"
def test_model(model, loader):
    test_dataset = TensorDataset(loader.test_images, loader.test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    total_time = 0
    results = []

    warm_up(model, DEVICE)


    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image, label = image.to(DEVICE), label.to(DEVICE)

            start_time = time.perf_counter()  # time.time() resulted in some times being 0.0
            
            output = model(image)

            end_time = time.perf_counter()
            _, predicted = torch.max(output, 1) # gets predicted value by choosing the max value of the outputs
            inference_time = (end_time - start_time) * 1000

            total_time += inference_time
            # print(f"start time: {start_time}\nend time: {end_time}\n total time: {total_time}\n inference time: {inference_time}")

            results.append([loader.test_ids[i], predicted.item(), inference_time])

            total += 1

            correct += (predicted == label).item() # if predicted equals label, add one to correct

    submission = pd.DataFrame(results, columns=["id", "label", "latency"])
    submission.to_csv("score.csv", index=False)

    # compute results. please make sure dataloader.py uses correct test.csv file.
    avg_latency = total_time / total
    avg_accuracy = (correct / total) * 100 
    score = avg_accuracy / avg_latency
    print(f"Total Correct: {correct}/{total}")
    print(f"Average Latency: {avg_latency:.6f} ms")
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Score: {score:.6f}")

if __name__ == "__main__":
    loader = DatasetLoader()

    model = VGG()
    model.load_state_dict(torch.load("vgg_trained2.pth", map_location=DEVICE))
    model.to(DEVICE)

    test_model(model, loader)
