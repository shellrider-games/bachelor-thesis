import torch

from sketch_detection.sketchdetection.train import get_model_instance_segmentation

def select_torch_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_model(path):
    device = select_torch_device()
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model

def main():
    device = select_torch_device()
    print(device)
    print('Hello, World!')

if __name__ == "__main__":
    main()