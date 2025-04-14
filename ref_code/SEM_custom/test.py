import kagglehub

# Download latest version
path = kagglehub.dataset_download("pytorch/resnet34")

print("Path to dataset files:", path)