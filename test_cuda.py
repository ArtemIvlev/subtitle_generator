import torch

print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Количество GPU:", torch.cuda.device_count())
    print("Имя GPU:", torch.cuda.get_device_name(0))
    print("Версия CUDA:", torch.version.cuda) 