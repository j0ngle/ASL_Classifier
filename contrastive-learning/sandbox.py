from network import Identity
import torchvision.transforms as T
import torch

IMG_SIZE=224

preprocess = T.Compose([
    T.Resize(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize((.5, .5, .5), (.5, .5, .5))
])

encoder = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=False)
encoder.fc = Identity()

data = torch.randn(64, 3, 224, 224)
output = encoder(data)

print(len(output))