import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Load the model
model = torchvision.models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')
num_features = model.classifier[1].in_features 
model.classifier[1] = nn.Linear(num_features, 555)
path = os.path.dirname(__file__)
my_file = path + '/model_checkpoint_effnetv2.pt'
device = torch.device('cpu')
checkpoint = torch.load(my_file, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define transformations
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load bird names into a list
my_file = path + '/names.txt'
bird_names = [line.strip() for line in open('names.txt')]

# Predicts the top 3 birds from an image
def predict(image):
    image_tensor = torch.unsqueeze(transformations(image), 0)
    image_tensor = image_tensor.to(device)

    output = model(image_tensor)

    preds = torch.nn.functional.softmax(output, dim=1)[0]
    _, indices = torch.sort(output, descending=True)
    return [(bird_names[i], preds[i].item()) for i in indices[0][:3]]


    


