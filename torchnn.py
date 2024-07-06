# imports
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# download datasets
# Convert MNIST to tensor
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
# process batches of 32 from dataset
dataset = DataLoader(train, 32)

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # sequential api from pytorch as the model
        self.model = nn.Sequential(
            # convolutional layer
            nn.Conv2d(1, 32, (3,3)),
            # activation function
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)
    
# Instance of neural network, loss, optimizer
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    with open('model_state.pt', 'rb') as f:
        # load the weights into classifier
        clf.load_state_dict(load(f))

    img = Image.open('img_3.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

    print(torch.argmax(clf(img_tensor)))
    # train 10 epochs
    # for epoch in range(10):
    #     for batch in dataset:
    #         # have x be train data and y be ground truth
    #         X, y = batch
    #         # send to cpu or cuda gpu
    #         X, y = X.to('cpu'), y.to('cpu')
    #         # apply model on X to get yhat as predictions
    #         yhat = clf(X)
    #         loss = loss_fn(yhat, y)

    #         # Apply backpropagation
    #         # Zero out existing gradients
    #         opt.zero_grad()
    #         # Calculate gradients
    #         loss.backward()
    #         # apply gradient descent
    #         opt.step()

    #     # Print out loss
    #     print(f"Epoch {epoch} loss is {loss.item()}")

    # # save this model to our environment
    # with open('model_state.pt', 'wb') as f:
    #     save(clf.state_dict(), f)