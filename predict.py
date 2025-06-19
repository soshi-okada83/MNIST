import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# モデルの定義（train.pyと同じもの）
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# モデルを読み込み
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# 入力画像の変換
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 手書き数字の画像を読み込み
image = Image.open("sample.png")
input_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加

# 予測
output = model(input_tensor)
predicted = torch.argmax(output, 1)
print(f"予測結果: {predicted.item()}")

# 画像表示
plt.imshow(image, cmap="gray")
plt.title(f"予測: {predicted.item()}")
plt.show()