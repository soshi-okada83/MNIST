import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# モデルの定義（train.pyと完全に同じもの）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # [1, 28, 28] -> [16, 14, 14]
        x = self.pool(torch.relu(self.conv2(x))) # -> [32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルを読み込み
model = Net()
model.load_state_dict(torch.load("mnist_model.pth")) # 保存したパラメータ（重み）を読み込む。
model.eval() # 推論モードに切り替える

# 入力画像の変換
transform = transforms.Compose([
    transforms.Grayscale(), # 白黒化
    transforms.Resize((28, 28)), # サイズ調整
    transforms.ToTensor(), # Tensor化（0~1に変換）
    transforms.Normalize((0.5,), (0.5,)) # 正規化（[-1, 1]）
])

# 手書き数字の画像を読み込み
image = Image.open("sample.png") # 手書き数字画像ファイルを開く
input_tensor = transform(image).unsqueeze(0)  # 画像をテンソルに変換。バッチ次元（一枚分）を追加

# 予測
output = model(input_tensor) # 推論（予測スコア10個が返る）
predicted = torch.argmax(output, 1) # 一番スコアの高いクラス（数字）を選ぶ
print(f"予測結果: {predicted.item()}") # Tensorを数値に変換

# 画像表示
plt.imshow(image, cmap="gray")
plt.title(f"予測: {predicted.item()}")
plt.show()