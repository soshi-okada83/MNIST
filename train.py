import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# データ準備

# 画像に対する前処理（Transform）を定義
# transforms.Compose([...]) -> 複数の前処理を順番に実行
# transforms.ToTensor() -> PIL画像やNumPy配列をPyTorchのテンソルに変換。更に画素値を0 ~ 255 -> 0 ~ 1 に自動スケーリング
# transforms.Normalize((0.5,), (0.5,)) -> 各ピクセルを平均0・標準偏差1の範囲に正規化（標準化）する。入力値xを(x-0.5)/0.5に変換。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNISTデータセットの訓練データを取得。
# root ='./data' : データを保存するディレクトリ。
# download=True : まだデータがなければ自動でダウンロード。
# transform=... : 取得時に上記の前処理を適用。
# DataLoader はミニバッチ学習用にデータを取り出してくれるクラス
# batch_size=64 : 1回の学習ステップで使い画像枚数。
# shuffle=True : エポック毎にデータをシャッフルする。
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# テストデータ（train=False）を読み込む。
# テストデータは基本的にシャッフルしない（性能評価で順番を変える必要がないため）
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# モデル定義

# Netというニューラルネットワークのクラスを定義。
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

net = Net()

# 損失関数と最適化

# 分類タスクでよく使う「交差エントロピー損失」。
# モデルが出力した10クラスのスコア(logits)と、正解ラベルとの誤差を計算。
# 内部でSoftmaxを自動で適用してくれるため、出力層にSoftmaxは不要。
criterion = nn.CrossEntropyLoss()
# モデルパラメータを更新するための最適化アルゴリズムでAdamというよく使われる手法。
# 学習率lr=0.001は「どれだけ重みを動かすか」を決めるハイパーパラメータ。
# net.parametersとは、Netモデルの全てのパラメーター（重みやバイアス）を取得し、optimizerに渡すことで、optimizer.step()で自動的に全部更新してくれる。
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 学習ループ
epochs = 30 # エポック数 = 30回繰り返す
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad() # 勾配リセット
        outputs = net(images) # 順伝播
        loss = criterion(outputs, labels) # 損失関数
        loss.backward() # 勾配計算（逆伝播）
        optimizer.step() # パラメータ更新
        running_loss += loss.item()

    print(f"[{epoch+1}/{epochs}] loss: {running_loss/len(trainloader):.4f}")

# モデル保存
torch.save(net.state_dict(), "mnist_model.pth")
print("モデルを保存しました。")