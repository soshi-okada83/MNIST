import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
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
# 'nn.Module'を継承して作成。PyTorchの全てのモデルは'nn.Module'を元に作る。
# __init__はPythonのコンストラクタ（初期化処理）。
# super()で親クラス(nn.Module)の初期化も忘れずに呼び出す。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) # 1層目 入力28x28=784次元 -> 出力128次元
        self.fc2 = nn.Linear(128, 64) # 2層目 128次元 -> 64次元 中間層。学習の表現力を高めるために使われる。
        self.fc3 = nn.Linear(64, 10) # 3層目 64次元 -> 10次元(0~9の数字) 出力層。最終的に、各数字に対するスコア（logits）を出力。

    def forward(self, x):
        x = x.view(-1, 28 * 28) # 画像は[バッチサイズ, 1, 28, 28]の4次元テンソル。[バッチサイズ, 784]の2次元にreshape(展開)して, 'Linear'層に入れられるようにする。
        x = torch.relu(self.fc1(x)) #1層目に通し、ReLu活性化関数を適用。
        x = torch.relu(self.fc2(x)) #2層目にもReLuを通す。ReLu(正規化線形関数)は勾配消失を起こしにくく学習が進みやすい。
        x = self.fc3(x) # 出力層へ。活性化関数を通さない（損失関数 CrossEntropyLossが内部でSoftmaxを適用するため）。
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
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"[{epoch+1}/{epochs}] loss: {running_loss/len(trainloader):.4f}")

# モデル保存
torch.save(net.state_dict(), "mnist_model.pth")
print("モデルを保存しました。")