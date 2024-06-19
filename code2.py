import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

# 이미지 크기 설정
IMG_SIZE = (64, 64)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# 사용자 정의 데이터셋 클래스
class StegoDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# 경로 설정
folder_path_cover = "/content/drive/MyDrive/dataset/cover/"
folder_path_stego = "/content/drive/MyDrive/dataset/stego/"

# 파일 리스트 생성
cover_files = [os.path.join(folder_path_cover, f"{i}.png") for i in range(3000, 6000)]
stego_files = [os.path.join(folder_path_stego, f"{i}.png") for i in range(3000, 6000)]

# 레이블 생성
cover_labels = [0] * len(cover_files)
stego_labels = [1] * len(stego_files)

# 데이터 통합
file_paths = cover_files + stego_files
labels = cover_labels + stego_labels

# 레이블 이진화
lb = LabelBinarizer()
labels = lb.fit_transform(labels).flatten()

# 각 클래스에서 900개의 이미지를 새로운 테스트셋으로 사용
cover_test_paths = cover_files[:900]
stego_test_paths = stego_files[:900]
cover_train_paths = cover_files[900:]
stego_train_paths = stego_files[900:]

train_paths = cover_train_paths + stego_train_paths
train_labels = [0] * len(cover_train_paths) + [1] * len(stego_train_paths)
test_paths = cover_test_paths + stego_test_paths
test_labels = [0] * len(cover_test_paths) + [1] * len(stego_test_paths)

# 데이터셋 및 데이터로더 생성
train_dataset = StegoDataset(train_paths, train_labels, transform=transform)
test_dataset = StegoDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# CNN 모델 정의
class StegoCNN(nn.Module):
    def __init__(self):
        super(StegoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# 모델 초기화 및 손실 함수, 옵티마이저 정의
model = StegoCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# GPU 사용 가능 시, 모델 이동
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 함수
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            labels = labels.float().unsqueeze(1)  # Reshape labels for BCEWithLogitsLoss
            inputs, labels = inputs.to(device), labels.to(device)

            # 이전 gradients 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 역전파 및 옵티마이저 스텝
            loss.backward()
            optimizer.step()

        # 매 애포크마다 테스트셋의 정확도 출력
        test_accuracy = evaluate_model(model, test_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Test Accuracy: {test_accuracy}")

# 평가 함수
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.float().unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = outputs > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# 모델 학습
# 모델 학습
train_losses, test_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50)
