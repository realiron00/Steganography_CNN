import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 학습 데이터셋 경로
folder_path_cover = "/content/drive/MyDrive/dataset/cover/"
folder_path_stego = "/content/drive/MyDrive/dataset/stego/"

# 테스트 데이터셋 경로
folder_path_test = "/content/drive/MyDrive/case2.9_output/"

# RGBA 이미지를 RGB로 변환하는 함수
def rgba_to_rgb(image):
    # RGBA 이미지를 RGB로 변환
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3]) # 3번째 채널은 Alpha 채널입니다.
        return background
    else:
        return image

# 이미지 파일을 읽어서 numpy 배열로 변환하는 함수
def process_image(image_path):
    image = Image.open(image_path)
    image = rgba_to_rgb(image)  # RGBA를 RGB로 변환
    image = image.resize((32, 32))
    image_array = np.array(image)
    return image_array

# 학습 데이터셋 준비
X_train_all = []
y_train_all = []

# cover_2와 stego_2 폴더에 있는 파일들을 순회하면서 학습 데이터를 준비합니다.
for i in range(3000, 6000):
    filename_cover = f"{i}.png"
    filename_stego = f"{i}.png"

    image_path_cover = os.path.join(folder_path_cover, filename_cover)
    image_path_stego = os.path.join(folder_path_stego, filename_stego)

    image_array_cover = process_image(image_path_cover)
    image_array_stego = process_image(image_path_stego)

    X_train_all.append(image_array_cover)
    X_train_all.append(image_array_stego)

    y_train_all.append(0)  # 원본 이미지의 레이블
    y_train_all.append(1)  # 스테가노그래피가 적용된 이미지의 레이블

# 전체 데이터셋을 하나로 결합
X_train = np.array(X_train_all)
y_train = np.array(y_train_all)

# 테스트 데이터셋 준비
X_test_all = []
y_test_all = []

# 테스트 데이터셋을 준비합니다.
for filename in os.listdir(folder_path_test):
    if filename.endswith(".png"):
        image_path_test = os.path.join(folder_path_test, filename)
        image_array_test = process_image(image_path_test)
        X_test_all.append(image_array_test)
        y_test_all.append(1)  # 테스트 셋의 경우 모두 스테가노그래피가 적용된 이미지이므로 레이블을 1로 설정합니다.

# 전체 테스트 데이터셋을 numpy 배열로 변환
X_test = np.array(X_test_all)
y_test = np.array(y_test_all)

# 모델 정의 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일 
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습 
model.fit(X_train, y_train, epochs=30, batch_size=32)

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)
