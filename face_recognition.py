import cv2
import os
import numpy as np
from skimage.feature import hog

# 初始化 OpenCV 的人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 提取 HOG 特征
def extract_hog_features(image, resize_dim=(128, 128)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, resize_dim)
    
    # 提取 HOG 特征
    features, _ = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                      block_norm="L2-Hys", visualize=True)
    return features

# 检测人脸并提取特征
def extract_face_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    features = []

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]  # 裁剪人脸区域
        hog_features = extract_hog_features(face)
        features.append(hog_features)
    
    return features

def process_lfw_data(dataset_path, max_images_per_person=10):
    features = []
    labels = []

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        images = os.listdir(person_path)
        images = images[:max_images_per_person]  # 限制每人最多的图片数

        for img_name in images:
            img_path = os.path.join(person_path, img_name)
            face_features = extract_face_features(img_path)
            
            # 如果检测到人脸，则添加到结果中
            if len(face_features) == 1:  # 确保每张图片仅检测到一个人脸
                features.append(face_features[0])
                labels.append(person)

    return np.array(features), np.array(labels)

# 数据路径
dataset_path = "lfw"
X, y = process_lfw_data(dataset_path)
print(f"Total faces encoded: {len(X)}")

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 测试分类器
accuracy = knn.score(X_test, y_test)
print(f"KNN Classifier Accuracy: {accuracy * 100:.2f}%")

def predict_image(image_path, knn_classifier):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]  # 裁剪人脸区域
        features = extract_hog_features(face)
        
        # 预测人脸身份
        name = knn_classifier.predict([features])[0]
        print(f"Detected: {name}")

        # 绘制人脸框和名字
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 测试图片路径
test_image_path = "path_to_test_image.jpg"
predict_image(test_image_path, knn)
