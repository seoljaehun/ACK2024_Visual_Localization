#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from imgaug import augmenters as iaa
from PIL import Image
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Concatenate, Add
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers.schedules import CosineDecay


# In[2]:


# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# In[3]:


# 경로 정규화 함수
def normalize_path(path):
    return os.path.normpath(path).replace("\\", "/")

# 경로 설정
train_road_image_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/1.Training/원천데이터_231107_add/512_roads_images")
train_road_mask_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/1.Training/라벨링데이터_231107_add/512_roads_masks")
val_road_image_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/원천데이터_231107_add/512_roads_images")
val_road_mask_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/라벨링데이터_231107_add/512_roads_masks")

train_building_image_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/1.Training/원천데이터_231107_add/512_buildings_images")
#train_building_snow_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/1.Training/원천데이터_231107_add/resized_buildings_snow")
train_building_mask_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/1.Training/라벨링데이터_231107_add/512_buildings_masks")
val_building_image_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/원천데이터_231107_add/512_buildings_images")
#val_building_snow_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/원천데이터_231107_add/resized_buildings_snow")
val_building_mask_dir = normalize_path("C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/라벨링데이터_231107_add/512_buildings_masks")

model_save_dir = normalize_path("C:/Users/mycom/Desktop/data/deeplabV3_models_ver3")


# In[4]:


# 데이터 증강 파이프라인 설정 (이미지를 512x512로 고려)
augmentation_pipeline = iaa.Sequential([
    iaa.Resize({"height": 512, "width": 512}),
    iaa.LinearContrast((0.75, 3.0)),  # 대비 조정 범위 증가
    iaa.Multiply((0.5, 3.0)),  # 밝기 조정 범위 증가
    iaa.Affine(rotate=(-45, 45), shear=(-20,20)),  # 회전 범위 확장
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.PerspectiveTransform(scale=(0.01, 0.1))  # 🔹 도로의 원래 형태 유지
])


# In[5]:


class SegmentationDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, mask_type='road', augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.mask_type = mask_type
        self.augment = augment
        self.image_filenames = sorted(os.listdir(image_dir))
        self.target_size = (512, 512)

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_images = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []

        for image_name in batch_images:
            image_path = os.path.join(self.image_dir, image_name)
            mask_filename = image_name.replace(".png", "_mask.png")
            mask_path = os.path.join(self.mask_dir, mask_filename)

            image = Image.open(image_path).convert("RGB").resize(self.target_size)
            image = np.array(image, dtype=np.float32) / 255.0

            mask = self.load_and_binarize_mask(mask_path)

            if self.augment:
                segmap = SegmentationMapsOnImage(mask, shape=self.target_size)
                augmented = augmentation_pipeline(image=image, segmentation_maps=segmap)
                image, mask = augmented[0], augmented[1].get_arr()

            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)

            images.append(image)
            masks.append(mask.astype(np.float32))

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

    def load_and_binarize_mask(self, mask_path):
        mask = np.array(Image.open(mask_path).convert("RGB").resize(self.target_size))

        if self.mask_type == 'road':
            mask = (mask[:, :, 0] > 127).astype(np.uint8)  # 도로 마스크 이진화

        elif self.mask_type == 'building':
            # 주황색 라벨에 해당하는 범위 설정 (조정 가능)
            lower_orange = np.array([200, 100, 0], dtype=np.uint8)
            upper_orange = np.array([255, 180, 100], dtype=np.uint8)

            # 주황색 픽셀만 추출하여 이진화
            mask = cv2.inRange(mask, lower_orange, upper_orange)
            mask = (mask > 0).astype(np.uint8)  # 0 또는 1로 변환

        return mask


# In[6]:


def aspp_block(x):
    """ASPP (Atrous Spatial Pyramid Pooling) 블록"""
    y1 = Conv2D(256, 1, padding="same", use_bias=False, activation='relu')(x)
    y1 = BatchNormalization()(y1)

    y2 = Conv2D(256, 3, dilation_rate=6, padding="same", use_bias=False, activation='relu')(x)
    y2 = BatchNormalization()(y2)

    y3 = Conv2D(256, 3, dilation_rate=12, padding="same", use_bias=False, activation='relu')(x)
    y3 = BatchNormalization()(y3)

    y4 = Conv2D(256, 3, dilation_rate=18, padding="same", use_bias=False, activation='relu')(x)
    y4 = BatchNormalization()(y4)

    y = Concatenate()([y1, y2, y3, y4])
    y = Conv2D(256, 1, padding="same", use_bias=False, activation='relu')(y)
    y = BatchNormalization()(y)
    
    return y

def deeplab_v3_plus(input_shape=(512, 512, 3), num_classes=1):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # 🔹 저수준 특징 맵 가져오기 (이 부분 수정)
    low_level_features = base_model.get_layer("conv2_block3_out").output  # 예상 크기: (128,128,48)
    low_level_features = Conv2D(64, (3, 3), padding="same", activation="relu")(low_level_features)  # 채널 맞추기

    # 🔹 ASPP 적용
    x = base_model.output  # 예상 크기: (16,16,2048)
    x = aspp_block(x)  # 예상 크기: (16,16,256)
    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)  # (64,64,256)

    # 🔹 고수준 특징 맵 업샘플링 (원본 해상도로 되돌리기)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # (128,128,256)

    # 🔹 고수준과 저수준 특징 결합
    print("Before Concatenate - x:", x.shape, "low_level_features:", low_level_features.shape)  # 디버깅용
    x = Concatenate()([x, low_level_features])  # 여기서 오류 발생할 가능성 있음

    # 🔹 최종 업샘플링
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # (256,256,128)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)  # (512,512,64)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid', padding='same')(x)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs)


# In[7]:


def weighted_focal_loss(alpha=0.5, gamma=3.0, class_weights={0: 0.3, 1: 3.0}):
    def loss(y_true, y_pred):
        # log 연산 안정성을 위해 epsilon 추가
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # 클래스 가중치 적용
        weights = tf.where(tf.equal(y_true, 1), class_weights[1], class_weights[0])

        # Focal Loss 계산
        focal_loss_pos = -alpha * weights * (1 - y_pred) ** gamma * y_true * tf.math.log(y_pred + epsilon)
        focal_loss_neg = -(1 - alpha) * weights * y_pred ** gamma * (1 - y_true) * tf.math.log(1 - y_pred + epsilon)

        return tf.reduce_mean(focal_loss_pos + focal_loss_neg)

    return loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice Loss: IoU 기반 손실 함수"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

def combined_loss(alpha=0.5, gamma=3.0, class_weights={0: 0.3, 1: 3.0}, focal_ratio=0.35):
    """Focal Loss + Dice Loss 결합"""
    focal = weighted_focal_loss(alpha=alpha, gamma=gamma, class_weights=class_weights)
    
    def loss(y_true, y_pred):
        return focal_ratio * focal(y_true, y_pred) + (1 - focal_ratio) * dice_loss(y_true, y_pred)

    return loss


# In[8]:


# IoU 계산 함수
def iou_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # 임계값 이상을 1, 이하를 0으로 변환
    y_true = tf.cast(y_true, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)  # 교집합
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection  # 합집합

    return (intersection + smooth) / (union + smooth)  # IoU 계산


# In[9]:


# 데이터 로더 생성
train_road_gen = SegmentationDataGenerator(train_road_image_dir, train_road_mask_dir, batch_size=2, mask_type='road', augment=True)
val_road_gen = SegmentationDataGenerator(val_road_image_dir, val_road_mask_dir, batch_size=2, mask_type='road', augment=False)
train_building_gen = SegmentationDataGenerator(train_building_image_dir, train_building_mask_dir, batch_size=2, mask_type='building', augment=True)
val_building_gen = SegmentationDataGenerator(val_building_image_dir, val_building_mask_dir, batch_size=2, mask_type='building', augment=False)


# In[10]:


def debug_generator(generator, num_batches=1, mask_type='road'):
    for i, (images, masks) in enumerate(generator):
        print(f"\n[디버깅 - 배치 {i+1} ({mask_type})]")
        print(f"이미지 배치 shape: {images.shape}, dtype: {images.dtype}")
        print(f"마스크 배치 shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"마스크 고유값: {np.unique(masks[0])}")  # 마스크 고유값 확인

        # 첫 번째 배치의 이미지와 마스크 시각화
        plt.figure(figsize=(10, 5))
        
        # 첫 번째 이미지 시각화
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(images[0])
        plt.axis("off")

        # 마스크 시각화 (0 또는 1인 값을 명확히 보기 위해 0~255로 스케일 업)
        mask_scaled = (masks[0].squeeze() * 255).astype(np.uint8)

        plt.subplot(1, 2, 2)
        plt.title("Mask (Scaled)")
        plt.imshow(mask_scaled, cmap='gray')
        plt.axis("off")

        plt.show()

        if i + 1 >= num_batches:
            break

# 디버깅 실행 - 도로 데이터
print("== 디버깅: 도로 마스크 ==")
road_gen = SegmentationDataGenerator(train_road_image_dir, train_road_mask_dir, batch_size=2, mask_type='road')
debug_generator(road_gen)

# 디버깅 실행 - 건물 데이터
print("== 디버깅: 건물 마스크 ==")
building_gen = SegmentationDataGenerator(train_building_image_dir, train_building_mask_dir, batch_size=2, mask_type='building')
debug_generator(building_gen, mask_type='building')


# In[67]:


# 도로 데이터의 클래스 가중치 계산
all_masks_road = np.concatenate([masks.flatten() for _, masks in train_road_gen], axis=0)
class_weights_road = compute_class_weight('balanced', classes=np.array([0, 1]), y=all_masks_road)
print(f"계산된 도로 클래스 가중치: {class_weights_road}")

# 건물 데이터의 클래스 가중치 계산
all_masks_building = np.concatenate([masks.flatten() for _, masks in train_building_gen], axis=0)
class_weights_building = compute_class_weight('balanced', classes=np.array([0, 1]), y=all_masks_building)
print(f"계산된 건물 클래스 가중치: {class_weights_building}")


# In[11]:


# 학습률 스케줄러 설정
initial_lr = 0.001  # 초기 학습률
epochs = 40  # 전체 학습 에포크
steps_per_epoch_road = len(train_road_gen)  # 도로 학습 데이터 배치 개수
steps_per_epoch_building = len(train_building_gen)  # 건물 학습 데이터 배치 개수

# Cosine Decay 스케줄러
lr_schedule_road = CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=epochs * steps_per_epoch_road,  # 전체 decay 스텝
    alpha=0.1  # 최저 학습률 (초기 대비 10%)
)

lr_schedule_building = CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=epochs * steps_per_epoch_building,  # 전체 decay 스텝
    alpha=0.1
)

# 모델 생성 (ASPP 적용)
road_model = deeplab_v3_plus(input_shape=(512, 512, 3), num_classes=1)
building_model = deeplab_v3_plus(input_shape=(512, 512, 3), num_classes=1)

# 가중치 설정
class_weights_road = {0: 0.25, 1: 3.0}  # 도로 가중치 강화
class_weights_building = {0: 0.4, 1: 3.0}  # 건물 가중치 유지

# 도로 모델 컴파일
road_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_road),
    loss=combined_loss(alpha=0.5, gamma=3, class_weights=class_weights_road),
    metrics=['accuracy', iou_metric]  # IoU 추가
)

# 건물 모델 컴파일
building_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_building),
    loss=combined_loss(alpha=0.5, gamma=3, class_weights=class_weights_building),
    metrics=['accuracy', iou_metric]  # IoU 추가
)


# In[12]:


# Early Stopping 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_loss',  # val_loss가 개선되지 않을 때 중지
    patience=10,  # 개선이 없을 때 10 에포크 이후 중지
    restore_best_weights=True,  # 가장 좋은 가중치를 복원
    verbose=1
)


# In[16]:


# 체크포인트 콜백 설정 (도로 모델)
checkpoint_filepath_road = os.path.join(model_save_dir, "best_road_model.h5")
checkpoint_callback_road = ModelCheckpoint(
    filepath=checkpoint_filepath_road,
    monitor='val_loss',  # val_loss 기준으로 저장
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# 도로 모델 학습
road_model.fit(
    train_road_gen,
    validation_data=val_road_gen,
    epochs=epochs,
    callbacks=[early_stopping, checkpoint_callback_road]
)

# 최종 도로 모델 저장
road_model.save(os.path.join(model_save_dir, "final_road_model.h5"))


# In[17]:


# 체크포인트 콜백 설정 (건물 모델)
checkpoint_filepath_building = os.path.join(model_save_dir, "best_building_model.h5")
checkpoint_callback_building = ModelCheckpoint(
    filepath=checkpoint_filepath_building,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# 건물 모델 학습
building_model.fit(
    train_building_gen,
    validation_data=val_building_gen,
    epochs=epochs,
    callbacks=[early_stopping, checkpoint_callback_building]
)

# 최종 건물 모델 저장
building_model.save(os.path.join(model_save_dir, "final_building_model.h5"))


# In[13]:


# 모델 경로
road_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models_ver2/best_road_model.h5"
building_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models/final_building_model.h5"

# 모델 로드 시 IoU 추가
road_model = tf.keras.models.load_model(
    road_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)

building_model = tf.keras.models.load_model(
    building_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)
# 이미지와 마스크를 불러오는 함수 (항상 RGB로 변환)
def load_image_and_mask(image_path, mask_path, target_size=(512, 512)):
    # RGB로 변환하여 알파 채널 제거
    image = np.array(Image.open(image_path).convert("RGB").resize(target_size), dtype=np.float32) / 255.0
    mask = np.array(Image.open(mask_path).resize(target_size), dtype=np.float32)
    return image, mask

# 예측과 시각화 함수
def visualize_predictions(model, image_dir, mask_dir, num_samples=3):
    image_filenames = sorted(os.listdir(image_dir))[:num_samples]

    for image_name in image_filenames:
        # 경로 설정 및 이미지/마스크 로드
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name.replace(".png", "_mask.png"))

        image, true_mask = load_image_and_mask(image_path, mask_path)
        image_input = np.expand_dims(image, axis=0)  # 배치 차원 추가

        # 예측 마스크 생성
        predicted_mask = model.predict(image_input)[0].squeeze()

        # 마스크 이진화 (0과 1로)
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

        # 시각화
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(true_mask.squeeze(), cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(predicted_mask, cmap='gray')
        plt.axis("off")

        plt.show()

# 도로 모델 시각화
print("== 도로 모델 예측 시각화 ==")
visualize_predictions(road_model, train_road_image_dir, train_road_mask_dir)

# 건물 모델 시각화
print("== 건물 모델 예측 시각화 ==")
visualize_predictions(building_model, train_building_image_dir, train_building_mask_dir)


# In[17]:


# 모델 경로
road_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models_ver2/final_road_model.h5"
building_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models/final_building_model.h5"


# 모델 로드 시 IoU 추가
road_model = tf.keras.models.load_model(
    road_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)

building_model = tf.keras.models.load_model(
    building_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)
# 이미지와 마스크를 불러오는 함수 (항상 RGB 변환)
def load_image(image_path, target_size=(512, 512)):
    image = np.array(Image.open(image_path).convert("RGB").resize(target_size), dtype=np.float32) / 255.0
    return image

# 예측과 시각화 함수 (마스크 없이)
def visualize_predictions_without_mask(model, image_dir, num_samples=3):
    image_filenames = sorted(os.listdir(image_dir))[:num_samples]

    for image_name in image_filenames:
        # 이미지 경로 설정 및 로드
        image_path = os.path.join(image_dir, image_name)
        image = load_image(image_path)
        image_input = np.expand_dims(image, axis=0)  # 배치 차원 추가

        # 예측 마스크 생성
        predicted_mask = model.predict(image_input)[0].squeeze()

        # 마스크 이진화 (0과 1로)
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

        # 시각화
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(predicted_mask, cmap='gray')
        plt.axis("off")

        plt.show()

# 🔹 테스트 데이터셋 경로 설정
test_road_image_dir = "C:/Users/mycom/Desktop/data/test/localization"
test_building_image_dir = "C:/Users/mycom/Desktop/data/test/localization"

# 도로 모델의 테스트 데이터 시각화
print("== 도로 모델 테스트 데이터 예측 시각화 ==")
visualize_predictions_without_mask(road_model, test_road_image_dir)

# 건물 모델의 테스트 데이터 시각화
print("== 건물 모델 테스트 데이터 예측 시각화 ==")
visualize_predictions_without_mask(building_model, test_building_image_dir)


# In[13]:


# 모델 경로
road_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models_ver2/best_road_model.h5"
building_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models/final_building_model.h5"

# 도로 및 건물 색상 지정 (BGR -> RGB)
ROAD_COLOR = (255, 165, 0)  # 주황색
BUILDING_COLOR = (51, 204, 255)  # 하늘색
BACKGROUND_COLOR = (0, 0, 0)  # 검정색 (배경)

# 시각화할 특정 이미지 파일명
target_filename = "BLD00017_PS3_K3A_NIA0276.png"

# 이미지 및 마스크 경로 설정
image_path = os.path.join(train_building_image_dir, target_filename)  # 도로/건물 이미지 동일
road_mask_path = os.path.join(train_building_mask_dir, target_filename.replace(".png", "_mask.png"))
building_mask_path = os.path.join(train_building_mask_dir, target_filename.replace(".png", "_mask.png"))

# 이미지 및 마스크 로드
def load_image_and_mask(image_path, road_mask_path, building_mask_path, target_size=(512, 512)):
    image = np.array(Image.open(image_path).convert("RGB").resize(target_size), dtype=np.float32) / 255.0
    road_mask = np.array(Image.open(road_mask_path).resize(target_size), dtype=np.uint8)
    building_mask = np.array(Image.open(building_mask_path).resize(target_size), dtype=np.uint8)

    # 차원 정리 (512, 512, 1) -> (512, 512)
    if len(road_mask.shape) == 3:
        road_mask = road_mask[:, :, 0]
    if len(building_mask.shape) == 3:
        building_mask = building_mask[:, :, 0]
    
    return image, road_mask, building_mask

# 도로 및 건물 모델 불러오기
road_model = tf.keras.models.load_model(
    road_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)

building_model = tf.keras.models.load_model(
    building_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)

# 예측 및 시각화
def visualize_prediction(image_path, road_mask_path, building_mask_path):
    # 원본 이미지 및 마스크 로드
    image, road_mask, building_mask = load_image_and_mask(image_path, road_mask_path, building_mask_path)
    
    # 예측 수행
    image_input = np.expand_dims(image, axis=0)
    road_pred = road_model.predict(image_input)[0].squeeze()
    building_pred = building_model.predict(image_input)[0].squeeze()

    # 예측 마스크 이진화
    road_pred = (road_pred > 0.5).astype(np.uint8)
    building_pred = (building_pred > 0.5).astype(np.uint8)

    # 마스크 색상화
    mask_image = np.zeros((512, 512, 3), dtype=np.uint8)  # 초기 검정색 배경
    mask_image[road_mask.squeeze() == 1] = ROAD_COLOR  # 도로 마스크
    mask_image[building_mask.squeeze() == 1] = BUILDING_COLOR  # 건물 마스크

    # 예측 마스크 색상화
    predict_image = np.zeros((512, 512, 3), dtype=np.uint8)  # 초기 검정색 배경
    predict_image[road_pred == 1] = ROAD_COLOR  # 예측된 도로
    predict_image[building_pred == 1] = BUILDING_COLOR  # 예측된 건물

    # 시각화
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask Image")
    plt.imshow(mask_image)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predict Image")
    plt.imshow(predict_image)
    plt.axis("off")

    plt.show()

# 특정 이미지 시각화 실행
visualize_prediction(image_path, road_mask_path, building_mask_path)


# In[26]:


# 📌 IoU 계산 함수 (추가)
def calculate_iou(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """두 개의 이진 마스크 간 IoU 계산"""
    y_pred = (y_pred > threshold).astype(np.float32)  # 예측값 이진화
    y_true = (y_true > 0.5).astype(np.float32)  # 실제 마스크 이진화

    intersection = np.sum(y_true * y_pred)  # 교집합
    union = np.sum(y_true) + np.sum(y_pred) - intersection  # 합집합

    return (intersection + smooth) / (union + smooth)  # IoU 계산

# 📌 특정 이미지에 대해 IoU 계산 및 시각화
def evaluate_iou(image_path, true_mask_path, model, title=""):
    """
    - image_path: 입력 이미지 경로
    - true_mask_path: 실제 마스크 이미지 경로
    - model: 학습된 모델
    """
    # 이미지 로드 및 전처리
    image = np.array(Image.open(image_path).convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
    true_mask = np.array(Image.open(true_mask_path).convert("L").resize((512, 512)), dtype=np.uint8)  # 흑백 변환

    # 모델 예측
    image_input = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image_input)[0].squeeze()

    # IoU 계산
    iou_score = calculate_iou(true_mask, pred_mask, threshold=0.5)

    # 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(true_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Predicted Mask (IoU: {iou_score:.4f})")
    plt.imshow((pred_mask > 0.5).astype(np.uint8), cmap="gray")  # 이진화 후 시각화
    plt.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.show()

    return iou_score

# 📌 도로 예측 IoU 계산 (BLD00001_PS3_K3A_NIA0276)
road_image_path = "C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/원천데이터_231107_add/512_roads_images/BLD01492_PS3_K3A_NIA0373.png"
road_mask_path = "C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/라벨링데이터_231107_add/512_roads_masks/BLD01492_PS3_K3A_NIA0373_mask.png"
road_iou = evaluate_iou(road_image_path, road_mask_path, road_model, title="Road Model IoU Evaluation")
print(f"📢 도로 IoU 결과: {road_iou:.4f}")

# 📌 건물 예측 IoU 계산 (BLD00002_PS3_K3A_NIA0276)
building_image_path = "C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/원천데이터_231107_add/512_buildings_images/OBJ00352_PS3_K3_NIA0083.png"
building_mask_path = "C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/라벨링데이터_231107_add/512_buildings_masks/OBJ00352_PS3_K3_NIA0083_mask.png"
building_iou = evaluate_iou(building_image_path, building_mask_path, building_model, title="Building Model IoU Evaluation")
print(f"📢 건물 IoU 결과: {building_iou:.4f}")


# In[14]:


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# 모델 경로
road_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models_ver2/best_road_model.h5"
building_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models/final_building_model.h5"

# 도로 및 건물 색상 지정 (BGR -> RGB)
ROAD_COLOR = (255, 165, 0)  # 주황색
BUILDING_COLOR = (51, 204, 255)  # 하늘색
BACKGROUND_COLOR = (0, 0, 0)  # 검정색 (배경)

# 사용자가 설정한 이미지 및 마스크 경로
image_path = "C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/원천데이터_231107_add/512_buildings_images/BLD00652_PS3_K3A_NIA0277.png"
road_mask_path = "C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/원천데이터_231107_add/512_buildings_images/BLD00652_PS3_K3A_NIA0277.png"
building_mask_path = "C:/Users/mycom/Desktop/data/019.위성영상객체판독/2.Validation/원천데이터_231107_add/512_buildings_images/BLD00652_PS3_K3A_NIA0277.png"
# 파일 존재 여부 확인
if not os.path.exists(image_path):
    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
if not os.path.exists(road_mask_path):
    raise FileNotFoundError(f"도로 마스크 파일을 찾을 수 없습니다: {road_mask_path}")
if not os.path.exists(building_mask_path):
    raise FileNotFoundError(f"건물 마스크 파일을 찾을 수 없습니다: {building_mask_path}")

# 이미지 및 마스크 로드 함수
def load_image_and_mask(image_path, road_mask_path, building_mask_path, target_size=(512, 512)):
    image = np.array(Image.open(image_path).convert("RGB").resize(target_size), dtype=np.float32) / 255.0
    road_mask = np.array(Image.open(road_mask_path).resize(target_size), dtype=np.uint8)
    building_mask = np.array(Image.open(building_mask_path).resize(target_size), dtype=np.uint8)

    # 차원 정리 (512, 512, 1) -> (512, 512)
    if len(road_mask.shape) == 3:
        road_mask = road_mask[:, :, 0]
    if len(building_mask.shape) == 3:
        building_mask = building_mask[:, :, 0]
    
    return image, road_mask, building_mask

# 도로 및 건물 모델 불러오기
road_model = tf.keras.models.load_model(
    road_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)

building_model = tf.keras.models.load_model(
    building_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)

# 예측 및 시각화
def visualize_prediction(image_path, road_mask_path, building_mask_path):
    # 원본 이미지 및 마스크 로드
    image, road_mask, building_mask = load_image_and_mask(image_path, road_mask_path, building_mask_path)
    
    # 예측 수행
    image_input = np.expand_dims(image, axis=0)
    road_pred = road_model.predict(image_input)[0].squeeze()
    building_pred = building_model.predict(image_input)[0].squeeze()

    # 예측 마스크 이진화
    road_pred = (road_pred > 0.5).astype(np.uint8)
    building_pred = (building_pred > 0.5).astype(np.uint8)

    # 예측 마스크 색상화
    predict_image = np.zeros((512, 512, 3), dtype=np.uint8)  # 초기 검정색 배경
    predict_image[road_pred == 1] = ROAD_COLOR  # 예측된 도로
    predict_image[building_pred == 1] = BUILDING_COLOR  # 예측된 건물

    # 시각화 (Predict Image만 출력)
    plt.figure(figsize=(5, 5))
    plt.title("Predict Image")
    plt.imshow(predict_image)
    plt.axis("off")
    plt.show()

# 특정 이미지 시각화 실행
visualize_prediction(image_path, road_mask_path, building_mask_path)


# In[39]:


# 모델 경로
road_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models_ver2/best_road_model.h5"
building_model_path = "C:/Users/mycom/Desktop/data/deeplabV3_models/final_building_model.h5"

# 도로 및 건물 색상 지정 (BGR -> RGB)
ROAD_COLOR = (255, 165, 0)  # 주황색
BUILDING_COLOR = (51, 204, 255)  # 하늘색
BACKGROUND_COLOR = (0, 0, 0)  # 검정색 (배경)

# 3개의 이미지 경로 설정
image_paths = [
    "C:/Users/mycom/Desktop/data/test/localization/5_0.png",
    "C:/Users/mycom/Desktop/data/test/localization/5_1.png",
    "C:/Users/mycom/Desktop/data/test/localization/5_2.png"
]

road_mask_paths = [
    "C:/Users/mycom/Desktop/data/test/localization/5_0.png",
    "C:/Users/mycom/Desktop/data/test/localization/5_1.png",
    "C:/Users/mycom/Desktop/data/test/localization/5_2.png"
]

building_mask_paths = [
    "C:/Users/mycom/Desktop/data/test/localization/5_0.png",
    "C:/Users/mycom/Desktop/data/test/localization/5_1.png",
    "C:/Users/mycom/Desktop/data/test/localization/5_2.png"
]

# 파일 존재 여부 확인
for i in range(3):
    if not os.path.exists(image_paths[i]):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_paths[i]}")
    if not os.path.exists(road_mask_paths[i]):
        raise FileNotFoundError(f"도로 마스크 파일을 찾을 수 없습니다: {road_mask_paths[i]}")
    if not os.path.exists(building_mask_paths[i]):
        raise FileNotFoundError(f"건물 마스크 파일을 찾을 수 없습니다: {building_mask_paths[i]}")

# 이미지 및 마스크 로드 함수
def load_image_and_mask(image_path, road_mask_path, building_mask_path, target_size=(512, 512)):
    image = np.array(Image.open(image_path).convert("RGB").resize(target_size), dtype=np.float32) / 255.0
    road_mask = np.array(Image.open(road_mask_path).resize(target_size), dtype=np.uint8)
    building_mask = np.array(Image.open(building_mask_path).resize(target_size), dtype=np.uint8)

    # 차원 정리 (512, 512, 1) -> (512, 512)
    if len(road_mask.shape) == 3:
        road_mask = road_mask[:, :, 0]
    if len(building_mask.shape) == 3:
        building_mask = building_mask[:, :, 0]
    
    return image, road_mask, building_mask

# 도로 및 건물 모델 불러오기
road_model = tf.keras.models.load_model(
    road_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)

building_model = tf.keras.models.load_model(
    building_model_path, 
    custom_objects={'loss': weighted_focal_loss(alpha=0.25, gamma=2.0), 'iou_metric': iou_metric}
)

# 예측 및 시각화
def visualize_predictions(image_paths, road_mask_paths, building_mask_paths):
    plt.figure(figsize=(15, 5))  # 한 줄에 3개 출력

    for i in range(3):
        # 원본 이미지 및 마스크 로드
        image, road_mask, building_mask = load_image_and_mask(image_paths[i], road_mask_paths[i], building_mask_paths[i])
        
        # 예측 수행
        image_input = np.expand_dims(image, axis=0)
        road_pred = road_model.predict(image_input)[0].squeeze()
        building_pred = building_model.predict(image_input)[0].squeeze()

        # 예측 마스크 이진화
        road_pred = (road_pred > 0.5).astype(np.uint8)
        building_pred = (building_pred > 0.5).astype(np.uint8)

        # 예측 마스크 색상화
        predicted_mask = np.zeros((512, 512, 3), dtype=np.uint8)  # 검정 배경
        predicted_mask[road_pred == 1] = ROAD_COLOR  # 예측된 도로
        predicted_mask[building_pred == 1] = BUILDING_COLOR  # 예측된 건물

        # 각 예측 결과 출력
        plt.subplot(1, 3, i + 1)
        plt.imshow(predicted_mask)
        plt.title(f"Prediction {i+1}")
        plt.axis("off")

    # 출력
    plt.show()

# 3개 이미지에 대해 시각화 실행
visualize_predictions(image_paths, road_mask_paths, building_mask_paths)


# In[ ]:




