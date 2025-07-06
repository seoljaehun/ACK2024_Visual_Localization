import torch
import cv2
import numpy as np
import os
import os.path as osp
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
from sklearn.cluster import DBSCAN
from PIL import Image
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config

def process_image_set(image_set, input_dir, output_dir, model_ckpt_b, model_ckpt_r):
    img_1 = mmcv.imread(os.path.join(input_dir, f"{image_set}_0.png"))
    img_2 = mmcv.imread(os.path.join(input_dir, f"{image_set}_1.png"))
    template_path = os.path.join(input_dir, f"{image_set}_1.png")
    tem_or_path = os.path.join(input_dir, f"{image_set}_2.png")
    original_path = os.path.join(input_dir, f"{image_set}_0.png")

    # 세그멘테이션 수행
    result_b1 = inference_segmentor(model_ckpt_b, img_1)
    result_r1 = inference_segmentor(model_ckpt_r, img_1)

    # 4️⃣ 마스크 변환
    mask_b1 = np.array(result_b1[0])  # 건물 마스크
    mask_r1 = np.array(result_r1[0])  # 도로 마스크

    # 5️⃣ 빈 RGB 마스크 생성 (배경: 검정)
    mask_color_1 = np.zeros((mask_b1.shape[0], mask_b1.shape[1], 3), dtype=np.uint8)

    # 6️⃣ 건물 마스크 적용 (하늘색: 135, 206, 235)
    mask_color_1[mask_b1 > 0] = (51, 204, 255)

    # 7️⃣ 도로 마스크 적용 (주황색: 255, 140, 0), 건물과 겹치지 않도록 조건 추가
    mask_color_1[(mask_r1 > 0) & (mask_b1 == 0)] = (255, 165, 0)

        # 3️⃣ 이미지 입력 (외부에서 불러오는 것이 아니라 기존 img 사용)
    result_b2 = inference_segmentor(model_ckpt_b, img_2)
    result_r2 = inference_segmentor(model_ckpt_r, img_2)

    # 4️⃣ 마스크 변환
    mask_b2 = np.array(result_b2[0])  # 건물 마스크
    mask_r2 = np.array(result_r2[0])  # 도로 마스크

    # 5️⃣ 빈 RGB 마스크 생성 (배경: 검정)
    mask_color_2 = np.zeros((mask_b2.shape[0], mask_b2.shape[1], 3), dtype=np.uint8)

    # 6️⃣ 건물 마스크 적용 (하늘색: 135, 206, 235)
    mask_color_2[mask_b2 > 0] = (51, 204, 255)

    # 7️⃣ 도로 마스크 적용 (주황색: 255, 140, 0), 건물과 겹치지 않도록 조건 추가
    mask_color_2[(mask_r2 > 0) & (mask_b2 == 0)] = (255, 165, 0)

    # 원본 이미지와 템플릿 이미지 읽기
    original_image_color = cv2.imread(original_path)  # 컬러 이미지
    tem_or_image_color = cv2.imread(tem_or_path)  # 컬러 이미지 (원본과 동일한 크기일 경우)
    original_image = cv2.cvtColor(tem_or_image_color, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    # 이미지가 정상적으로 로드되었는지 확인
    if original_image_color is None or template_image is None:
        print("Error: 이미지 파일을 불러올 수 없습니다. 경로를 확인하세요.")
        exit()

    # SIFT 특징점 추출기 생성
    sift = cv2.SIFT_create()

    # 특징점과 디스크립터 계산
    keypoints1, descriptors1 = sift.detectAndCompute(original_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)

    # 특징점 디스크립터 매칭 (KNN 매칭 사용)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 좋은 매칭 선택 (Lowe's Ratio Test)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 매칭된 점의 최소 개수 확인
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 호모그래피 계산
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if matrix is not None:
            h, w = template_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            # 사각형을 그리기 위한 좌상단 (x_min, y_min)과 우하단 (x_max, y_max) 좌표 찾기
            x_min_r, y_min_r = np.int32(dst.min(axis=0).ravel())
            x_max_r, y_max_r = np.int32(dst.max(axis=0).ravel())

            # 원본 컬러 이미지에 사각형 그리기
            original_image_with_rect = original_image_color.copy()
            cv2.rectangle(original_image_with_rect, (x_min_r, y_min_r), (x_max_r, y_max_r), (0, 0, 255), 3)

        else:
            print("호모그래피를 계산할 수 없습니다. 매칭이 충분하지 않습니다.")
    else:
        print("매칭된 특징점이 충분하지 않습니다.")

    # 1️⃣ 건물 및 도로 마스크 이미지 (RGB)
    mask_color_1_gray = cv2.cvtColor(mask_color_1, cv2.COLOR_BGR2GRAY)  # 원본 마스크
    mask_color_2_gray = cv2.cvtColor(mask_color_2, cv2.COLOR_BGR2GRAY)  # 템플릿 마스크

    # SIFT 특징점 추출기 생성
    sift = cv2.SIFT_create()

    # 특징점과 디스크립터 계산
    keypoints1, descriptors1 = sift.detectAndCompute(mask_color_1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(mask_color_2_gray, None)

    # BFMatcher를 이용한 KNN 매칭
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's Ratio Test 적용
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 매칭된 점이 충분한지 확인
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 🔹 호모그래피 행렬 계산 (오차를 더 허용)
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)  # reprojectionError 증가 (기존 5.0 → 10.0)

        if matrix is not None:
            h, w = mask_color_2_gray.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            # 🔹 1차 바운딩 박스 좌표 계산
            min_x = max(int(np.min(dst[:, 0, 0])), 0)
            min_y = max(int(np.min(dst[:, 0, 1])), 0)
            max_x = min(int(np.max(dst[:, 0, 0])), mask_color_1_gray.shape[1])
            max_y = min(int(np.max(dst[:, 0, 1])), mask_color_1_gray.shape[0])

            # 🔹 바운딩 박스를 10% 확장하여 유연한 탐색 가능하도록 조정
            expand_ratio = 4  # 10% 확장
            dx = int((max_x - min_x) * expand_ratio)
            dy = int((max_y - min_y) * expand_ratio)

            min_x = max(min_x - dx, 0)
            min_y = max(min_y - dy, 0)
            max_x = min(max_x + dx, mask_color_1.shape[1] - 1)
            max_y = min(max_y + dy, mask_color_1.shape[0] - 1)

            # 🔹 ROI가 유효한 크기인지 확인
            if max_y > min_y and max_x > min_x:
                roi = mask_color_1_gray[min_y:max_y, min_x:max_x]
                roi_color = mask_color_1[min_y:max_y, min_x:max_x]

                if roi.size == 0:
                    print("ROI가 비어 있음: 2차 매칭을 진행하지 않음")
                else:
                    # 2차 매칭: ROI에서 다시 SIFT 적용
                    keypoints_roi, descriptors_roi = sift.detectAndCompute(roi, None)

                    # BFMatcher를 이용한 2차 KNN 매칭
                    if descriptors_roi is not None and len(descriptors_roi) > 0:
                        knn_matches_2 = bf.knnMatch(descriptors_roi, descriptors2, k=2)

                        # Lowe's Ratio Test 적용
                        good_matches_2 = []
                        for m, n in knn_matches_2:
                            if m.distance < 0.75 * n.distance:
                                good_matches_2.append(m)

                        if len(good_matches_2) > 4:
                            match_pts = np.float32([keypoints_roi[m.queryIdx].pt for m in good_matches_2])

                            # DBSCAN을 이용하여 밀집된 영역(클러스터) 찾기
                            dbscan = DBSCAN(eps=20, min_samples=3).fit(match_pts)
                            labels = dbscan.labels_

                            # 가장 큰 클러스터(최다 샘플이 포함된 클러스터) 찾기
                            unique_labels, counts = np.unique(labels, return_counts=True)
                            largest_cluster_label = unique_labels[np.argmax(counts[:-1])]  # -1은 노이즈 제외

                            # 가장 큰 클러스터에 속하는 점들 선택
                            cluster_pts = match_pts[labels == largest_cluster_label]

                            # 클러스터 중심 계산
                            center_x = int(np.mean(cluster_pts[:, 0])) + min_x
                            center_y = int(np.mean(cluster_pts[:, 1])) + min_y

                            # 🔹 2차 바운딩 박스 설정
                            min_x_final = max(center_x - (x_max_r - x_min_r) // 2, 0)
                            min_y_final = max(center_y - (y_max_r - y_min_r) // 2, 0)
                            max_x_final = min(center_x + (x_max_r - x_min_r) // 2, mask_color_1.shape[1] - 1)
                            max_y_final = min(center_y + (y_max_r - y_min_r) // 2, mask_color_1.shape[0] - 1)

                        # 🟢 3️⃣ `road.png` 불러오기
                            road_img = cv2.imread(original_path)

                    # 크기 맞추기 (mask_color_1과 동일한 크기로 조정)
                            road_img_resized = cv2.resize(road_img, (mask_color_1.shape[1], mask_color_1.shape[0]))

                    # 🟢 4️⃣ `road.png`에 바운딩 박스 오버랩
                            #cv2.rectangle(road_img_resized, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)  # 1차 바운딩 박스 (파란색)
                            cv2.rectangle(road_img_resized, (min_x_final, min_y_final), (max_x_final, max_y_final), (0, 255, 0), 5)  # 2차 바운딩 박스 (초록색)

                        else:
                            print("2차 매칭된 특징점 부족")
                    else:
                        print("ROI에서 SIFT 특징점 부족")
            else:
                print("1차 바운딩 박스 크기가 유효하지 않음, 2차 매칭 불가능")
        else:
            print("호모그래피 계산 불가: 매칭 점 부족")
    else:
        print("매칭된 특징점 부족")
    # ✅ 최종 이미지에 두 개의 결과 오버랩
    final_image = original_image_color.copy()

    # 초록색 바운딩 박스 (첫 번째 결과)
    cv2.rectangle(final_image, 
                (min_x_final, min_y_final), 
                (max_x_final, max_y_final), 
                (0, 255, 0), 3)  # 초록색

    # 빨간색 바운딩 박스 (두 번째 결과)
    cv2.rectangle(final_image, 
                (x_min_r, y_min_r), 
                (x_max_r, y_max_r), 
                (0, 0, 255), 3)  # 빨간색

    # ✅ 교집합 영역 계산
    #inter_x_min = max(min_x_final, x_min_r)
    #inter_y_min = max(min_y_final, y_min_r)
    #inter_x_max = min(max_x_final, x_max_r)
    #inter_y_max = min(max_y_final, y_max_r)

    # ✅ 겹치는 영역의 넓이 계산
    #inter_width = max(0, inter_x_max - inter_x_min)
    #inter_height = max(0, inter_y_max - inter_y_min)
    #intersection_area = inter_width * inter_height  # 교집합 넓이

    intersection_area = compute_intersect_area(x_min_r, x_max_r, min_x_final, max_x_final, y_min_r, y_max_r, min_y_final, max_y_final)

    # ✅ 빨간색 바운딩 박스의 넓이
    red_box_area = (x_max_r - x_min_r) * (y_max_r - y_min_r)

    # ✅ 겹치는 영역 비율 계산 (빨간색 박스 기준)
    overlap_percentage = (intersection_area / red_box_area) * 100 if red_box_area > 0 else 0

    print(f"겹치는 영역 비율: {overlap_percentage:.2f}%")

    # 저장할 파일 경로 설정
    output_path = os.path.join(output_dir, f"{image_set}_result.png")

    # 이미지 저장
    cv2.imwrite(output_path, final_image)

    print(f"Final result saved at: {output_path}")

    return overlap_percentage

def compute_intersect_area(x1, x2, x3, x4, y1, y2, y3, y4):
        ## case1 오른쪽으로 벗어나 있는 경우

    if x2 < x3:
        return 0

        ## case2 왼쪽으로 벗어나 있는 경우
    if x1 > x4:
        return 0

        ## case3 위쪽으로 벗어나 있는 경우
    if  y2 < y3:
        return 0

        ## case4 아래쪽으로 벗어나 있는 경우
    if  y1 > y4:
        return 0

    left_up_x = max(x1, x3)
    left_up_y = max(y1, y3)
    right_down_x = min(x2, x4)
    right_down_y = min(y2, y4)

    width = right_down_x - left_up_x
    height =  right_down_y - left_up_y
    
    return width * height

classes_b = ('background', 'building')
palette_b = [[0, 0, 0], [48, 200, 248]]

@DATASETS.register_module()
class Dataset(CustomDataset):
  CLASSES = classes_b
  PALETTE = palette_b
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None

# config 파일을 설정하고, 다운로드 받은 pretrained 모델을 checkpoint로 설정. 
config_file = 'C:/Users/kkm/Downloads/mmsegmentation/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py'
checkpoint_file = 'C:/Users/kkm/Downloads/mmsegmentation/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'



cfg_b = Config.fromfile(config_file)

cfg_b.norm_cfg = dict(type='BN', requires_grad=True)
#cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg_b.model.decode_head.norm_cfg = cfg_b.norm_cfg

cfg_b.device='cuda'

cfg_b.model.decode_head.num_classes = 2

#cfg.data.samples_per_gpu=2
#cfg.data.workers_per_gpu=2

cfg_b.model.decode_head.loss_decode = [dict(type='CrossEntropyLoss', loss_weight = 1.0),
                                     dict(type='FocalLoss', loss_weight = 1.0),
                                     dict(type='LovaszLoss', loss_weight = 1.0, reduction='none')]

cfg_b.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg_b.crop_size = (512, 512)
cfg_b.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg_b.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg_b.img_norm_cfg),
    dict(type='Pad', size=cfg_b.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg_b.val_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='MultiScaleFlipAug',
                        img_scale=(1024, 1024),
                        flip=False,
                        transforms=[
                                    dict(type='Resize', keep_ratio=True),
                                    dict(type='RandomFlip'),
                                    dict(
                                        type='Normalize',
                                        mean=[123.675, 116.28, 103.53],
                                        std=[58.395, 57.12, 57.375],
                                        to_rgb=True),
                                    dict(type='ImageToTensor', keys=['img']),
                                    dict(type='Collect', keys=['img'])
                                    ]),
                    
]


cfg_b.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg_b.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

cfg_b.data.samples_per_gpu=1
cfg_b.data.workers_per_gpu=0
cfg_b.data.persistent_workers=False
cfg_b.dataset_type = 'Dataset'
cfg_b.data_root = 'D:/'

cfg_b.data.train.type = 'Dataset'
cfg_b.data.train.data_root = 'D:/'
cfg_b.data.train.img_dir = 'Data_set/building_dataset/img_dir/train'
cfg_b.data.train.ann_dir = 'Data_set/building_dataset/contour_ann_dir/train'
cfg_b.data.train.pipeline = cfg_b.train_pipeline
cfg_b.data.train.split = 'D:/Data_set/building_dataset/mask_dir/train/train_building.txt'

cfg_b.data.val.type = 'Dataset'
cfg_b.data.val.data_root = 'D:/'
cfg_b.data.val.img_dir = 'Data_set/building_dataset/img_dir/train'
cfg_b.data.val.ann_dir = 'Data_set/building_dataset/contour_ann_dir/train'
cfg_b.data.val.pipeline = cfg_b.test_pipeline
cfg_b.data.val.split = 'D:/Data_set/building_dataset/mask_dir/train/val_building.txt'

cfg_b.data.test.type = 'Dataset'
cfg_b.data.test.data_root = 'D:/'
cfg_b.data.test.img_dir = 'Data_set/building_dataset/img_dir/val'
cfg_b.data.test.ann_dir = 'Data_set/building_dataset/contour_ann_dir/val'
cfg_b.data.test.pipeline = cfg_b.test_pipeline
cfg_b.data.test.split = 'D:/Data_set/building_dataset/mask_dir/val/test.txt'

cfg_b.load_from = 'C:/Users/kkm/Downloads/mmsegmentation/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

# Set up working dir to save files and logs.
cfg_b.work_dir = 'D:/checkpoint/Segformer/building/'

cfg_b.runner.max_iteTrs = 200
cfg_b.log_config.interval = 100
cfg_b.evaluation.interval = 1000  # 모델 학습시 평가를 몇 번째 iteration마다 할 것인지 지정
cfg_b.checkpoint_config.interval = 1000  # 모델 학습시 학습한 모델을 몇 번째 iteration마다 저장할 것인지 지정

cfg_b.runner = dict(type='IterBasedRunner', max_iters=20000)  # Iteration으로 동작, Epoch로 동작하게 변경할 수도 있음
# cfg.runner = dict(type='EpochBasedRunner', max_epochs=4000)  # Epoch로 변경
cfg_b.workflow = [('train', 1)]

# Set seed to facitate reproducing the result
cfg_b.seed = 0
#set_random_seed(0, deterministic=False)
cfg_b.gpu_ids = range(1)

classes_r = ('background', 'road')
palette_r = [[0, 0, 0], [255, 127, 0]]

cfg_r = Config.fromfile(config_file)

cfg_r.norm_cfg = dict(type='BN', requires_grad=True)
#cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg_r.model.decode_head.norm_cfg = cfg_r.norm_cfg

cfg_r.device='cuda'

cfg_r.model.decode_head.num_classes = 2

#cfg.data.samples_per_gpu=2
#cfg.data.workers_per_gpu=2

cfg_r.model.decode_head.loss_decode = [dict(type='CrossEntropyLoss', loss_weight = 1.0),
                                     dict(type='FocalLoss', loss_weight = 1.0),
                                     dict(type='LovaszLoss', loss_weight = 1.0, reduction='none')]

cfg_r.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg_r.crop_size = (512, 512)
cfg_r.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg_r.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg_r.img_norm_cfg),
    dict(type='Pad', size=cfg_r.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg_r.val_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='MultiScaleFlipAug',
                        img_scale=(1024, 1024),
                        flip=False,
                        transforms=[
                                    dict(type='Resize', keep_ratio=True),
                                    dict(type='RandomFlip'),
                                    dict(
                                        type='Normalize',
                                        mean=[123.675, 116.28, 103.53],
                                        std=[58.395, 57.12, 57.375],
                                        to_rgb=True),
                                    dict(type='ImageToTensor', keys=['img']),
                                    dict(type='Collect', keys=['img'])
                                    ]),
                    
]


cfg_r.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg_r.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

cfg_r.data.samples_per_gpu=1
cfg_r.data.workers_per_gpu=0
cfg_r.data.persistent_workers=False
cfg_r.dataset_type = 'Dataset'
cfg_r.data_root = 'D:/'

cfg_r.data.train.type = 'Dataset'
cfg_r.data.train.data_root = 'D:/'
cfg_r.data.train.img_dir = 'Data_set/road_dataset/img_dir/train'
cfg_r.data.train.ann_dir = 'Data_set/road_dataset/contour_ann_dir/train'
cfg_r.data.train.pipeline = cfg_r.train_pipeline
cfg_r.data.train.split = 'D:/Data_set/road_dataset/mask_dir/train/train_road.txt'

cfg_r.data.val.type = 'Dataset'
cfg_r.data.val.data_root = 'D:/'
cfg_r.data.val.img_dir = 'Data_set/road_dataset/img_dir/train'
cfg_r.data.val.ann_dir = 'Data_set/road_dataset/contour_ann_dir/train'
cfg_r.data.val.pipeline = cfg_r.test_pipeline
cfg_r.data.val.split = 'D:/Data_set/road_dataset/mask_dir/train/val_road.txt'

cfg_r.data.test.type = 'Dataset'
cfg_r.data.test.data_root = 'D:/'
cfg_r.data.test.img_dir = 'Data_set/road_dataset/img_dir/val'
cfg_r.data.test.ann_dir = 'Data_set/road_dataset/contour_ann_dir/val'
cfg_r.data.test.pipeline = cfg_r.test_pipeline
cfg_r.data.test.split = 'D:/Data_set/road_dataset/mask_dir/val/test.txt'

cfg_r.load_from = 'C:/Users/kkm/Downloads/mmsegmentation/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

# Set up working dir to save files and logs.
cfg_r.work_dir = 'D:/checkpoint/Segformer/road/'

cfg_r.runner.max_iteTrs = 200
cfg_r.log_config.interval = 100
cfg_r.evaluation.interval = 1000  # 모델 학습시 평가를 몇 번째 iteration마다 할 것인지 지정
cfg_r.checkpoint_config.interval = 1000  # 모델 학습시 학습한 모델을 몇 번째 iteration마다 저장할 것인지 지정

cfg_r.runner = dict(type='IterBasedRunner', max_iters=20000)  # Iteration으로 동작, Epoch로 동작하게 변경할 수도 있음
# cfg.runner = dict(type='EpochBasedRunner', max_epochs=4000)  # Epoch로 변경
cfg_r.workflow = [('train', 1)]

# Set seed to facitate reproducing the result
cfg_r.seed = 0
#set_random_seed(0, deterministic=False)
cfg_r.gpu_ids = range(1)

# 실행 코드
input_dir = "D:/exp/"
output_dir = "D:/results/"
os.makedirs(output_dir, exist_ok=True)

checkpoint_file_b = "D:/checkpoint/Segformer/building/Segformer_building.pth"
checkpoint_file_r = "D:/checkpoint/Segformer/road/Segformer_road.pth"
model_ckpt_b = init_segmentor(cfg_b, checkpoint_file_b, device='cuda:0')
model_ckpt_r = init_segmentor(cfg_r, checkpoint_file_r, device='cuda:0')

image_sets = [f.split('_')[0] for f in os.listdir(input_dir) if f.endswith("_0.png")]
overlap_percentages = []

for image_set in set(image_sets):
    overlap_percentage = process_image_set(image_set, input_dir, output_dir, model_ckpt_b, model_ckpt_r)

    overlap_percentages.append(f"{image_set}: {overlap_percentage:.2f}%")

    print(f"Final result saved at: {output_dir}")

overlap_results_path = os.path.join(output_dir, "overlap_results.txt")

with open(overlap_results_path, "w") as f:
    f.write("\n".join(overlap_percentages))

print(f"Overlap percentages saved at: {overlap_results_path}")