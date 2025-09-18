# 🥇ACK2025_Visual_Localization

---
### Image Segmentation과 특징점 매칭 알고리즘을 활용한 Vision-Based-Localization 시스템 구현

GNSS/INS를 활용한 항법 시스템은 무인항공기의 위치 추정에 있어 매우 중요한 요소이다. 
그러나 GNSS/INS 시스템은 두 센서의 상호보완 시스템이므로 하나의 센서에 오차가 발생할 시, 위치 좌표를 신뢰하기 어렵다. 
따라서 GNSS/INS 항법 시스템 고장에도 독립적으로 위치 추정이 가능한 시스템이 필요하다. 
본 논문은 이미지 데이터를 활용한 항법 시스템을 제안한다. 항공 이미지에서 Segmentation 모델로 건물과 도로를 검출한 뒤, 
특징점 매칭 알고리즘을 적용하여 위치 좌표를 추정한다. 이는 무인 항공기의 GNSS 통신이 원활하지 않은 상황에서도 위치 추정을 가능케하며,
Visual Navigation 시스템에 활용하여 무인 항공기의 자율 비행 성능 개선에 기여할 수 있다.

---

# 1. 데이터 셋
- AI Hub (aihub.or.kr)의 "위성영상 객체판독" 이미지 데이터

  - Building: 1393장 (.PNG)

  - Road: 1271장 (.PNG)

# 2. 시스템 프로세스
![Localization System Process](https://github.com/seoljaehun/ACK2024_Visual_Localization/blob/main/Image_Data/Localization%20System%20Process.PNG)

+ **Map Data와 UAV Image 준비**

   
   - Map Data Image : Google Earth 고도 1.9km 상공 위성 이미지
   - UAV Image : AI Hub 위성 이미지

+ **Image Segmentation** 

   - Segmentation 모델을 활용해 Map Data, UAV Image에서 Building과 Road 검출

+ **Localization**

   - 특징점 매칭 알고리즘을 활용해 Map Data와 UAV Image에서 일치하는 부분을 현재 위치로 추정

# 3. 시스템 구현

**1. Image Segmentation**

3개의 Segmentation 모델 성능 비교 및 최적 모델 선정

- YOLOv11-Seg : YOLO 아키텍쳐를 기반으로, 단일 순전파 과정에서 객체의 Bounding Box와 Instance Mask를 통합적으로 예측하는 분할 모델
- DeepLabV3 : 다양한 비율의 Atrous Convolusion을 적용하여 고밀도의 특징맵을 유지하고, 이를 ASPP 모듈로 결합하여 강건한 다중 스케일 문맥 정보를 포착하는 분할 모델
- SegFormer : 계층적 구조의 Hierarchical Transformer 인코더를 통해 다중 스케일 특징을 추출하고, 이를 Light-Weight MLP 디코더와 결합하여 높은 성능과 효율성을 보여주는 분할 모델

![Segmentation Result](https://github.com/seoljaehun/ACK2024_Visual_Localization/blob/main/Image_Data/Segmentation%20Result.PNG)

성능 평가 결과, SegFormer 모델이 가장 우수한 지표를 보여 최종 모델로 선정

**2. 특징점 매칭 알고리즘**

Map Data와 UAV Image에서 추출된 Building, Road Segmentation 이미지를 비교하여 최종 현재 위치 추정

- 새로운 관심영역 ROI 생성 : Mean-Shift Clustering 알고리즘 적용

![ROI](https://github.com/seoljaehun/ACK2024_Visual_Localization/blob/main/Image_Data/ROI.PNG)

- 클러스터의 경계 생성: 밀도 기반 클러스터링(DBSCAN) 알고리즘 적용

![Clustering](https://github.com/seoljaehun/ACK2024_Visual_Localization/blob/main/Image_Data/Clustering.PNG)

**3. Localization**

- 현재 위치 추정: 가장 큰 클러스터 4개를 선택한 뒤, 그 중심좌표 도출

# 4. 실험 결과

- Image Segmentation 모델과 특징점 매칭 알고리즘을 결합한 방식이 평균 75.60%의 높은 정확도 달성

![Localization Result](https://github.com/seoljaehun/ACK2024_Visual_Localization/blob/main/Image_Data/Localization%20Result.PNG)

---

## 관련 자료

- Paper: <https://www.koreascience.kr/article/CFKO202520961205149.page?&lang=ko>
- Dataset: <https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EC%9C%84%EC%84%B1%EC%98%81%EC%83%81%20%EA%B0%9D%EC%B2%B4%20%ED%8C%90%EB%8F%85&aihubDataSe=data&dataSetSn=73>
- 참고문헌: <https://github.com/seoljaehun/ACK2024_Visual_Localization/blob/main/Reference/%EC%B0%B8%EA%B3%A0%EB%AC%B8%ED%97%8C>
