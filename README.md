# 이미지 분류 모델 성능 비교 분석: ResNet50 vs DenseNet121

## 프로젝트 개요

본 프로젝트는 서로 다른 두 개의 CNN 아키텍처(ResNet50, DenseNet121)를 사용하여 두 개의 이미지 데이터셋(ani, CUB200)에 대한 분류 성능을 비교 분석합니다.

## 데이터셋 정보

1. **ani 데이터셋**: 동물 이미지 분류를 위한 데이터셋

   - 클래스: 다양한 동물 종류에 따른 분류

2. **CUB200 데이터셋**: 200종의 새 이미지 분류를 위한 데이터셋
   - 클래스: 200종의 새 종류에 따른 분류

## 실험 구성

총 4가지 실험을 구성하여 모델과 데이터셋 간의 성능 차이를 비교합니다:

1. ani 데이터셋 + ResNet50
2. CUB200 데이터셋 + ResNet50
3. ani 데이터셋 + DenseNet121
4. CUB200 데이터셋 + DenseNet121

## 구현 파일

프로젝트는 각 실험에 대한 독립적인 파이썬 스크립트로 구성됩니다:

1. `01_train_ani_resnet50.py`: ani 데이터셋에 ResNet50 모델 학습
2. `02_train_cub200_resnet50.py`: CUB200 데이터셋에 ResNet50 모델 학습
3. `03_train_ani_densenet121.py`: ani 데이터셋에 DenseNet121 모델 학습
4. `04_train_cub200_densenet121.py`: CUB200 데이터셋에 DenseNet121 모델 학습

## 구현 방법론

### 1. 데이터 전처리

- PyTorch의 `transforms` 모듈을 사용하여 이미지 크기 조정, 정규화, 데이터 증강 적용
- `ImageFolder` 클래스를 활용하여 폴더 구조 기반의 데이터셋 로드
- `DataLoader`를 통한 미니배치 처리 및 셔플링

### 2. 모델 구현

- PyTorch의 `torchvision.models`에서 사전 학습된 ResNet50 및 DenseNet121 모델 로드
- 각 데이터셋의 클래스 수에 맞게 모델의 최종 분류층 조정
- 전이학습(Transfer Learning) 기법 적용: 사전 학습된 가중치 활용

### 3. 학습 과정

- 손실 함수: Cross-Entropy Loss
- 최적화 알고리즘: Adam
- 학습률(Learning Rate): 0.001
- 에폭(Epochs): 30
- 배치 크기(Batch Size): 32
- 검증: 각 에폭마다 검증 데이터셋으로 성능 평가

### 4. 성능 평가

- 정확도(Accuracy): 올바르게 분류된 이미지의 비율
- 손실(Loss): 모델의 예측과 실제 레이블 간의 오차
- 학습 및 검증 정확도/손실 그래프를 통한 과적합 모니터링

## 결과 저장

학습 결과는 그래프 형태로 저장합니다:

- **학습 그래프**
  - 경로: `/content/drive/MyDrive/AI/results/`
  - 파일명 예시: `resnet50_ani_plot.png`, `densenet121_cub200_plot.png`
  - 내용: 에폭별 학습/검증 정확도 및 손실 그래프

## 실행 방법

1. Google Colab 환경에서 각 스크립트 실행
2. 필요한 경우 Google Drive 마운트 코드 실행
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. 필요한 디렉토리가 없는 경우 자동으로 생성
   ```python
   import os
   os.makedirs('/content/drive/MyDrive/AI/results/', exist_ok=True)
   ```

## 기대 효과 및 분석 포인트

1. 서로 다른 CNN 아키텍처(ResNet50 vs DenseNet121)의 성능 비교
2. 서로 다른 데이터셋(ani vs CUB200)에 대한 모델 적응성 평가
3. 모델 복잡성과 학습 시간 대비 성능 효율성 평가
4. 전이학습의 효과 및 한계점 분석

## 향후 확장 가능성

1. 더 다양한 CNN 아키텍처(VGG, EfficientNet 등) 비교
2. 하이퍼파라미터 튜닝을 통한 성능 최적화
3. 설명 가능한 AI 기법 적용을 통한 모델 해석
4. 데이터셋 통합 실험을 통한 데이터 확장 효과 분석
