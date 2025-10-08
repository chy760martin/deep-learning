# model_utils.py
import torch
from torch import nn
from torchvision import models

# Transfer Learning 모델 클래스 정의
class TransferLearningModel(nn.Module):
    def __init__(self, base_model, feature_extractor=True, num_classes=2):
        super().__init__()

        if feature_extractor: # - feature_extractor: True일 경우 기존 레이어는 동결하고 분류기만 학습
            for param in base_model.parameters(): # - 기존 모델의 모든 파라미터를 순회하며
                param.requires_grad = False  # 특징 추출기 동결

        # Vision Transformer의 분류기 부분 재정의
        in_features = base_model.heads.head.in_features # 기존 ViT의 에서 입력 특징 수()를 가져옴
        base_model.heads = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        nn.Linear(128, num_classes) # 클래스 수 2로 설정 (고양이, 강아지)
        )

        self.model = base_model

    def forward(self, x):
        return self.model(x)

# 모델 로딩 함수 - 모델을 불러올 때 사용
def load_model(model_path, device): # model_path: 저장된 모델 경로, device: 'cpu' 또는 'cuda'
    base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model = TransferLearningModel(base_model, feature_extractor=True, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model