# model_utils.py
import torch
from torch import nn
from torchvision import models

# Transfer Learning 모델 클래스 정의
class TransferLearningModel(nn.Module):
    def __init__(self, base_model, feature_extractor, num_classes=4):
        super().__init__()
        # ViT는 encoder.layer.N 형태로 여러 Transformer 블록을 포함하고 있으며 예를 들어 아래처럼 일부만 동결
        # - 예시) encoder.layer.0~3까지 freeze → 일반적인 특징 유지
        # - 예시) encoder.layer.4~11은 fine-tune → 데이터셋에 맞게 조정
        if(feature_extractor):
            for name, param in base_model.named_parameters():
                # if 'encoder.layer.0' in name or 'encoder.layer.1' in name: # Vision Transformer 기반 구조 
                if 'layer1' in name or 'layer2' in name or 'conv1' in name: # ResNet 기반 구조
                    param.requires_grad = False  # 초기 레이어 동결
                else:
                    param.requires_grad = True   # 나머지 레이어는 학습
                    
        # base_model.heads = nn.Sequential(
        #     nn.Linear(base_model.heads[0].in_features, 256), # pretrained_model.heads를 새로 정의 → 기존 MLP Head를 커스터마이징하여 2개 클래스 분류에 맞춤
        #     nn.ReLU(), # 활성화 함수
        #     nn.Dropout(p=0.5), # 드롭아웃
        #     nn.Linear(256, 64), # 은닉층
        #     nn.ReLU(), 
        #     nn.Dropout(p=0.5),
        #     nn.Linear(64, len(train_dataset.classes)) # 최종 출력층 - 4가지 brain tumor 분류(classification)
        # )
        base_model.fc = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 256), # pretrained_model.fc를 새로 정의 → 기존 MLP Head를 커스터마이징하여 2개 클래스 분류에 맞춤
            nn.ReLU(), # 활성화 함수
            nn.Dropout(p=0.5), # 드롭아웃
            nn.Linear(256, 64), # 은닉층
            nn.ReLU(), 
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes) # 최종 출력층 - 4가지 brain tumor 분류(classification)
        )
        self.model = base_model
    
    def forward(self, x):
        x = self.model(x) # - 사전학습된 모델의 순전파 메서드 호출
        return x

# 모델 로딩 함수 - 모델을 불러올 때 사용
def load_model(model_path, device): # model_path: 저장된 모델 경로, device: 'cpu' 또는 'cuda'
    # base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # ResNet 모델을 사전학습 가중치로 불러옴.
    model = TransferLearningModel(base_model, feature_extractor=True, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model