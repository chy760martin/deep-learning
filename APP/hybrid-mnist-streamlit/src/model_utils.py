# model_utils.py
import torch
from torch import nn
from torchvision import models

# 하이브리드 모델 설계 - CNN → RNN 조합으로 이미지 내 시퀀스적 패턴 학습 가능
class HybridModel_CNN_RNN(nn.Module): # 입력: (B, 1, 28, 28) — MNIST 흑백 이미지
    def __init__(self, in_channels=1, rnn_hidden=128, num_classes=10, dropout_rate=0.3):
        super(HybridModel_CNN_RNN, self).__init__()

        # CNN 부분
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1), # (B, 32, 28, 28)
            nn.ReLU(), # 비선형성 확보
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 64, 7, 7)

            nn.Dropout(dropout_rate) # 과적합 방지
        )
        
        # RNN 부분: CNN 출력 feature map을 시퀀스로 변환
        self.rnn_input_size = 64 # feature dimension
        self.rnn_seq_len = 7 * 7 # 시퀀스 길이
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size, # 입력: (B, 49, 64)
            hidden_size=rnn_hidden, # 출력: (B, 49, 128)
            num_layers=2, # 2-layer LSTM으로 깊은 시퀀스 학습
            dropout=dropout_rate, # 과적합 방지
            batch_first=True,
            bidirectional=False
        )
        
        # 최종 분류기
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden, 64), # LSTM 출력(128) -> 64 차원으로 축소
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes) # 10개 클래스 분류
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN 통과 - 2개의 Conv + ReLU + MaxPool 블록으로 28×28 → 7×7로 다운샘플링
        x = self.cnn(x)  # (batch_size, 64, 7, 7)

        # CNN 출력 -> 시퀀스 형태로 변환
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch_size, 7, 7, 64), 안전하게 메모리 연속성 확보
        x = x.view(batch_size, self.rnn_seq_len, self.rnn_input_size) # (batch_size, seq_len=49, features=64)
        
        # RNN 통과 - CNN 출력(64×7×7)을 flatten 후 LSTM에 시퀀스 길이 1로 입력
        x, _ = self.rnn(x) # (batch_size, seq_len=49, rnn_hidden=128)
        x = x[:, -1, :]  # 마지막 시퀀스 출력 사용 (batch_size, rnn_hidden=128)
        
        # 최종 분류 - LSTM의 마지막 hidden state를 받아 10개 클래스 분류
        x = self.fc(x) # (batch_size, num_classes=10)
        return x