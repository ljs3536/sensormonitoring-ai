# sensor-ai/preprocess.py
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

class SpectrogramTransformer:
    def __init__(self, sample_rate=1000, n_fft=64, hop_length=16, n_mels=32): 
        self.sample_rate = sample_rate
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels, # 수정된 값 적용
            power=2.0,
            normalized=True
        )
        
        # 데시벨(dB) 스케일 변환 (로그 스케일) -> 미세한 소리 변화를 극대화
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def __call__(self, waveform):
        """
        Args:
            waveform_1d: shape [128] 인 1D 입력 텐서 (V 또는 g 단위)
        Returns:
            db_spectrogram: shape [1, 64, 가로크기] 인 2D dB-Spectrogram 텐서 (이미지)
        """
        """
        waveform: 
          - Piezo일 경우: [128] 또는 [1, 128]
          - ADXL일 경우: [3, 128] (X, Y, Z)
        """
        # 1. 입력 데이터를 텐서로 변환 및 디바이스(CPU/GPU) 설정
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float)
            
        # torchaudio는 [batch, time] 형태를 기대하므로 배치 차원 추가
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0) # [1, 128]

        # 각 채널별로 스펙트로그램 생성
        # [Channels, 128] -> [Channels, n_mels, time_frames]
        # 2. Mel-Spectrogram 생성 [1, n_mels, time_frames]
        mel_spec = self.mel_spectrogram(waveform)
        # 3. 데시벨 스케일로 변환 (추론 성능 향상 핵심)
        db_spec = self.amplitude_to_db(mel_spec)
        
        # 정규화 (채널별 독립 정규화)
        # (0~1 사이 값으로 맞춰줌 -> CNN 학습 가속화)
        db_spec_norm = (db_spec - db_spec.min()) / (db_spec.max() - db_spec.min() + 1e-8)

        return db_spec_norm # 결과: [C, 64, 9] (C는 1 또는 3)

# --- 간단한 테스트 코드 ---
if __name__ == "__main__":
    # 1. 가짜 1D 진동 데이터 생성 (사인파 + 노이즈)
    dummy_signal = np.sin(np.linspace(0, 10, 128)) + np.random.normal(0, 0.1, 128)
    
    # 2. 변환기 인스턴스 생성
    transformer = SpectrogramTransformer(sample_rate=1000)
    
    # 3. 변환 실행
    spectrogram_image = transformer(dummy_signal)
    
    print(f"✅ 원본 1D 데이터 크기: {dummy_signal.shape}")
    print(f"✅ 변환된 2D Spectrogram 크기: {spectrogram_image.shape}")
    # 아마 [1, 64, 9] 정도의 크기가 나올 겁니다. (n_mels=64, 128샘플을 hop 16으로 나눔)