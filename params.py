
#NOTE: this params.py is similar to Google Yamnet model params
# For more information: https://github.com/tensorflow/models/blob/d9541052aaf6fdc015c8150cf6576a2da68771f7/research/audioset/yamnet/params.py
# The following hyperparameters (except patch_hop_seconds) were used to train YAMNet,
# so expect some variability in performance if you change these. The patch hop can
# be changed arbitrarily: a smaller hop should give you more patches from the same
# clip and possibly better performance at a larger computational cost.
from dataclasses import dataclass
@dataclass(frozen=True)  # Instances of this class are immutable.
class Params:
  sample_rate: float = 16000.0
  stft_window_seconds: float = 0.025
  stft_hop_seconds: float = 0.010
  mel_bands: int = 32
  mel_min_hz: float = 80.0# 125.0
  mel_max_hz: float = 7600.0# 7500.0
  log_offset: float = 0.001
  patch_window_seconds: float = 0.96# 8640/16000 #1.0# 0.96
  patch_hop_seconds: float = 0.48# 8640/(2*16000) #0.5# 0.48

  @property
  def patch_frames(self):
    return int(round(self.patch_window_seconds / self.stft_hop_seconds))

  @property
  def patch_bands(self):
    return self.mel_bands

  num_classes: int = 521
  conv_padding: str = 'same'
  batchnorm_center: bool = True
  batchnorm_scale: bool = False
  batchnorm_epsilon: float = 1e-4
  classifier_activation: str = 'sigmoid'

  tflite_compatible: bool = True
