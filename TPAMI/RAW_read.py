import numpy as np
import cv2

raw_path = r"./adr__frame_prod.mo5-mt.0085_cam.mid_center_top_tele_bayer.0_1740153836014635327.raw"
H, W = 2160, 3840

raw = np.fromfile(raw_path, dtype=np.uint16).reshape(H, W)

# 移除低4位垃圾位
raw = raw >> 4

# BGGR bayer阵列 demosaic
rgb = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2RGB)

# 12bit 归一化（黑电平已减去）
rgb = rgb.astype(np.float32) / 4095.0







