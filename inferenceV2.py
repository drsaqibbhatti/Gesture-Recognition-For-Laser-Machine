import CubeEye as cu
import torch
import cv2
import numpy as np
import ctypes
from PIL import Image
from torchvision import transforms
from Model.Mobile3DV2 import Mobile3DV2
from Model.SoftmaxModel import SoftmaxModel
from collections import deque
import time

# ----------------- Config ----------------- #
labels = ["NoGesture", "DoOtherThings", "StopMachine", "ResumeMachine", "PDLC_UP", "PDLC_DOWN", "LightON", "LightOFF"]
imageHeight, imageWidth = 480, 640
cropHeight, cropWidth = 400, 560
resizeWidth, resizeHeight = 320, 240
minThreshold=200
maxThreshold=2000
frameNumber = 16
classNum = len(labels)
scoreTh = 99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "D:/hvs/Hyvsion_Projects/Actions_project/action_recognition_git/actionclassification/trained/Tof_Cam/run_7/Best_E_26_Acc_0.9991_loss_0.0058_W320_H240_D16_No_DP_State_dict.pt"

# ----------------- Transform ----------------- #
transformProcess = transforms.Compose([
    transforms.Resize((imageHeight, imageWidth)),
    transforms.CenterCrop((cropHeight, cropWidth)),
    transforms.Resize((resizeHeight, resizeWidth)),
    transforms.ToTensor()
])

# ----------------- Model Load ----------------- #
model = Mobile3DV2(num_classes=classNum, width_mult=0.5).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
softmax_model = SoftmaxModel(backbone=model).to(device)
softmax_model.eval()

# ----------------- Frame Queue ----------------- #
depth_frames = deque(maxlen=frameNumber)
tempResult = ""
tempScore = 0.0

# ----------------- CubeEye Sink ----------------- #
class _CubeEyePythonSink(cu.Sink):
    def __init__(self):
        super().__init__()
        self.last_inference_time = time.time()

    def name(self):
        return "_CubeEyePythonSink"

    def onCubeEyeCameraState(self, name, serial_number, uri, state):
        print(f"[Camera State] {name} ({serial_number}) → State: {state}")

    def onCubeEyeCameraError(self, name, serial_number, uri, error):
        print(f"[Camera Error] {name} ({serial_number}) → Error: {error}")
    
    def onCubeEyeFrameList(self, name, serial_number, uri, frames):
        global depth_frames, tempResult, tempScore
        for frame in frames:
            if frame.isBasicFrame() and frame.frameType() == cu.FrameType_Depth and frame.dataType() == cu.DataType_U16:
                u16_frame = cu.frame_cast_basic16u(frame)
                ptr = ctypes.c_uint16 * u16_frame.dataSize()
                ptr = ptr.from_address(int(u16_frame.dataPtr()))
                raw_np = np.ctypeslib.as_array(ptr).reshape(frame.height(), frame.width())
                #raw_8bit = np.uint8(raw_np / 256)
                
                raw_np[(raw_np < minThreshold) | (raw_np > maxThreshold)] = maxThreshold
                
                raw_norm = raw_np.astype(np.float32) / 4095.0 # Normalize to [0, 1] range
                image = Image.fromarray(raw_norm, mode='F')  # 'F' = 32-bit float grayscale

                tensor = transformProcess(image).unsqueeze(0)  # [1, H, W]
                depth_frames.append(tensor)

                if len(depth_frames) == frameNumber:
                    clip = torch.cat(list(depth_frames)).unsqueeze(0).to(device)  # [1, 16, H, W]
                    clip = clip.permute(0, 2, 1, 3, 4)  # [1, 1, 16, H, W]

                    with torch.no_grad():
                        output = softmax_model(clip)
                        output[0][0] = 0
                        output[0][1] = 0
                        score = output.max().item() * 100
                        index = output.argmax().item()

                        if score > scoreTh:
                            tempResult = labels[index]
                            tempScore = score
                            print(f"[Prediction] {tempResult} ({tempScore:.2f}%)")

# ----------------- Camera Setup ----------------- #
source_list = cu.search_camera_source()
if not source_list or source_list.size() == 0:
    print("No CubeEye camera found.")
    exit(1)

camera = cu.create_camera(source_list[0])
sink = _CubeEyePythonSink()
camera.addSink(sink)

if camera.prepare() != cu.Result_Success:
    print("Camera preparation failed.")
    exit(1)

if camera.run(6) != cu.Result_Success:
    print("Camera run failed.")
    exit(1)

# ----------------- Display ----------------- #
cv2.namedWindow("ToF Depth", cv2.WINDOW_AUTOSIZE)
try:
    while True:
        if len(depth_frames) > 0:
            # latest_frame = depth_frames[-1].squeeze().cpu().numpy() * 255  # [H, W]
            # latest_frame = np.uint8(latest_frame)
            
            ##########Normalizing image for display####################################
            latest_frame = depth_frames[-1].squeeze().cpu().numpy()  # Still in [0, 1] range
            latest_frame = cv2.normalize(latest_frame, None, 0, 255, cv2.NORM_MINMAX)
            latest_frame = latest_frame.astype(np.uint8)
            ##############################################

            cv2.putText(latest_frame, f"{tempResult} {tempScore:.1f}%", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
            cv2.imshow("ToF Depth", latest_frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

finally:
    print("Cleaning up camera...")
    camera.stop()
    cu.destroy_camera(camera)
    cv2.destroyAllWindows()
