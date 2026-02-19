import CubeEye as cu
import onnxruntime as ort
import cv2
import numpy as np
import ctypes
from PIL import Image
from torchvision import transforms
from collections import deque
import time
import torch

import torch
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

print("ONNXRuntime Version:", ort.__version__)
print("Available providers:", ort.get_available_providers())


# ----------------- Config ----------------- #
labels = ["NoGesture", "DoOtherThings", "StopMachine", "ResumeMachine", "PDLC_UP", "PDLC_DOWN", "LightON", "LightOFF"]
imageHeight, imageWidth = 480, 640
cropHeight, cropWidth = 400, 560
resizeWidth, resizeHeight = 320, 240
minThreshold = 200
maxThreshold = 2000
frameNumber = 16
classNum = len(labels)
scoreTh = 99
onnx_model_path = "D:/hvs/Hyvsion_Projects/Actions_project/action_recognition_git/actionclassification/trained/Tof_Cam/run_7/GestureToF20250513_W320_H240_D16_SoftMax_OP17.onnx"

# ----------------- Transform ----------------- #
transformProcess = transforms.Compose([
    transforms.Resize((imageHeight, imageWidth)),
    transforms.CenterCrop((cropHeight, cropWidth)),
    transforms.Resize((resizeHeight, resizeWidth)),
    transforms.ToTensor()
])

# ----------------- ONNX Runtime Session ----------------- #
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

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

                raw_np[(raw_np < minThreshold) | (raw_np > maxThreshold)] = maxThreshold
                raw_norm = raw_np.astype(np.float32) / 4095.0  # Normalize to [0, 1]
                image = Image.fromarray(raw_norm, mode='F')  # 32-bit float grayscale

                tensor = transformProcess(image).unsqueeze(0)  # [1, H, W]
                depth_frames.append(tensor)

                if len(depth_frames) == frameNumber:
                    clip = torch.cat(list(depth_frames)).unsqueeze(0)  # [1, 16, H, W]
                    clip = clip.permute(0, 2, 1, 3, 4)  # [1, 1, 16, H, W]
                    clip_np = clip.cpu().numpy()

                    # ONNX Inference
                    output = ort_session.run([output_name], {input_name: clip_np})[0]
                    output[0][0] = 0  # Skip NoGesture
                    output[0][1] = 0  # Skip DoOtherThings

                    score = output.max() * 100
                    index = output.argmax()

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
            latest_frame = depth_frames[-1].squeeze().cpu().numpy()
            latest_frame = cv2.normalize(latest_frame, None, 0, 255, cv2.NORM_MINMAX)
            latest_frame = latest_frame.astype(np.uint8)

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
