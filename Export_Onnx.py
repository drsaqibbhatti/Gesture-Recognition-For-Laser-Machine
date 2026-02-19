
import torch
import io
import base64
import json
import onnx.helper
import onnx.numpy_helper

# Model configuration
frameWidth=320
frameHeight=240
depth=16

model_path = "D:\hvs\\Hyvsion_Projects\\Actions_project\\action_recognition_git\\actionclassification\\trained\\Tof_Cam\\run_10\\Best_E_21_Acc_0.9982_loss_0.0081_W320_H240_D16_No_DP.pt"
onnx_model_path = "D:\hvs\\Hyvsion_Projects\\Actions_project\\action_recognition_git\\actionclassification\\trained\\Tof_Cam\\run_10\\GestureToF20250515_W320_H240_D16.onnx"
onnx_model_path_SM = "D:\hvs\\Hyvsion_Projects\\Actions_project\\action_recognition_git\\actionclassification\\trained\\Tof_Cam\\run_10\\GestureToF20250515_W320_H240_D16_SoftMax.onnx"

#dsm_model_path = "D:\hvs\\Hyvsion_Projects\\Actions_project\\action_recognition_git\\actionclassification\\trained\\run_9\\Best_E_36_Acc_0.9938_loss_0.0327_W320_H240_D16_No_DP.dsm"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the full model
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# Dummy input for ONNX export
trace_input = torch.randn(1, 1, depth, frameHeight, frameWidth).to(device)

# Export to ONNX
buffer = io.BytesIO()
#compiled_model = torch.jit.script(model)
torch.onnx.export(
    model,
    trace_input,
    buffer,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['logits']
)

with open(onnx_model_path, "wb") as f:
    f.write(buffer.getvalue())


onnx_model = onnx.load(onnx_model_path)

onnx.checker.check_model(onnx_model)

softmax_node = onnx.helper.make_node(
    'Softmax',
    inputs=['logits'], 
    outputs=['output'],
    axis=1 
)

onnx_model.graph.node.append(softmax_node)

onnx_model.graph.output[0].name = "output"

onnx.save(onnx_model, onnx_model_path_SM)

# #keep_initializers_as_inputs= True,    
# # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
# # Convert buffer to base64 for DSM format
# buffer.seek(0)
# bytes_data = buffer.read()
# base64Model = base64.b64encode(bytes_data).decode('ascii')

# # Create custom DSM JSON structure
# hvs_model_json = {
#     "inputHeight": frameHeight,
#     "inputWidth": frameWidth,

#     "module": base64Model,

#     "classNames": ["pinhole"]
# }

# # Save DSM format
# jsonFormatString = json.dumps(hvs_model_json)
# with open(dsm_model_path, "w") as text_file:
#     text_file.write(jsonFormatString)

# # Save the ONNX model to disk
# with open(onnx_model_path, "wb") as f:
#     f.write(bytes_data)

print(f"ONNX model exported to {onnx_model_path}")
#print(f"DSM format saved to {dsm_model_path}")
