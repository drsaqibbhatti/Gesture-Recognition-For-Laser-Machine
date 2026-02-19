import torch
from Model.Mobile3DV2 import Mobile3DV2

# Configuration
model_path = "/home/saqib/Projects/action_recognition/action_recognition_git/actionclassification/trained/ToF_Cam/run_10/Best_E_21_Acc_0.9982_loss_0.0081_W320_H240_D16.pt"
num_classes = 8  # Replace with the number of classes in your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = Mobile3DV2(num_classes=num_classes, width_mult=0.5).to(device)

# Load the checkpoint and handle "module." prefix
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
# Remove "module." from keys if it exists
state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

# Save the entire model
model_save_path = "/home/saqib/Projects/action_recognition/action_recognition_git/actionclassification/trained/ToF_Cam/run_10/Best_E_21_Acc_0.9982_loss_0.0081_W320_H240_D16_No_DP.pt"
torch.save(model, model_save_path)
print(f"Entire model saved to {model_save_path}")

# Save the state_dict
state_dict_save_path = "/home/saqib/Projects/action_recognition/action_recognition_git/actionclassification/trained/ToF_Cam/run_10/Best_E_21_Acc_0.9982_loss_0.0081_W320_H240_D16_No_DP_State_dict.pt"
torch.save(model.state_dict(), state_dict_save_path)
print(f"State_dict saved to {state_dict_save_path}")
