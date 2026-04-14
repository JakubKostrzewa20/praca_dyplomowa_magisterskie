import timm
import torch

model = timm.create_model("efficientnet_b0", pretrained=True)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "input_data/models/efficientnet/efficientnet_b0.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18
)