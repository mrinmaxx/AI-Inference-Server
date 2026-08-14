#!/usr/bin/env python3
"""Export torchvision ResNet-50 to resnet50.onnx with a FLEXIBLE batch axis.

The dynamic_axes on dim 0 is what makes batching possible: without it the model
is locked to batch size 1 and the batcher has nothing to push against. Weights
are random (weights=None) on purpose -- this is a throughput project, not an
accuracy project, so downloading pretrained weights would be wasted bandwidth.

Usage:  python3 scripts/export_resnet50.py [output_path]
"""
import os
import sys
import torch
import torchvision

out = sys.argv[1] if len(sys.argv) > 1 else "resnet50.onnx"
model = torchvision.models.resnet50(weights=None).eval()
dummy = torch.randn(1, 3, 224, 224)  # sample input; dim 0 is made dynamic below
torch.onnx.export(
    model, dummy, out,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},  # <-- flexible batch
    opset_version=13,
)

# torch 2.x may spill the ~100 MB of weights into a sidecar "<out>.data"
# (external data). Consolidate everything back into ONE self-contained .onnx so
# the file is portable and can't be separated from its weights by accident.
import onnx
m = onnx.load(out)  # load_external_data=True by default -> pulls weights in
# Pin the ONNX IR version to 9. Recent onnx packages stamp IR v10, which older
# ONNX Runtime builds (<= 1.17) reject with "Unsupported model IR version: 10".
# IR 9 is valid for this opset-13 model and loads on both old and new runtimes.
m.ir_version = 9
onnx.save_model(m, out, save_as_external_data=False)
if os.path.exists(out + ".data"):
    os.remove(out + ".data")
print(f"wrote self-contained {out} ({os.path.getsize(out)/1e6:.0f} MB), "
      f"dynamic batch axis: input [batch,3,224,224]")
