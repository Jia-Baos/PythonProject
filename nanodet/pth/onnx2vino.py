from openvino.runtime import Core
from openvino.runtime import serialize
 
ie = Core()
onnx_model_path = r"./pth/nanodet.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
# compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
serialize(model=model_onnx, xml_path="./pth/nanodet.xml", bin_path="./pth/nanodet.bin", version="UNSPECIFIED")