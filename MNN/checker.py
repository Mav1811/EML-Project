import onnx
from onnx import helper


model = onnx.load("mnist_cnn.onnx")
for i, node in enumerate(model.graph.node):
    if node.op_type == "Clip":
        print(i, node.input, node.attribute)
        print(i, node)
