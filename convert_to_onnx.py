import numpy as np
import torch
import alphapose
from alphapose.models import builder
from alphapose.utils.config import update_config

cfg_path = r'D:\AlphaPose-master\configs\coco\resnet\256x192_res50_lr1e-3_1x.yaml'
weight = 'fast_res50_256x192.pth'
onnx_model_name = 'alphapose.onnx'

if __name__ == "__main__":
    cfg = update_config(cfg_path)
    device = torch.device('cuda')
    input = torch.rand( 1, 3, 256, 192 ).cuda()

    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    pose_model.load_state_dict(torch.load(weight, map_location=device))
    pose_model.cuda()
    pose_model.eval()


    torch.onnx.export(pose_model,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  onnx_model_name,          # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
    print("Finish!")