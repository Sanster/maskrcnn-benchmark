# MaskRCNN



In side this branch of original maskrcnn-benmark, I am focus on exporting trained model to onnx, so that I can inference this onnx model with TensorRT or ONNXRuntime.

**note**: this master branch does not support training and inference now! it's only used for exporting model to onnx!

Currently supported:

- [x] exporting model to fasterrcnn worked.
- [ ] exporting model to maskrcnn not worked for now.



## Changelog

**2019.09.11**: Just fix a bug of pytorch 1.3, the warning still exists. needs to supress the warning, the byte check has been changed from uint8 to bool inside pytorch1.3;


## Export ONNX

to export onnx, there mainly 2 issue to solve:

1. apex issue, you gonna need:

   ```
   git clone https://github.com/ptrblck/apex.git
   git checkout apex_no_distributed
   ```

   otherwise you will got `torch.distributed.deprecated` error message if you installed pytorch 1.3.

then, one can using:

```
python3 export_to_onnx.py
```

to get a onnx model.



## Inference ONNX

to inference on exported onnx, you can checkout `onnx_deploy` project for details.
