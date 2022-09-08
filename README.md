# TouchDesigner-YOLOv7

## Prerequisite

- [TouchDesinger's Release Notes](https://docs.derivative.ca/Release_Notes#Build_2022.28040_-_Aug_29,_2022)

> Build version : 2022.28040  
TouchDesigner Python version: 3.9.5  
TouchDesigner CUDA version: 11.2  
TouchDesigner cuDNN version: 8.1.1  
> 

## CUDA, cuDNN

1. Install [CUDA Toolkit 11.2 Update 2 Downloads](https://developer.nvidia.com/cuda-11.2.2-download-archive)
2. Download [cuDNN 8.1.1 for CUDA 11.2](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse811-111)
3. Unzip cudnn-11.2-windows-x64-v8.1.1.33.zip
4. Copy files (`cudnn-11.2-windows-x64-v8.1.1.33\cuda`) to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`

## Anaconda

1. Install [Anaconda Distribution](https://www.anaconda.com/products/distribution)
2. Run “Anaconda Prompt” application
3. Create virtual environment `conda create -n td python=3.9.5`
4. Activate virtual environment `conda activate td`
5. Install onnxruntime-gpu `pip install onnxruntime-gpu`
    

## YOLOv7

- [Official Repository](https://github.com/WongKinYiu/yolov7)

## TouchDesigner

- Add External Python to Search Path    
    [Edit]→[Preferences]→[General]→[Add External Python to Search Path]  
    Set `C:/Users/USERNAME/anaconda3/envs/ENV_NAME/Lib/site-packages`  

- Code
    
    ```python
    import numpy as np
    import onnxruntime as ort
    
    class YOLOv7:
    	def __init__(self, model):
    		providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    		self.session = ort.InferenceSession(model, providers=providers)
    
    	def run(self):
    		image = op('src').numpyArray()
    		image = image[:,:,:3]
    		image = image.transpose((2,0,1))
    		image = np.expand_dims(image, 0)
    		image = np.ascontiguousarray(image)
    		
    		outputs = self.session.run(['output'], {'images':image})[0]
    
    		op('output').clear()
    		op('output').appendRows(outputs)
    		
    yolo = YOLOv7('./yolov7-tiny.onnx')
    yolo.run()
    ```
