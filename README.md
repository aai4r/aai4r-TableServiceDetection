# Table Service Detection Module

This is an implementation of the Table Service Detection module.
This module is a part of [the Cloud Robot Project](https://github.com/aai4r/aai4r-master) and the updated version of the previous [Service Context Understanding module](https://github.com/aai4r/aai4r-ServiceContextUnderstanding).

The module has three parts, an object detector, a table status classifier, and a table service detector.
The object detector is modified based on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Environment
* python 3.7
* pytorch 1.5.1
* pytorchvision 0.6.1

### Installation
1. Clone this repository.
    ```bash
    git clone https://github.com/aai4r/aai4r-TableServiceDetection
    cd aai4r-ServiceContextUnderstanding
    ```

2. Install required modules
    ```bash
    pip install pretrainedmodels
    pip install opencv-python
    pip install numpy
    pip install imageio
    ```

3. Download [all weight files (detection and classification)](https://drive.google.com/drive/folders/1rT2DYaiywGt8gqdl2YGnd6RLP1rxZV9I?usp=sharing).

   
### Run
Run the demo code with the sample images.
   ```bash
   python table_service_alarm_interface.py
   ```
   
