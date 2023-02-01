# Table Service Detection Module

This is an implementation of the Table Service Detection module.
This module is a part of [Cloud Robot Project](https://github.com/aai4r/aai4r-master) and the updated version of the previous [Service Context Understanding module](https://github.com/aai4r/aai4r-ServiceContextUnderstanding).

The module has three parts, an object detector, a table status classifier, and a table service detector.
The object detector is modified based on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Environment
* python 3.7
* pytorch 1.5.1
* pytorchvision 0.6.1
* (All included in the env.yaml)

### Installation
1. Clone this repository.
    ```bash
    git clone https://github.com/aai4r/aai4r-TableServiceDetection
    cd aai4r-TableServiceDetection
    ```

2. Install required modules
    ```bash
    conda env create -f env.yaml
    ```

3. Download a [checkpoint.pth]() file.

   
### Run
Run the demo code with the sample images.
   ```bash
   python table_service_alarm_interface.py
   ```
   
