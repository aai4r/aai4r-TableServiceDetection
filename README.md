# Table Service Detection Module

This is an implementation of the Table Service Detection module.
This module is a part of [Cloud Robot Project](https://github.com/aai4r/aai4r-master) and the updated version of the previous [Service Context Understanding module](https://github.com/aai4r/aai4r-ServiceContextUnderstanding).

The module has three parts, an object detector, a table status classifier, and a table service detector.
The object detector is modified based on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Installation
0. Requirements
   * Linux, CUDA>=9.2, GCC>=5.4
   * Python>=3.7


1. Clone this repository.
    ```bash
    git clone https://github.com/aai4r/aai4r-TableServiceDetection
    cd aai4r-TableServiceDetection
    ```

2. Install required modules.
    ```bash
    pip install torch==1.5.1 torchvision==0.6.1
    pip install -r requirements.txt
    pip install pycocotools
    pip install tqdm
    pip install cython
    pip install scipy
    ```

3. Compiling CUDA operators in [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
    ```bash
    cd ./models/ops
    sh ./make.sh
    # unit test (should see all checking is True)
    python test.py
    ```

4. Download a [checkpoint.pth](https://drive.google.com/file/d/1L4JduDlczm5M2tj_LSswGN2WSbYPNBaR/view?usp=sharing) file.

   
### Run
Run the demo code with the sample image (that is in the python code).
   ```bash
   python table_service_alarm_interface.py
   ```
A detailed explanation is written in the sample code.
