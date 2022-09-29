# PatchAttackTool
Welcome to our repo for the code for attacks against autonomous driving vision tasks. This code was used to craft the adversarial patches used in our papers "[CARLA-GeAR: a Dataset Generator for a Systematic Evaluation of Adversarial Robustness of Vision Models](https://carlagear.retis.santannapisa.it/)" [1] and "[On the Real-World Adversarial Robustness of Real-Time Semantic Segmentation Models for Autonomous Driving](https://arxiv.org/abs/2201.01850)" [2]. 

This code constitutes a generalization of our repository https://github.com/retis-ai/SemSegAdvPatch to extend [scene-specific attacks](https://openaccess.thecvf.com/content/WACV2022/html/Nesti_Evaluating_the_Robustness_of_Semantic_Segmentation_for_Autonomous_Driving_Against_WACV_2022_paper.html) to tasks other than **semantic segmentation**, namely **2d object detection**, **monocular depth estimation** and **stereo 3d object detection**.

![image](https://user-images.githubusercontent.com/92364988/173370023-ade7e6cf-dec2-4c75-9a1f-f4ca4405c9fe.png)

Please visit https://carlagear.retis.santannapisa.it/#research for a list of our papers concerning real-world adversarial attacks.

If our work was useful for your research, please consider citing it!
```
@ARTICLE{2022arXiv220604365N,
       author = {{Nesti}, Federico and {Rossolini}, Giulio and {D'Amico}, Gianluca and {Biondi}, Alessandro and {Buttazzo}, Giorgio},
        title = "{CARLA-GeAR: a Dataset Generator for a Systematic Evaluation of Adversarial Robustness of Vision Models}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2022,
        month = jun,
          eid = {arXiv:2206.04365},
        pages = {arXiv:2206.04365},
archivePrefix = {arXiv},
       eprint = {2206.04365},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220604365N%7D,
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
```
@ARTICLE{2022arXiv220101850R,
    author = {{Rossolini}, Giulio and {Nesti}, Federico and {D'Amico}, Gianluca and {Nair}, Saasha and {Biondi}, Alessandro and {Buttazzo}, Giorgio},
    title = "{On the Real-World Adversarial Robustness of Real-Time Semantic Segmentation Models for Autonomous Driving}",
    journal = {arXiv e-prints},
    keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Artificial Intelligence},
    year = 2022,
    month = jan,
    eid = {arXiv:2201.01850},
    pages = {arXiv:2201.01850},
    archivePrefix = {arXiv},
    eprint = {2201.01850},
    primaryClass = {cs.CV},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220101850R},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```



### Installation
We tested the code with Python 3.6.9. It is strongly suggested to use a virtual environment.
Install PyTorch 1.10.1 with CUDA 11.1 support (https://pytorch.org/get-started/previous-versions/ - any CUDA version should work).

Then, install the requirements: `pip install -r requirements.txt`

You should be all set!
If you want to use the same models we used, checkout the Datasets and models section.

### Tasks
To keep the structure general, each task has its own Task Interface object that handles the task-specific optimization and evaluation steps. 
Also, each attack is task-specific and defined in its own class. Models, losses, metrics, and utilities are in the task-specific libraries  `ptsemseg`, `ptod`, `ptdepth`, and `pt3dod`.

This generalization helps maintain the same structure for each task, as well as the possibility to extend the optimization to any user-defined attack of model.

### Datasets and models
This repo is intended to work with CARLA-generated datasets as the ones that you can find [here](https://carlagear.retis.santannapisa.it/#datasets).
This is because the scene-specific attack requires exact camera-to-billboard roto-translation information for accurate patch placing.

The models used for the generation of the patches included in these datasets are in the following table:
| Network  | Link to repo |
| ------------- | ------------- |
| DDRNet23Slim  | https://github.com/ydhongHIT/DDRNet  |
| BiseNetX39  | https://github.com/ycszen/TorchSeg  |
| Faster R-CNN  | [torchvision zoo](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)  |
| RetinaNet  | [torchvision zoo](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)  |
| GLPDepth  | https://github.com/vinvino02/GLPDepth  |
| AdaBins  | https://github.com/shariqfarooq123/AdaBins  |
| Stereo R-CNN*  | https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN/tree/1.0  |

\* Differently from the other models that only require the weights to be downloaded, Stereo R-CNN requires the entire repo to be cloned and installed in `pt3dod/models`. Please follow the repo instructions, also for the installation of the evaluation script.

### Running the code
To evaluate the performance of a network, it is sufficient to run 
```python eval_script.py --config path/to/config.yml```, while for the optimization you must run ```python optimization_script.py --config path/to/config.yml```.

The `.yml` config file (of which we shared a template for each network used) is the same for optimization and evaluation. In the latter case, the part dedicated to the optimization is ignored (but results are saved anyway --- so double check the experiment folder to make sure you don't overwrite anything important).


### Extending attacks and models
You can extend the code to include:
* user-defined models: it is sufficient to copy the model .py file in the corresponding task-specific model library (e.g., for a segmentation CNN, in `ptsemgseg/models`), add the initialization line in the `models/__init__.py`, and add the pytorch weights loading function in `utils.py`. All the other network-specific configuration parameters are in the config file.
* user-defined attacks: it is sufficient to define a new loss in the corresponding attack class file (`attacks/` folder).

If you have any questions feel free to contact us!



Code is coming soon.
