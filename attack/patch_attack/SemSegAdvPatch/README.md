# SemSegAdvPatch
This is the code repository for 

* "*Evaluating the Robustness of Semantic Segmentation for Autonomous Driving against Real-World Adversarial Patch Attacks*" by Federico Nesti, Giulio Rossolini, Saasha Nair, Alessandro Biondi, and Giorgio Buttazzo. The paper was accepted for [WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Nesti_Evaluating_the_Robustness_of_Semantic_Segmentation_for_Autonomous_Driving_Against_WACV_2022_paper.pdf).[pre-print](https://arxiv.org/abs/2108.06179).

* "*On the Real-World Adversarial Robustness of Real-Time Semantic Segmentation Models for Autonomous Driving*" by Giulio Rossolini, Federico Nesti, Gianluca D'Amico, Saasha Nair, Alessandro Biondi, and Giorgio Buttazzo - (Extended journal version) - [pre-print](https://arxiv.org/abs/2108.06179)


In these papers we extensively explore the robustness of real-time Semantic Segmentation models against adversarial patches in the context of autonomous driving. We perform experiments both on Cityscapes and on datasets collected with the CARLA simulator.
We propose a novel loss function that improves attack performance. We also propose the scene-specific attack, a novel attack methodology that exploits geometrical information from the simulator to apply accurately re-projected patches onto a known, fixed attackable surface (i.e., a billboard). We found that this attack outperforms the standard Expectation-Over-Transformation attack methodology.

The networks under tests proved to be somehow robust to these kinds of attacks, hence raising interesting questions not only about the actual real-world adversarial attack effect, but also on the intrinsic robustness of semantic segmentation models.



https://user-images.githubusercontent.com/92364988/145780409-9f4582eb-1916-456f-8e80-13fbbade6778.mp4




## Code
The code is out now!
The Semantic Segmentation utilities are from https://github.com/meetps/pytorch-semseg.

#### Setup
This code was tested in a virtual environment with Python3.6, on a GPU cluster that supported CUDA>=11.1. 
You might want to check the installation requirements of your machine before proceeding.
```
pip install -r requirements-gpu1.txt
```

#### Datasets
The datasets used for this paper were:
* Cityscapes, that you can download from here: https://www.cityscapes-dataset.com/
* A few custom CARLA datasets, that we can release upon request.

#### Models
Two versions of each network were used for the experiments: the pre-trained version and the fine-tuned version on CARLA. 
The pretrained versions can be found at:
- https://github.com/ycszen/TorchSeg (BiseNet)
- https://github.com/ydhongHIT/DDRNet (DDRNet)
- https://github.com/hszhao/ICNet (ICNet)

The CARLA fine-tuned versions can be provided upon request.

#### Config files
To perform attacks, a config.yml file is required. Adapt each network's template to your own weights and dataset paths.
Also, make sure that the path in the field adv_patch/optimization/loss/NPS/argument points to a .txt listing of the printable colors (RGB tuples).
You can change the optimization/loss parameters, the patch parameters, and the path to save the experiments results.

#### Launch the attacks!
Two different attacks are provided: the EOT-based attack (untargeted_patch_attack.py) that can be performed on both Cityscapes and CARLA datasets, and the scene-specific attack, that can only be performed on CARLA datasets.
```
python untargeted_patch_attack.py --config configs/config.yml
python specific_patch_attack.py --config configs/config.yml
```

#### Citation
If you found this code useful, consider citing
```
@InProceedings{Nesti_2022_WACV,
    author    = {Nesti, Federico and Rossolini, Giulio and Nair, Saasha and Biondi, Alessandro and Buttazzo, Giorgio},
    title     = {Evaluating the Robustness of Semantic Segmentation for Autonomous Driving Against Real-World Adversarial Patch Attacks},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {2280-2289}
}

# Journal Version
@article{rossolini2022real,
  title={On the Real-World Adversarial Robustness of Real-Time Semantic Segmentation Models for Autonomous Driving},
  author={Rossolini, Giulio and Nesti, Federico and D'Amico, Gianluca and Nair, Saasha and Biondi, Alessandro and Buttazzo, Giorgio},
  journal={arXiv preprint arXiv:2201.01850},
  year={2022}
}

```

