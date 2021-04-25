# Zoo of Fundus Disc Segmentation Methods
This repository provides a zoo of different methods for sementing the disc in retina Fundus images. The aim of this to make a consistent test environment for different methods. In addition, this environment would be ready to use easily. 

[index.html](https://htmlpreview.github.io/?https://github.com/mohaEs/Zoo-Fundus-Disc-Segmentation/blob/main/index_4_GitHub.html) shows the results on a short test set. As you can see, the results are so different by different methods. </br> Notice that, each method is trained on different datasets and so we can not report their performance on a new dataset as a benchmark.
For that, reason you may need to retrain them on your dataset. Consider their repositories for training process. 

## requirements
The codes are modified to be used easily with CPUs too and you don't need to have GPU. </br>
You just need to install the requirements, with conda or pip, either you prefer. </br>
Use _requirements.txt_ or _requirements_conda.txt_ respectively.

## Usage

## 


## Sources
Notice that, the collected librarris folder, contains the required files to run each method in which copied from the source repositories. Therefore all rights and responsibilities of the each method and corresponding library is reserved by their own developer.

See the sources repositories and make sure yo consider themselves too:

**DENet:** </br>
    DENet_GlaucomaScreen </br>
    https://github.com/HzFu/DENet_GlaucomaScreen
    https://doi.org/10.1109/TMI.2018.2837012
    https://doi.org/10.1109/TMI.2018.2791488

**MNet:** </br>
    mnet_deep_cdr    </br>
    https://github.com/HzFu/MNet_DeepCDR
    https://doi.org/10.1109/TMI.2018.2791488

**AttnUnet:** </br>
    Optic-Disc-Unet</br>
    https://github.com/DeepTrial/Optic-Disc-Unet
    https://arxiv.org/pdf/1804.03999v3.pdf

**Adaptive Threshold:** </br>
    Cup-and-disc-segmentation-for-glaucoma-detection-CDR-Calculation- </br>
    https://github.com/NupurBhaisare/Cup-and-disc-segmentation-for-glaucoma-detection-CDR-Calculation-
    https://doi.org/10.1109/SPIN.2015.7095384 

## Citing

make sure, to cite the proper method with the provided doi above. Also, for citing this repository you can use following reference:

