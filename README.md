# Zoo of Fundus Disc Segmentation Methods
This repository provides a zoo of different methods for sementing the disc in retina Fundus images. The aim of this to make a consistent test environment for different methods. In addition, this environment would be ready to use easily. 

with _index.html_, you can summarize and browse all the results in a one html table. As you can see, none of the methods are perfect and also the results are so different by different methods. </br> ___Notice that, each method is trained on different datasets and so we can not report their performance as benchmark on a new dataset.___
For that reason you may need to retrain the models on your dataset. Consider their repositories for training process. 

<img src="./images_readme/index.html.png" alt="drawing" width="600"/>

## requirements
The codes are modified to be usable easily with __CPUs__ too and so you don't need to have any GPU. </br>
You just need to install the requirements, with conda or pip, either one you prefer. </br>
Use ___requirements.txt___ or ___requirements_conda.txt___ respectively. </br> 
If using the pip version, you need to install _OpenCV_ seperately. </br> 
__Notice:__ make sure the python version is 3.5 

## Usage
for each method, there is a main.py file and you can use them with appropriate arguments (path of input images and results). for example:

> python main_DENet.py --img_dir ./tmp_images  --result_dir ./results_DENet

finally, you can make a summerization of the results by:
> python main_html.py 

View the generated index.html file with a browser.

## Sources
Notice that, the collected librarris contains the required files to run each method in which copied from the source repositories. Therefore all the rights and responsibilities of the each method and corresponding library is reserved by their own developer.
See the source repositories and make sure to consider themselves too:

**DENet:** </br>
    DENet_GlaucomaScreen </br>
    https://github.com/HzFu/DENet_GlaucomaScreen </br>
    https://doi.org/10.1109/TMI.2018.2837012 </br>
    https://doi.org/10.1109/TMI.2018.2791488 </br>

**MNet:** </br>
    mnet_deep_cdr    </br>
    https://github.com/HzFu/MNet_DeepCDR </br>
    https://doi.org/10.1109/TMI.2018.2791488 </br>

**AttnUnet:** </br>
    Optic-Disc-Unet</br>
    https://github.com/DeepTrial/Optic-Disc-Unet </br>
    https://arxiv.org/pdf/1804.03999v3.pdf </br>

**Adaptive Threshold:** </br>
    Cup-and-disc-segmentation-for-glaucoma-detection-CDR-Calculation- </br>
    https://github.com/NupurBhaisare/Cup-and-disc-segmentation-for-glaucoma-detection-CDR-Calculation- </br>
    https://doi.org/10.1109/SPIN.2015.7095384  </br>

## Citing

make sure, to cite the proper method with the provided doi above. Also, for citing this repository you can use following reference:

## 
The _main_DENet_ONH.py_ is not for the zoo and you don't need it. It is a cusotmized version for our own project. 