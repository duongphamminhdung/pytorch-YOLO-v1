
echo installing data
# wget https://drive.google.com/file/d/1xtpLsvMBfaw2pj8s4m5lf05i4aB0jmwj/view?usp=drive_link
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

# tar xvf VOCtrainval_06-Nov-2007.tar
# tar xvf VOCtest_06-Nov-2007.tar
# tar xvf VOCdevkit_08-Jun-2007.tar
pip install gdown
gdown https://drive.google.com/uc?id=1HEIi_eo7IwCPbmAI_hwR4NlKKY7rUp3G
unzip coconut-yolo.zip

conda config --append channels pytorch
echo conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch

# install other dependancy
echo pip install scikit-image tqdm ipdb matplotlib torchnet
pip install scikit-image tqdm  ipdb matplotlib torchnet opencv-python pytorchsummary

echo Set up complete
