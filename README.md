# Deep Geometry Analysis for Multi-View 3D Human Pose Estimation
This repository is the official implementation of [Deep Geometry Analysis for Multi-View 3D Human Pose Estimation]. 
## Requirements

To install requirements:

```setup
#1. Create a conda virtual environment.
conda create -n mvhpe python=3.7.11
conda activate mvhpe

#2. Install Pytorch
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#3. Install requirements.
pip install -r requirements.txt
```

## Preparing Data and Rre-trained model
1. Download the required data.
   * Download our data from [Google Drive](https://drive.google.com/drive/folders/1Z6-fLuANi2Y67w-VZrx-oG_K9IrSINtK?usp=sharing) 
   * Download our pretrained model from [Google Drive](https://drive.google.com/drive/folders/1zxcGUvszOH2Sh1JOa_cSvHdNyRdwBwpO?usp=sharing)
   
2. You need to follow directory structure of the `data` as below.
```
|-- data
`-- |-- h36m_sub1.npz
    `-- ...
    `-- h36m_sub11.npz
    `-- score.pkl
|-- checkpoint
`-- |-- h36m_cpn_uncalibration.pth
    `-- h36m_gt_uncalibration.pth
    `-- h36m_cpn_calibration.pth
```

## Evaluation

To evaluate our model, run:

```eval
python eval_h36m_cpn_uncalibration.py --test --out_chans 3 --previous_dir ./checkpoint/h36m_cpn_uncalibration.pth --root_path /home/zzj/TMM/MV-3D-HPE/data/ --gpu 0
python eval_h36m_gt_uncalibration.py --test --in_chans 2 --previous_dir ./checkpoint/h36m_gt_uncalibration.pth --root_path /home/zzj/TMM/MV-3D-HPE/data/ --gpu 0
python eval_h36m_cpn_calibration.py --test --out_chans 2 --previous_dir ./checkpoint/h36m_cpn_calibration.pth --root_path /home/zzj/TMM/MV-3D-HPE/data/ --gpu 0
```
## Results

Our model achieves the following performance on Human3.6M:

| Methods            |Camera     |MPJPE|
| -------------------|-----------|------------|
| Ours (CPN, T=27)   |Uncalibration|     24.5mm |      
| Ours (GT, T=27)    |Uncalibration|     5.2mm  |  
| Ours (CPN, T=27)   |Calibration  |     23.8mm |  

## License
By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.
