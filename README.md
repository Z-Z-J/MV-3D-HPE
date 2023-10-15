# Deep Geometry Analysis for Multi-View 3D Human Pose Estimation
This repository is the official implementation of [Deep Geometry Analysis for Multi-View 3D Human Pose Estimation]. 
## Requirements

To install requirements:

```setup
#1. Create a conda virtual environment.
conda create -n mvhpe python=3.6.10
conda activate mvhpe

#2. Install requirements.
pip install -r requirements.txt
```

## Preparing Data and Rre-trained model
1. Download the required data.
   * Download our smpliks_db from [Google Drive]() 
   * Download our smpliks_data from [Google Drive]()
   * Download our pretrained model from [Google Drive]()
   
2. You need to follow directory structure of the `data` as below.
```
|-- data
`-- |-- amass_train_db.pt
    `-- amss_test_db.pt
    `-- 3dpw_test_db.pt
    `-- agora_test_db.pt
```
## Evaluation

To evaluate our model, run:

```eval
python eval_si_apr_hybrik.py --cfg configs/config_eval.yaml
python eval_si_apr_analyik.py --cfg config/config_eval.yaml
```
## Results

Our model achieves the following performance on Human3.6M:

| Methods            |Camera     |MPJPE|
| -------------------|-----------|------------|
| Ours (CPN, T=27)   |Uncalibration|     -    |      
| Ours (GT, T=27)    |Uncalibration|     -    |  
| Ours (CPN, T=27)   |Calibration  |     -    |  

## License
By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.
