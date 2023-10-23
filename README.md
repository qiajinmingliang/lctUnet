1. Organizing the Data Prior to Training the Model:
First, you need to decompress the dataset and place the volume data and segmentation labels into separate local folders, such as ./dataset/data and ./dataset/label.
Next, you should adjust the root path of the volume data and segmentation labels in preprocess.py. For instance:
```arduino
row_dataset_path = './dataset/'  # path of the original dataset
fixed_dataset_path = './fixed/'  # path of the fixed (preprocessed) dataset
```
After making these changes, you can run python preprocess.py. If executed correctly, you will find the following files in the directory ./fixed:
```
│  test_name_list.txt
│  train_name_list.txt
│  val_name_list.txt
│
├─data
│      volume-0.nii
│      volume-1.nii
│      volume-10.nii
│      ...
└─label
       segmentation-0.nii
       segmentation-1.nii
       segmentation-10.nii
       ...
```

2. Requirements for the Virtual Environment:
I recommend running the model within the PyTorch framework, specifically using the PyTorch 10.2 version.

3. Detailed Model Implementation:
I have graduated and am no longer engaged in related work. Unfortunately, I did not retain the code on my personal computer. However, I suggest attempting to reproduce the model based on the descriptions provided in the paper.

4. Training and Testing the Model:
Training 3DUNet:
Firstly, adjust some parameters in parameter.py. It's crucial to set --dataset_path to ./fixed. All parameters are commented within the parameter.py file.
Subsequently, run python train.py --save model_name.
Testing 3DUNet:
Firstly, adjust some parameters in parameter_test.py. It's crucial to set --dataset_path to ./fixed.
Subsequently, run python test.py --save model_name.

If you have any questions, please open an issue in the repository to ask for help or report the problem.
