# Thesis: Continual Learning and Catastrophic Forgetting in Composites Production
This is the repo for the master thesis at ITA - RWTH Aachen

Please note, that the code is still heavy work in progress and thus might not be fully operational <br />
Some preliminary experiments have been conducted with Graph Neural Networks and Superpixel Image Segmentation, this code can be found on branch "graphs"

## Abstract
Major breakthroughs in machine learning have led to its wide-spread use in various fields of industry, including automotive and retail, showing (super-) human performance in prediction tasks on structured data like image classification. Composites production is an especially promising field of application, as it remains dominated by human labor and a strong demand for expert-level human domain knowledge, therefore benefiting strongly from process automation.
In general, Convolutional Neural Networks (CNN) are used for image classification tasks. CNN require huge amounts of human-labelled data to reach a reliable prediction accuracy, assuming that the underlying data distribution is stationary and identically and independently distributed. All class labels must be known in advance and represented in the training dataset for the model to work as intended. <br />

However, real-world data becomes available incrementally from non-stationary data-distributions. This leads to a phenomenon called catastrophic forgetting – experience from new tasks interferes with or overwrites previously learned knowledge stored in the model weights. Prediction accuracy is severely impeded, posing an unresolved challenge in machine learning. Whereas most models are trained in a supervised way, i.e., requiring annotated data, self-supervised learning aims to render this costly human annotation step obsolete by learning features and representations from unlabelled data. Thus, making deep learning feasible for applications where labelled data is either unavailable or prohibitively expensive, such as in production environments. <br />

A model architecture for continual self-supervised learning is still missing, esp. with applications in the fields of composites production and defect detection.
Thus, the goal of this work is to develop a deep learning model that exhibits superior performance in multi-class classification for composite production compared to previous model architectures. Performance metrics include total prediction accuracy (+0.5 %) and a forgetting measure, evaluated after adding new tasks or classes to the continually presented data stream. <br />

Potential model structures are developed and juxtaposed according to state-of-the-art research findings, fulfilling the below stated desiderata:<br />
•	Continual learning capability, i.e., Task Incremental Learning and Class Incremental Learning<br />
•	Support of unsupervised, self-supervised or semi-supervised training settings <br />

The most promising model architecture is then chosen and empirically-iteratively refined. The proposed model is benchmarked against a conventional CNN-classifier in supervised, as well as self-supervised training regime with access to all training data at once, acting as the upper bound. A comparison to selected state-of-the-art models for continual learning settings is performed, using a variety of classes of fiber lay-up images. <br />

## General approach
The general idea focusses on combining self-supervised learning approaches, e.g., the recently published VICReg-Approach with Prototypical losses.
The prototypes of previous classes act like a regularization against Catastrophic forgetting and allow the model keep learning continuously.
Incorporation of VICReg allows for semi-supervised training settings and for learning richer representations from the underlying data. <br />
![image](https://user-images.githubusercontent.com/96831420/167251601-67681478-3400-42bc-a453-060f3a6dc912.png)

## Dataset
The dataset consists of a total of 8 eight classes of fiber lay-up images. A total of two materials (NCF and Woven), comprised of one immaculate class and three defective classes: <br/>
![image](https://user-images.githubusercontent.com/96831420/181204963-1bd3764a-54f6-40d9-afac-26fa79f11ce7.png) <br/>
The dataset is always presented in the same order, as seen above. Each class consists of approx. 1000 sample images.
Each experience consists of three classes incrementing forward by a step of one. <br />

![image](https://user-images.githubusercontent.com/96831420/181208151-152f2dfe-bf3f-4402-b3a7-51cf8176068e.png)


## Loss function
The new method is coined ProReC (Prototypical Regularization for Continual Learning)
The loss function consists initially of five terms, three distance metrics and two information maximization terms: <br/>
![image](https://user-images.githubusercontent.com/96831420/181211092-d6920f4e-7f03-45c3-b769-640854586ad9.png)
<br/>
The invariance term i(P) aims at increasing the distance (here: Euclidean) of the prototypes in the feature space. <br/>
![image](https://user-images.githubusercontent.com/96831420/181211251-76fe9ed5-7d00-427b-966d-22eed17fcddf.png)
<br/>

The covariance term c(P) aims at decorrelating the variables in each prototype to maximize information (see VICReg paper by Bardes, Ponce, LeCun 2022): <br/>
![image](https://user-images.githubusercontent.com/96831420/181211338-ab6854c9-e982-487a-ae11-0b83db176e1e.png)
<br/>
The variance term v(Z) aims at maintaining a minimum variance along the batch-dimension of the prototypes: <br/>
![image](https://user-images.githubusercontent.com/96831420/181211391-c3bb4c7b-afc2-4d79-9e15-6d50e9402d14.png)
<br/>
The cluster cohesion term aims at decreasing the distance inside a given cluster:<br/>
![image](https://user-images.githubusercontent.com/96831420/181211485-92582d12-e9ff-48c0-bb5a-e9c1b508bbad.png)
<br/>

The prototype regularization term aims at regularizing changes in prototypes from one epoch to the next: <br/>
![image](https://user-images.githubusercontent.com/96831420/181211533-a35ea904-1a1b-4419-a2a5-db9f113698c3.png)
<br/>

Our research has shown that the covariance and variance term are sufficient to reach competitive classification results.
All other terms can be discarded.

## Pseudo Code
![image](https://user-images.githubusercontent.com/96831420/181211917-f10aa5d5-6074-4406-a41e-e2754b870d61.png) <br/>
![image](https://user-images.githubusercontent.com/96831420/181211953-6057c233-e635-4cd5-9f43-75be01568ba9.png)
<br/>


## Results
The model is evaluated on different metrics, including Accuracy and Forgetting, where Forgetting is defined as the maximum Accuracy on a given class at a certain point minus the current Accuracy (normalized by the maximum Accuracy). 
The A-Metric and F-Metric average over Accuracies and Forgetting values at the end of all experiences respectively.
MS describes Model Storage efficiency and is given by: <br/>
![image](https://user-images.githubusercontent.com/96831420/181212270-5499f229-ef0c-460e-a2df-a86bdd7b507d.png)
<br/>
SSS describes Sample Storage Size efficiency and is given by: </br>
![image](https://user-images.githubusercontent.com/96831420/181212328-7b804eef-48d8-4345-90ea-404a3a0d9b47.png)
<br/>
The model achieves the following results:

![image](https://user-images.githubusercontent.com/96831420/181208248-14c6d343-f24b-41e7-8b66-5cc9648d3ed3.png)

Feature Storage describes a model that is storing the intermediate feature vectors as extracted from the ResNet18 backbone.
Here, 2048 samples are stored.
At best, a final Accuracy of 50 % (37 %) can be reached, limiting Forgetting to 47 % (100 %) for the ProReC model (Vanilla).

The model is able to remember classes better, with minimal storage size. The class accuracy is shown below:
![image](https://user-images.githubusercontent.com/96831420/181208718-c46e3223-5012-4daa-95c8-a90c38f4eff8.png)
<br/>
## Further research

An interesting emerging propoerty is the finding that the model was able to separate the feature vectors in the representation space by material.
This happened in a fully unsupervised way and could be replicated for different orders of class presentation. 
After all, the information maximization terms might be useful in combining supervised and unsupervised information extraction and learning.

![image](https://user-images.githubusercontent.com/96831420/181209452-e78b10fe-9639-4a79-8843-e5de25c9c730.png)


## Usage
```
### Create List of Dataloaders like so (makes use of ImageDatasetTrain and ImageDatasetTest in ./thesis/helper/dataset.py)
dataloader_list, dataloader_test_list = create_dataloader(
    root_dir = "your_image_directory" # root_dir contains folders with images; each foldername is the class name
    num_classes = 8, 
    cls_per_run = 3, 
    batch_size = 128,
    drop_last = True
    )
### Instantiate model
resnet18_model = torchvision.models.resnet18(pretrained = True, progress = True, zero_init_residual=True)

### Define run_name and save to your private wandb backend
run_name = "test_method"

### Instantiate train dict
train_dict = {"model":resnet18_model,
              "epochs":131,
              "epoch_list": [30, 50, 70, 90, 110, 130],
              "dataloader_list":dataloader_list,
              "dataloader_test_list":dataloader_test_list,
              "loss_fn":"proto_vicreg",
              "proto_feature_size":512,
              "proj_hidden_size":512,
              "sim_loss_weight":0., #49.,
              "cov_loss_weight":32., #32.,
              "var_loss_weight":42., #42.,
              "dist_loss_weight":0., # 81.
              "proto_reg_loss_weight":0., #5.,
              "proj_layers":1,
              "lr":0.0,
              "lr_proj":0.0003,
              "total_classes":8,
              "cls_per_run":3,
              "max_storage_per_cls":0, #4096
              "new_samples_batch":0, #256
              "store":"random",
              "gamma":0.9,
              }
entity = "ME"
project = "MYPROJECT"

run = wandb.init(project = project, entity = entity, config = train_dict)
wandb.run.name = run_name

### Run Training
loss_list, test_acc_lists, forgetting_m_list, forgetting_per_cls_list, model, protos = resnet18_training(
    path = "my_path_to_save_to, 
    run_name = run_name, 
    **train_dict
    )
```
