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

