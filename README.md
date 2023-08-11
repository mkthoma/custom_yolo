# Custom YOLOv3 on Pascal VOC 
This repo contains code that was adapted from [another library](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3). A few changes are made with respect to the code:
1. The entire module is converted into PyTorch Lightning
2. Best LR is found using the [Tuner](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#using-lightning-s-built-in-lr-finder) functionality. 
3. One Cycle Policy is introduced as a scheduler.
4. Since One Cycle Policy is used, the total number of epochs used is only 40% of the original.
5. Mosaic transformation applied to 75% of train images

The live application on HuggingFace Spaces can be found [here](https://huggingface.co/spaces/mkthoma/custom_yolo_v3).

## Dataset
Pascal VOC (Visual Object Classes) is a collection of standardized image datasets for object class recognition. Initiated from 2005 to 2012, this dataset has become one of the most widely used benchmarks for evaluating the performance of different algorithms for object detection and image segmentation.

Pascal VOC has been instrumental in the development and evaluation of many state-of-the-art computer vision algorithms, especially before the dominance of the COCO dataset. Many deep learning models for object detection, like Faster R-CNN, SSD, and YOLO, have been trained and evaluated on Pascal VOC, providing a common ground for comparison.

As the field of computer vision evolved, datasets with larger numbers of images and more diverse annotations became necessary. This led to the development of more comprehensive datasets like MS COCO. Due to its limited size and diversity, Pascal VOC has become less dominant in recent years, but it still remains an important benchmark in the history of object detection.

## YOLOv3
YOLOv3 is the third version of the YOLO architecture, a state-of-the-art, real-time object detection system. YOLO, as the name suggests, processes images in one pass, making it incredibly fast while maintaining a balance with accuracy.

### Key Features:
1. Single Shot Detector: Unlike two-step detectors which first identify regions of interest and then classify those regions, YOLO performs both tasks in a single pass, making it faster.

2. Darknet-53: YOLOv3 introduces a new 53-layer architecture called Darknet-53, which is a hybrid of the Darknet architecture and some characteristics of the more complex architectures like ResNet.

3. Three Scales: YOLOv3 makes detections at three different scales by using three different sizes of anchor boxes. This helps in capturing objects of different sizes more accurately.

4. Bounding Box Predictions: Instead of predicting the coordinates for the bounding boxes, YOLOv3 predicts the offsets from a set of anchor boxes. This helps in stabilizing the model's predictions.

5. Multi-label Classification: Unlike YOLOv2 which used Softmax, YOLOv3 uses independent logistic classifiers to determine the probability of the object's presence, allowing the detection of multiple object classes in one bounding box.

6. Loss Function: The loss function in YOLOv3 is designed in a way that it treats object detection as a regression problem rather than a classification problem. This approach is more suitable for single-shot detection.

7. Use of Three Anchor Boxes: For each grid cell, it uses three anchor boxes (pre-determined shapes). This helps the network adjust its predictions to the shape of objects.

While YOLOv3 is not the most accurate object detection algorithm, its strength lies in its speed, making it suitable for real-time applications. When compared to its predecessors, YOLOv3 offers a good balance between speed and accuracy. It performs particularly well in detecting smaller objects due to its multi-scale predictions.

## Mosaic Transformation
Mosaic augmentation is a data augmentation technique commonly used in computer vision tasks, especially for object detection. It was introduced in the YOLOv4 (You Only Look Once version 4) paper and has since become quite popular due to its effectiveness in improving detection performance.

The basic idea behind mosaic augmentation is to combine four different training images into a single mosaic image. This process not only increases the variety of objects and backgrounds in a single image but also helps the model learn to detect objects at different scales and orientations.

Here's how mosaic augmentation is typically performed:

- Random Selection: Randomly select four images from the training dataset.
- Determine a Split Point: Randomly select a point within the dimensions of the image. This point will determine the boundary for the four images.
- Place Images:
    - Place the first image in the top-left corner and resize it such that the split point divides it.
    - Place the second image in the top-right corner and resize it accordingly.
    - Place the third image in the bottom-left corner.
    - Place the fourth image in the bottom-right corner.
- Output Mosaic Image: The resulting image will have objects from all four images, possibly at different scales and orientations.

Benefits of Mosaic Augmentation:
- Diverse Data: Mosaic augmentation increases the diversity of the training data without actually adding new images. This can help in better generalization.
- Multiple Scales: Since the original images are resized to fit the mosaic, objects can appear at different scales, teaching the model to recognize objects of varying sizes.
- Improved Detection Performance: As shown in the YOLOv4 paper, using mosaic augmentation can lead to better performance on object detection tasks.

It's worth noting that while mosaic augmentation can be beneficial, it's just one of many data augmentation techniques available. Depending on the specific application and dataset, other augmentation techniques might also be beneficial.

![image](https://github.com/mkthoma/custom_yolo/assets/135134412/ad9aab24-e63b-4aea-9ded-33bef564aeb8)

## Max LR using LR_find

The max LR was found using the [inbuilt functionality](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#using-lightning-s-built-in-lr-finder) of lightning module.

```python
# Initialize the model
model = YOLOv3Lightning(config=config)

total_epochs = int(config.NUM_EPOCHS * 0.4)

trainer = pl.Trainer(precision=16, max_epochs=total_epochs,
                              callbacks=[ModelCheckpoint(dirpath=config.CHECKPOINT_PATH,verbose=True,),
                              class_accuracy_callback(train_epoch_interval=1, test_epoch_interval=10),
                              plot_examples_callback(epoch_interval=10),
                              map_callback(epoch_interval=total_epochs),
                              LearningRateMonitor()])

from lightning.pytorch.tuner import Tuner

# Create a Tuner
tuner = Tuner(trainer)

# Finding the learning rate
lr_finder =tuner.lr_find(model)

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

# update hparams of the model
model.hparams.learning_rate = new_lr
```
![image](https://github.com/mkthoma/era_v1/assets/135134412/ca4cbb68-a681-4629-aeb4-6dc45d586af8)

## Model Metrics

#### Class Accuracy

This refers to the accuracy of the predicted class labels for the detected objects. For instance, if a system is trained to detect dogs and cats, and it correctly identifies 9 dogs and 8 cats out of 10 each, then its class accuracy is $(9 + 8)/20 =85$%

#### No Object Accuracy:

In object detection, an image is usually divided into a grid, and each grid cell is responsible for predicting objects. However, not all grid cells will contain an object. The "no object accuracy" metric measures how accurately the model predicts that there is no object in a given grid cell. It's particularly important in scenarios where false positives (wrongfully detecting an object) can be problematic.

#### Object Accuracy:

This is the opposite of "no object accuracy." It measures how accurately the model predicts the presence of an object in a given grid cell.

#### Mean Average Precision
MAP (mean Average Precision) is a key metric used in the evaluation of object detection models, and it's especially relevant for models like YOLO (You Only Look Once). MAP provides a single figure of merit that summarizes the performance of a detector across all classes and various overlap thresholds. It can be broken down into:
1. Precision and Recall:
    - Precision is the fraction of detected instances that are true positives (i.e., correctly detected)
    - Recall is the fraction of actual instances that are detected. 
2. Average Precision (AP):
    - For each class, you plot the precision-recall curve. Then, the area under this curve is computed, which gives the Average Precision (AP) for that class. The curve is obtained by varying the confidence threshold for detections.
    - The AP is a single value that summarizes the trade-off between precision and recall for different threshold values.
3. Mean Average Precision (mAP):
    - Once you have the AP for all classes, you compute the mean of these AP values to get the MAP. This gives an overall performance metric for the detector across all classes.

For some tasks, especially in older datasets like PASCAL VOC, the MAP is computed at a specific IoU (Intersection over Union) threshold, often 0.5. For more recent datasets like COCO, MAP is usually averaged over multiple IoU thresholds (e.g., from 0.5 to 0.95 with step size 0.05), making it a more robust metric against different overlap requirements.

It's worth noting that while MAP is a valuable metric, it's not the only one to consider when evaluating object detectors in real-world scenarios. Depending on the application, other metrics like runtime, memory usage, or robustness to different conditions might be equally or even more important.

We shall look at some of the metrics related to the model while training it. Initially we can see that the model has poor performance but starts to rapidly  increase from the $3^{rd}$ epoch for the next few epochs and then it starts to gradually climb.

![image](https://github.com/mkthoma/era_v1/assets/135134412/8152b77c-f43b-48dd-b275-2d5085da1ff3)
![image](https://github.com/mkthoma/era_v1/assets/135134412/f827e7af-78d0-4f02-99c5-69e57a1b3cff)

Towards the end of the maximum epochs we have set we can see that the model is performing well and has achieved a decent accuracy for class, no object and object detection. We have achieved 85% class accuracy, 98% no object accuracy and 78% object accuracy. The loss has also reduced from 20 ($1^{st}$ epoch) to 3.7 in the last epoch

![image](https://github.com/mkthoma/era_v1/assets/135134412/60dfe9e1-56f6-4e6a-8ef4-e9cc0902b4fe)

We have achieved a Mean Average Precision of 0.4, caculated at the last epoch.

![image](https://github.com/mkthoma/era_v1/assets/135134412/bf2c0702-e8cb-49f8-ab2b-6c3830431b57)

### Tensorboard Outputs

![image](https://github.com/mkthoma/era_v1/assets/135134412/0375f43d-5dd8-412b-ad3b-64fc263ddac0)

![image](https://github.com/mkthoma/era_v1/assets/135134412/0375f43d-5dd8-412b-ad3b-64fc263ddac0)

![image](https://github.com/mkthoma/era_v1/assets/135134412/78817002-22a5-49e6-b08d-01656f6c8576)

![image](https://github.com/mkthoma/era_v1/assets/135134412/69c17d93-2446-4a08-9118-c02e4c7cee67)


Sample output:

![image](https://github.com/mkthoma/era_v1/assets/135134412/67ae45dd-9b47-49fc-9431-1ceab6313870)

![image](https://github.com/mkthoma/era_v1/assets/135134412/a3a472fe-ed5f-4afa-b91c-56e7bfdf16df)

The model was run on a A100 Colab instance and took about 4 hours to run.

## Conclusion

In our recent experiment, we employed YOLOv3, a state-of-the-art object detection model, to the renowned PASCAL VOC dataset using the PyTorch Lightning framework. PyTorch Lightning facilitated a cleaner and more maintainable codebase, allowing us to focus on the model's performance and optimizations rather than boilerplate training loops. Notably, the integration of mosaic transformations, a data augmentation technique designed to enhance the diversity of training samples by stitching four training images together, demonstrated a promising enhancement in the model's robustness.

The results indicated that the combination of YOLOv3 with mosaic transformations not only improved the detection accuracy across various object categories in the dataset but also enhanced the model's generalization capabilities. This experiment underscores the significance of effective data augmentation techniques like mosaic transformations in elevating the performance of object detection models. It also reaffirms the utility of PyTorch Lightning in streamlining deep learning workflows, especially for complex architectures like YOLOv3.
