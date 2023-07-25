# Pedestrian Crossing Intention
Prediction of the intention of pedestrians to cross the street or not, using Graph Neural Networks and the coordinates of their skeleton. We made our own dataset of crossing and not-crossing scenarios using CARLA simulator.

# Spatial-Temporal Graph Convolutional Network

I initially designed a first spatial model using the GCNConv graph convolutional operator, with an input size equal to 3, which are the 3 features of each node (the 3 coordinate axes of each joint). However, the results only improved a bit, and training it more produced a high overfitting.

To get good results I had to design a Spatial-Temporal model. What I did was, for a set of *n* frames with their *n* skeletons, predict if in a future frame the pedestrian will perform the action of crossing or not. That is, from the movements and trajectory of the pedestrian during *n* frames (grouped using a sliding window), predict whether he or she will cross the street or not in the near future.

The layers of the model are detailed in the following table:

|     Layer                     |     Input shape    |     Output shape    |
|:------------------------------|:-------------------|:--------------------|
|     Recurrent part:           |     -              |     -               |
|      \|--> GConvGRU           |     [-1, 26, 3]    |     [-1, 26, 3]     |
|      \|--> Dropout   (0.3)    |     [-1, 26, 3]    |     [-1, 26, 3]     |
|      \|--> ReLU               |     [-1, 26, 3]    |     [-1, 26, 3]     |
|     End of recurrent part     |     -              |     -               |
|     Reshape                   |     [-1, 26, 3]    |     [-1, 78]        |
|     Linear                    |     [-1, 78]       |     [-1, 39]        |
|     Dropout (0.3)             |     [-1, 39]       |     [-1, 39]        |
|     ReLU                      |     [-1, 39]       |     [-1, 39]        |
|     Linear                    |     [-1, 39]       |     [-1, 19]        |
|     Dropout (0.3)             |     [-1, 19]       |     [-1, 19]        |
|     ReLU                      |     [-1, 19]       |     [-1, 19]        |
|     Linear                    |     [-1, 19]       |     [-1, 2]         |
|     Softmax                   |     [-1, 2]        |     [-1, 2]         |

The input to the network contains, for each frame, 26*3 elements because in CARLA there are 26 different joints, and we input the 3D coordinates of them.

# Results

## CARLA dataset

Final results when classifying our CARLA dataset of pedestrian skeletons using the Spatial-Temporal Graph Convolutional Network that uses 5 frames to define the temporal dimension. 

### Metrics

|            Metric |  train |    val |   test |
|:------------------|:------:|:------:|:-------|
|          Accuracy | 0.9171 | 0.8194 | 0.8834 |
| Balanced accuracy | 0.9177 | 0.8183 | 0.8857 |
|         Precision | 0.9334 | 0.8103 | 0.9207 |
|            Recall | 0.9078 | 0.8014 | 0.8638 |
|          f1-score | 0.9204 | 0.8058 | 0.8913 |

### Videos

#### Train videos

![2-1-0-0.gif](Videos_results/CARLA/train/2-1-0-0.gif)

![7-8-0-0.gif](Videos_results/CARLA/train/7-8-0-0.gif)

![10-3-0-0.gif](Videos_results/CARLA/train/10-3-0-0.gif)

![18-4-0-0.gif](Videos_results/CARLA/train/18-4-0-0.gif)

![23-10-0-0.gif](Videos_results/CARLA/train/23-10-0-0.gif)

![32-12-0-0.gif](Videos_results/CARLA/train/32-12-0-0.gif)


#### Test videos

![49-2-0-0.gif](Videos_results/CARLA/test/49-2-0-0.gif)

![12-4-0-0.gif](Videos_results/CARLA/test/12-4-0-0.gif)

![49-15-0-0.gif](Videos_results/CARLA/test/49-15-0-0.gif)

![23-4-0-0.gif](Videos_results/CARLA/test/23-4-0-0.gif)


## JAAD dataset

Final results when classifying JAAD dataset of pedestrian skeletons using the Spatial-Temporal Graph Convolutional Network that uses 87 frames to define the temporal dimension. 

### Videos

#### Train videos

![video_0003.gif](Videos_results/JAAD/AlphaPose/train/video_0003.gif)

![video_0011.gif](Videos_results/JAAD/AlphaPose/train/video_0011.gif)


#### Test videos

![video_0046.gif](Videos_results/JAAD/AlphaPose/test/video_0046.gif)

![video_0053.gif](Videos_results/JAAD/AlphaPose/test/video_0053.gif)