import numpy as np
import random
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class SkeletonsDataset(Dataset):

    def __init__(self, csv_path, numberOfJoints=25, normalization='minmax', norm_precomputed_values=None, target='cross', transform=None):

        self.data=[]
        self.csv_path = csv_path
        self.normalization = normalization
        self.norm_precomputed_values = norm_precomputed_values
        self.targetName = target
        
        
        if numberOfJoints == 25:
            
            self.body_parts = {
                "Nose": 0,
                "Neck": 1,
                "RShoulder": 2,
                "RElbow": 3,
                "RWrist": 4,
                "LShoulder": 5,
                "LElbow": 6,
                "LWrist": 7,
                "MidHip": 8,
                "RHip": 9,
                "RKnee": 10,
                "RAnkle": 11,
                "LHip": 12,
                "LKnee": 13,
                "LAnkle": 14,
                "REye": 15,
                "LEye": 16,
                "REar": 17,
                "LEar": 18,
                "LBigToe": 19,
                "LSmallToe": 20,
                "LHeel": 21,
                "RBigToe": 22,
                "RSmallToe": 23,
                "RHeel": 24,
            }
            
            
            self.pose_parts = [
                ["Nose", "Neck"],
                ["Neck", "RShoulder"],
                ["Neck", "LShoulder"],
                ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"],
                ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"],
                ["Neck", "MidHip"],
                ["MidHip", "RHip"],
                ["RHip", "RKnee"],
                ["RKnee", "RAnkle"],
                ["RAnkle", "RHeel"],
                ["RHeel", "RBigToe"],
                ["RBigToe", "RSmallToe"],
                ["MidHip", "LHip"],
                ["LHip", "LKnee"],
                ["LKnee", "LAnkle"],
                ["LAnkle", "LHeel"],
                ["LHeel", "LBigToe"],
                ["LBigToe", "LSmallToe"],
                ["Nose", "REye"],
                ["REye", "REar"],
                ["Nose", "LEye"],
                ["LEye", "LEar"],
            ]            
            
        else:
    
            self.body_parts = {
                "Nose": 0,
                "Neck": 1,
                "RShoulder": 2,
                "RElbow": 3,
                "RWrist": 4,
                "LShoulder": 5,
                "LElbow": 6,
                "LWrist": 7,
                "RHip": 8,
                "RKnee": 9,
                "RAnkle": 10,
                "LHip": 11,
                "LKnee": 12,
                "LAnkle": 13,
                "REye": 14,
                "LEye": 15,
                "REar": 16,
                "LEar": 17,
            }
        
        

            self.pose_parts = [
                ["Neck", "RShoulder"],
                ["Neck", "LShoulder"],
                ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"],
                ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"],
                ["Neck", "RHip"],
                ["RHip", "RKnee"],
                ["RKnee", "RAnkle"],
                ["Neck", "LHip"],
                ["LHip", "LKnee"],
                ["LKnee", "LAnkle"],
                ["Neck", "Nose"],
                ["Nose", "REye"],
                ["REye", "REar"],
                ["Nose", "LEye"],
                ["LEye", "LEar"],
            ]


        self.generateEdges()

        self.loadCSV()
        self.processData()


    def __len__(self):
        return len(self.data)

    def loadCSV(self):
        self.loadedData = pd.read_csv(self.csv_path)


    def generateEdges(self):

        directed = False

        numberOfEdges = len(self.pose_parts)

        if not directed:
            numberOfEdges = numberOfEdges * 2

        # Pytorch COO format:
        # 2 rows and each column is an edge from node_i to node_j
        self.edgeindex = np.zeros(shape=(2, numberOfEdges))

        part = 0
        rev = False
        for col in range(0, numberOfEdges):

            edge = self.pose_parts[part]

            node1 = edge[0]
            node2 = edge[1]

            node1_part = self.body_parts[node1]
            node2_part = self.body_parts[node2]

            self.edgeindex[1 if rev else 0][col] = node1_part
            self.edgeindex[0 if rev else 1][col] = node2_part

            part = part + 1
            if part == len(self.pose_parts):
                part = 0
                rev = True


    def processData(self):
        
        # Drop all samples in which the skeleton was not detected on the previous pipeline step
        self.loadedData = self.loadedData[self.loadedData['skeleton_detected']==True]

        # Preprocess the target of the network:
        
        crossing = self.loadedData[self.targetName].to_numpy()
        
        if self.targetName == 'cross':
            target_nocross = np.where(crossing=='crossing', 0, 1)
            target_cross = np.where(crossing=='crossing', 1, 0)
        else:
            #if self.targetName == 'crossing':
            #    self.loadedData = self.loadedData[self.loadedData['crossing']!=-1]
            
            target_nocross = np.where(crossing==1, 0, 1)
            target_cross = np.where(crossing==1, 1, 0)
        

        target = np.stack([target_nocross, target_cross], axis=1)
        
        
        # Preprocess the input of the network:
        
        skeletonData = self.loadedData['skeleton'].tolist()

        y_values = []
        x_values = []
        label_values = []

        # For each skeleton of the dataset
        for i, skeleton in enumerate(skeletonData):
            
            skeleton = skeleton.split('], ')

            jointCoords = []

            # Each skeleton is stored as a string representing the list of lists of each joint coordinates
            firstJoint = True
            for joint in skeleton[:-1]:
                coords = joint[2 if firstJoint else 1:]
                firstJoin = False
                        
                coords = np.fromstring(coords, dtype=np.float32, sep=',')

                jointCoords.append(coords)


            lastJoint = np.fromstring(skeleton[-1][1:-2], dtype=np.float32, sep=',')
            jointCoords.append(lastJoint)

            jointCoords = np.asarray(jointCoords)
            y = np.asarray(target[i])

            x = torch.tensor(jointCoords, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            label = torch.tensor(np.asarray(target_cross[i]), dtype=torch.float)

            y_values.append(y)
            x_values.append(x)
            label_values.append(label)

        
        x_values = torch.stack(x_values) # Dim0: n samples, Dim1: m nodes per sample, Dim2: k features per node
        
        if self.normalization is 'standardization':
            
            # Standardization:
            
            if self.norm_precomputed_values is None:
                xmean = torch.mean(x_values, dim=[0,1], keepdim=True)
                xstd = torch.std(x_values, dim=[0,1], keepdim=True)
            else:
                xmean, xstd = self.norm_precomputed_values
            
            x_values = (x_values - xmean) / xstd
        
            self.xmean = xmean
            self.xstd = xstd
            
        elif self.normalization is 'minmax':
            
            # Min-max normalization:
            
            if self.norm_precomputed_values is None:
                xmax = torch.amax(x_values, dim=[0,1], keepdim=True)
                xmin = torch.amin(x_values, dim=[0,1], keepdim=True)
            else:
                xmax, xmin = self.norm_precomputed_values
            
            x_values = (x_values - xmin) / (xmax - xmin)
            
            self.xmax = xmax
            self.xmin = xmin

        
        edge_index = torch.tensor(self.edgeindex, dtype=torch.long)

        for iy, x in enumerate(x_values):

            data_element = Data(x=x, y=y_values[iy], label=label_values[iy], edge_index=edge_index, num_nodes=x.shape[0])
            self.data.append(data_element)


    def shuffle(self):
        random.shuffle(self.data)
        return self

    def __getitem__(self, items):
        return self.data[items]
