import numpy as np
import random
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class SkeletonsDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.data=[]
        self.csv_path = csv_path


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
        crossing = self.loadedData['crossing'].to_numpy()
        target_nocross = np.where(crossing==-1, 1, 0)
        target_cross = np.where(crossing==-1, 0, 1)

        target = np.stack([target_nocross, target_cross], axis=1)

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

        
        x_values = torch.stack(x_values)
        x_values = (x_values - torch.mean(x_values)) / torch.std(x_values)

        edge_index = torch.tensor(self.edgeindex, dtype=torch.long)

        for iy, x in enumerate(x_values):

            data_element = Data(x=x, y=y_values[iy], label=label_values[iy], edge_index=edge_index, num_nodes=x.shape[0])
            self.data.append(data_element)


    def shuffle(self):
        random.shuffle(self.data)
        return self

    def __getitem__(self, items):
        return self.data[items]
