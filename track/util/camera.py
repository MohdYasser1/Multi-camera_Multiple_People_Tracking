import numpy as np
import cv2
import os
import json
import scipy


class Camera():
    # def __init__(self, cal_path):
    #     with open(cal_path, 'r') as file:
    #         data = json.load(file)
    #     self.project_mat = np.array(data["camera projection matrix"])
    #     self.homo_mat = np.array(data["homography matrix"])
    #     self.homo_inv = np.linalg.inv(self.homo_mat)
    #     self.project_inv = scipy.linalg.pinv(self.project_mat)
    #     self.pos = np.linalg.inv(self.project_mat[:,:-1]) @ - self.project_mat[:,-1]

    #     self.homo_feet = self.homo_mat.copy()
    #     # Add an offset to the last column which usually represents translation
    #     #by 0.15 cm. This helps to locate where where the object would touch the ground
    #     self.homo_feet[:, -1] = self.homo_feet[:,-1] + self.project_mat[:,2]*0.15 # z=0.15
    #     self.homo_feet_inv = np.linalg.inv(self.homo_feet)
    #     self.idx = cal_path.split("/")[-2][-4:]
    #     self.idx_int = int(self.idx)
    def __init__(self, sensor_data):
        # Load directly from sensor data dictionary
        self.project_mat = np.array(sensor_data["cameraMatrix"])
        self.homo_mat = np.array(sensor_data["homography"])
        
        # Calculate inverse matrices
        self.homo_inv = np.linalg.inv(self.homo_mat)
        self.project_inv = scipy.linalg.pinv(self.project_mat)
        
        # Calculate camera position
        self.pos = np.linalg.inv(self.project_mat[:,:-1]) @ -self.project_mat[:,-1]

        # Create feet homography matrix
        self.homo_feet = self.homo_mat.copy()
        self.homo_feet[:, -1] += self.project_mat[:, 2] * 0.15  # z=0.15 offset
        self.homo_feet_inv = np.linalg.inv(self.homo_feet)

        # Extract camera index from ID
        cam_id = sensor_data["id"]
        if "_" in cam_id:
            self.idx = cam_id.split("_")[-1].zfill(4)  # Zero-pad to 4 digits
        else:
            self.idx = "0000"  # Default for "Camera" with no number
        self.idx_int = int(self.idx)

def cross(R,V):
    h = [R[1] * V[2] - R[2] * V[1],
         R[2] * V[0] - R[0] * V[2],
         R[0] * V[1] - R[1] * V[0]]
    return h

def Point2LineDist(p_3d, pos, ray):
    return np.linalg.norm(np.cross(p_3d-pos, ray), axis=-1)

def Line2LineDist(pA, rayA, pB, rayB):
    if np.abs(np.dot(rayA, rayB)) > (1-(1e-5)) * np.linalg.norm(rayA, axis=-1) * np.linalg.norm(rayB, axis=-1): #quasi vertical
        return Point2LineDist(pA, pB, rayA)
    else:
        rayCP = np.cross(rayA, rayB)
        return np.abs((pA-pB).dot(rayCP / np.linalg.norm(rayCP, axis=-1), axis=-1))
    
def Line2LineDist_norm(pA, rayA, pB, rayB):
    rayCP = np.cross(rayA, rayB, axis=-1)
    rayCP_norm = np.linalg.norm(rayCP, axis=-1) + 1e-6
    return np.abs(np.sum((pA-pB) * (rayCP/rayCP_norm[:, None]), -1))
    return np.where(
        rayCP_norm < 1e-5,
        Point2LineDist(pA, pB, rayA),
        np.abs(np.sum((pA-pB) * (rayCP/rayCP_norm[:, None]), -1))
    )
    if np.abs(np.dot(rayA, rayB)) > (1-(1e-5)): #quasi parallel
        return Point2LineDist(pA, pB, rayA)
    else:
        rayCP = np.cross(rayA, rayB)
        return np.abs((pA-pB).dot(rayCP / np.linalg.norm(rayCP, axis=-1)))
    

def epipolar_3d_score(pA, rayA, pB, rayB, alpha_epi):
    dist = Line2LineDist(pA, rayA, pB, rayB)
    return 1- dist/alpha_epi

def epipolar_3d_score_norm(pA, rayA, pB, rayB, alpha_epi):
    dist = Line2LineDist_norm(pA, rayA, pB, rayB)
    return 1- dist/alpha_epi

import aic_cpp

epipolar_3d_score_norm = aic_cpp.epipolar_3d_score_norm