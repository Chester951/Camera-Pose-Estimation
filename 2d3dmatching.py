from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import os
import argparse
from mypnpsolver import mypnpsolver # my implementation   
from pnpsolver import pnpsolver     # opencv function
from display import displayPointcloud # my display camera result

print("Loadeing Data...")
images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")

cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
img_size = [1920, 1080]
print("Loading Done")


def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def calculate_error(R, t, gt_R, gt_t):
    err_R = []
    err_t = []
    
    for i in range(len(R)):
        
        err_t.append(np.linalg.norm(t[i]-gt_t[i],2))
        
        norm_R = R[i] / np.linalg.norm(R[i])
        norm_gt_R = gt_R[i] / np.linalg.norm(gt_R[i])
        diff_R = np.clip(np.sum(norm_R * norm_gt_R), 0, 1)
        err_R.append(np.degrees(np.arccos(2 * diff_R * diff_R - 1)))        
    err_R = np.array(err_R)
    err_t = np.array(err_t)

    return np.median(err_R), np.median(err_t)        


# Use OpenCV PnP function
def computeOpencvPnP():
    
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)    

    # Load quaery image
    img_id = images_df["IMAGE_ID"].to_list()
    
    # save result
    R_list = []
    t_list = []
    R_gth = []
    t_gth = []

    # if [R|T] exist
    if(os.path.isfile("Rotation.npy") and os.path.isfile("Translation.npy") and os.path.isfile("gtRotation.npy") and os.path.isfile("gtTranslation.npy") ):
        print("[R|T] file exist, skip solving PnP.")
        store_R = np.load("Rotation.npy") 
        store_t = np.load("Translation.npy")
        gt_R = np.load("gtRotation.npy") 
        gt_t = np.load("gtTranslation.npy")
        
        return store_R, store_t, gt_R, gt_t
    
    else:
        for i in range(len(img_id)):
            
            idx = img_id[i]
            print("processing {}/{} image {}".format(idx, len(img_id), ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]))
                        
            # Load query keypoints and descriptors
            points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
            kp_query = np.array(points["XY"].to_list())
            desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
            ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
            rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
            tvec_gt = ground_truth[["TX","TY","TZ"]].values            

            # Find correspondance and solve pnp
            retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
            
            # OpenCV return are Euler angle, change into Quaternion   
            R_array = R.from_rotvec(rvec.reshape(1,3)).as_quat()
            t_array = tvec.reshape(1,3)     
               
            # Store result
            R_list.append(R_array)
            t_list.append(t_array)
            R_gth.append(rotq_gt)
            t_gth.append(tvec_gt)
        
        # Store the result
        np.save("Rotation", np.array(R_list))
        np.save("Translation", np.array(t_list))
        np.save("gtRotation", np.array(R_gth))
        np.save("gtTranslation", np.array(t_gth))        
        
        return np.array(R_list), np.array(t_list), np.array(R_gth), np.array(t_gth)
    


# Compute Myself PnP
def computeMyselfP3P():
    
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)  

    # Load quaery image
    img_id = images_df["IMAGE_ID"].to_list()
    
    # save result
    R_list = []
    t_list = []
    R_gth = []
    t_gth = []

    # if [R|T] exist
    if(os.path.isfile("myRotation.npy") and os.path.isfile("myTranslation.npy") and os.path.isfile("gtRotation.npy") and os.path.isfile("gtTranslation.npy") ):
        
        print("[R|T] file exist, skip solving myselfPnP.")
        store_R = np.load("myRotation.npy") 
        store_t = np.load("myTranslation.npy")
        gt_R = np.load("gtRotation.npy") 
        gt_t = np.load("gtTranslation.npy")
        
        return store_R, store_t, gt_R, gt_t 
    
    else:
        for i in range(len(img_id)):
            
            idx = img_id[i]
            print("processing {}/{} image {}".format(idx, len(img_id), ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]))            
            
            # Load query keypoints and descriptors
            points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
            kp_query = np.array(points["XY"].to_list())
            desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

            ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
            rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
            tvec_gt = ground_truth[["TX","TY","TZ"]].values     
            
            # Find correspondence and solve pnp
            myR_array, myt_array = mypnpsolver((kp_query, desc_query),(kp_model, desc_model))
                
            # Store result
            R_list.append(myR_array)
            t_list.append(myt_array)
            R_gth.append(rotq_gt)
            t_gth.append(tvec_gt)

            # Store the result
            np.save("myRotation", np.array(R_list))
            np.save("myTranslation", np.array(t_list))
            np.save("gtRotation", np.array(R_gth))
            np.save("gtTranslation", np.array(t_gth))

        return np.array(R_list), np.array(t_list), np.array(R_gth), np.array(t_gth)       
    


def main(args):

    if args.mode == 'opencv':

        print("solving by opencv")
        # compute pnp 
        R_array, t_array, R_gth, t_gth = computeOpencvPnP()

        # calculate error
        err_R, err_t = calculate_error(R_array, t_array, R_gth, t_gth)
        
        print("Rotatin error: {} Pose error: {}".format(err_R, err_t))   

        # display the trajectory
        displayPointcloud(points3D_df, R_array, t_array, cameraMatrix, img_size)
       
    else:

        print("running reproduce result")
        # compute myself pnp
        R_array, t_array, R_gth, t_gth = computeMyselfP3P()

        # calculate error
        err_R, err_t = calculate_error(R_array, t_array, R_gth, t_gth)

        print("Rotatin error: {} Pose error: {}".format(err_R, err_t))
         
        # display the trajectory
        displayPointcloud(points3D_df, R_array, t_array, cameraMatrix, img_size)  

   
### MAIN FUNCTION
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, default='myself', choices=['myself','opencv'])
    args = parser.parse_args()
    main(args)
        