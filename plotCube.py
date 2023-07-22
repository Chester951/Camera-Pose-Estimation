from turtle import shape
import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd
from scipy.spatial.transform import Rotation


class Point():
    def __init__(self, position, color):
        self.position = position
        self.color = color

def make_points_to_cube():

    points = []

    #front
    front = [[0,0,0], [1,0,0], [0,0,1], [1,0,1]]
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , 0 , (j+1)*0.1]
            #print(point_pos)
            points.append(Point(point_pos, (0, 0, 255)))

    #back
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , 1 , (j+1)*0.1]
            #print(point_pos)
            points.append(Point(point_pos, (0, 0, 255)))

    #top
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , (j+1)*0.1, 1]
            #print(point_pos)
            points.append(Point(point_pos, (0, 255, 0)))


    #bottom
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , (j+1)*0.1, 0]
            #print(point_pos)
            points.append(Point(point_pos, (0, 255, 0)))

    #left
    for i in range(9):
        for j in range(9):
            point_pos = [0, (i+1)*0.1 , (j+1)*0.1]
            #print(point_pos)
            points.append(Point(point_pos, (255, 0, 0)))

    #right
    for i in range(9):
        for j in range(9):
            point_pos = [1, (i+1)*0.1 , (j+1)*0.1]
            #print(point_pos)
            points.append(Point(point_pos, (255, 0, 0)))

    return points



def draw_points(img,R,t,vertices, cameraMatrix):
    
    R = Rotation.from_quat(R).as_matrix()
    #print(R)
    points = make_points_to_cube()
    points.sort(key=lambda point: np.linalg.norm((point.position-t)), reverse=True)

    for i, p in enumerate(points):
        pos = (cameraMatrix @ (R @ (points[i].position-t).T)).T
        pos = (pos/pos[2])
        if ((pos<0).any()):
            continue
        img = cv2.circle(img, (int(pos[0]), int(pos[1])), radius=5, color=points[i].color, thickness=-1)
    
    return img


def main():
    # Initialize
    path_dir = './data/frames'
    image_df = pd.read_pickle("data/images.pkl")
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    
    # load cube
    cube = o3d.geometry.TriangleMesh.create_box(width=3.0, height=3.0, depth=3.0)
    cube_vertices = np.asarray(cube.vertices).copy()

    # load data
    R = np.load('./result/myRotation.npy')
    t = np.load('./result/myTranslation.npy')

    # load the valid_imgXXX.jpg
    file = [f for f in os.listdir(path_dir) if f.find('valid')!= -1]
    file.sort(key=lambda x:int(x[9:-4]))

    video_imgs = []
    for f in file:
        filename = os.path.join(path_dir,f)
        print('filename: ',f)
        
        # find the correspondance index
        idx = ((image_df.loc[image_df["NAME"]==f])["IMAGE_ID"].values)[0]
        rimg = cv2.imread(filename)
        video_imgs.append(draw_points(rimg,R[idx-1],t[idx-1],cube_vertices, cameraMatrix))
    img_shape = (rimg.shape[1], rimg.shape[0])

    video = cv2.VideoWriter("video.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 15, img_shape)

    for img in video_imgs:
        video.write(img)

    video.release()
        


if __name__ == '__main__':
    main()