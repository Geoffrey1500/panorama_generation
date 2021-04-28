import cv2 as cv
import numpy as np
import open3d as o3d

with np.load('sony_16mm.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

print(mtx, dist)

mtx_inv = np.linalg.inv(mtx)
print(mtx_inv)
img = cv.imread('img/small/_DSC5710-HDR.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray)

# objp = np.ones((gray.shape[0]*gray.shape[1], 3), np.float32)
objp = np.ones_like(gray)
lala = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]

img_cor_set = np.dstack((lala[1, :, :], lala[0, :, :], objp))
# img_cor_set = np.dstack((img_cor_set, objp))
img_cor_set_re = img_cor_set.reshape(-1, 3).T
color = img.reshape(-1, 3)/255
color[:, [0, 2]] = color[:, [2, 0]]

print('hi')
after = np.dot(mtx_inv, img_cor_set_re).T
# after = after*np.array([-1, -1, 1])
after[:, [1, 2]] = after[:, [2, 1]]

print(np.max(after[:, 0]), np.min(after[:, 0]), np.max(after[:, 1]), np.min(after[:, 1]), np.max(after[:, 2]), np.min(after[:, 2]))

r = np.sqrt(np.sum(after**2, axis=1))
lon = np.arctan2(after[:, 1], after[:, 0])
lat = np.arcsin(after[:, 2] / r)

nx = (np.max(lon) - np.min(lon))
print(lon[0]*180/np.pi, lon[-1]*180/np.pi)
nz = (np.max(lat) - np.min(lat))

print(nx/3988, nz/5982)
print(nx*180/np.pi, nz*180/np.pi)

print(np.min(after[:, 0]).size, np.min(after[:, 0]))

x_new = (lon/0.0002).astype(np.int32)
z_new = (lat/0.0002).astype(np.int32)

new_cor = np.vstack((x_new, np.ones_like(x_new), z_new)).T

# print(r.shape, np.max(r), np.min(r))

img2 = cv.imread('img/small/_DSC5707-HDR.jpg')

color2 = img2.reshape(-1, 3)/255
color2[:, [0, 2]] = color2[:, [2, 0]]

rotation_angle = 36/180*np.pi
rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                            [0, 0, 1]])

data_final = np.dot(after, rotation_matrix)

r2 = np.sqrt(np.sum(data_final**2, axis=1))
lon2 = np.arctan2(data_final[:, 1], data_final[:, 0])
lat2 = np.arcsin(data_final[:, 2] / r2)

x_new2 = (lon2/0.0002).astype(np.int32)
z_new2 = (lat2/0.0002).astype(np.int32)

new_cor2 = np.vstack((x_new2, np.ones_like(x_new2), z_new2)).T

cor_set = np.vstack((new_cor, new_cor2))
color_set = np.vstack((color, color2))


# my_points = o3d.io.read_point_cloud("my_points.txt", format='xyzrgb')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cor_set)
pcd.colors = o3d.utility.Vector3dVector(color_set)
#
# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(data_final)
# pcd2.colors = o3d.utility.Vector3dVector(color2)

# o3d.visualization.draw_geometries([pcd2, pcd])
o3d.visualization.draw_geometries([pcd])
