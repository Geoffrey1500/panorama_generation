import cv2 as cv
import numpy as np
import open3d as o3d


def rotation_matrix(rx, ry, rz):
    """ Constructs 4D homogeneous rotation matrix given the rotation angles
        (degrees) around the x, y and z-axis. Rotation is implemented in
        XYZ order.
    :param rx: Rotation around the x-axis in degrees.
    :param ry: Rotation around the y-axis in degrees.
    :param rz: Rotation around the z-axis in degrees.
    :return:   4x4 matrix rotation matrix.
    """
    # Convert from degrees to radians.
    rx = np.pi * rx / 180
    ry = np.pi * ry / 180
    rz = np.pi * rz / 180

    # Pre-compute sine and cosine of angles.
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])

    # Set up euler rotations.
    Rx = np.array([[1, 0,  0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    Ry = np.array([[cy,  0, sy],
                   [0,   1, 0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz, cz,  0],
                   [0,  0,   1]])

    return Rz.dot(Ry.dot(Rx))


with np.load('sony_16mm.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

mtx_inv = np.linalg.inv(mtx)
img = cv.imread('img/small/_DSC5710-HDR.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray)

# objp = np.ones((gray.shape[0]*gray.shape[1], 3), np.float32)
objp = np.ones_like(gray)
lala = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
x = lala[1, :, :]
y = lala[0, :, :]

img_cor_set = np.dstack((x, y[np.lexsort(-y.T)], objp))
# img_cor_set = np.dstack((img_cor_set, objp))
img_cor_set_re = img_cor_set.reshape(-1, 3)
color = img.reshape(-1, 3)/255
color[:, [0, 2]] = color[:, [2, 0]]

print('hi')
after = np.dot(mtx_inv, img_cor_set_re.T).T
after[:, [0, 1, 2]] = after[:, [2, 0, 1]]
new = np.copy(after)
y_adj = (np.max(new[:, 1]) + np.min(new[:, 1]))/2
z_adj = (np.max(new[:, 2]) + np.min(new[:, 2]))/2
after = after + np.array([[0, -y_adj, -z_adj]])

print(np.max(after[:, 1]*180/np.pi), np.min(after[:, 1]*180/np.pi), np.max(after[:, 2]*180/np.pi), np.min(after[:, 2]*180/np.pi))

r = np.sqrt(np.sum(after**2, axis=1))
lon = np.arctan2(after[:, 1], after[:, 0])
lat = np.arcsin(after[:, 2] / r)

nx = (np.max(lon) - np.min(lon))
nz = (np.max(lat) - np.min(lat))

# x_new = (lon/0.0002).astype(np.int32)
# z_new = (lat/0.0002).astype(np.int32)
x_new = (lon/(nx/3988)).astype(np.int32)
z_new = (lat/(nz/5982)).astype(np.int32)

new_cor = np.vstack((x_new, np.ones_like(x_new), z_new)).T

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(new_cor)
# pcd.colors = o3d.utility.Vector3dVector(color)
# o3d.visualization.draw_geometries([pcd])

img2 = cv.imread('img/small/_DSC5707-HDR.jpg')

color2 = img2.reshape(-1, 3)/255
color2[:, [0, 2]] = color2[:, [2, 0]]

# rotation_angle_z = -36/180*np.pi
# rotation_matrix_z = np.array([[np.cos(rotation_angle_z), -np.sin(rotation_angle_z), 0],
#                               [np.sin(rotation_angle_z), np.cos(rotation_angle_z), 0],
#                               [0, 0, 1]])

data_final = np.dot(after, rotation_matrix(0, 0, -36))

r2 = np.sqrt(np.sum(data_final**2, axis=1))
lon2 = np.arctan2(data_final[:, 1], data_final[:, 0])
lat2 = np.arcsin(data_final[:, 2] / r2)

x_new2 = (lon2/(nx/3988)).astype(np.int32)
z_new2 = (lat2/(nz/5982)).astype(np.int32)

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
