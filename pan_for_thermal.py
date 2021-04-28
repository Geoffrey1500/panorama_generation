import cv2 as cv
import numpy as np
import open3d as o3d
from math import cos, sin


def rotation_matrix(angle_set):
    """ Constructs 4D homogeneous rotation matrix given the rotation angles
        (degrees) around the x, y and z-axis. Rotation is implemented in
        XYZ order.
    :param rx: Rotation around the x-axis in degrees.
    :param ry: Rotation around the y-axis in degrees.
    :param rz: Rotation around the z-axis in degrees.
    :return:   4x4 matrix rotation matrix.
    """
    # Convert from degrees to radians.
    rx, ry, rz = angle_set
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


def rotation_orla(angle_set):
    rx, ry, rz = angle_set
    rx = np.pi * rx / 180
    ry = np.pi * ry / 180
    rz = np.pi * rz / 180

    r11, r12, r13 = cos(rz) * cos(ry), cos(rz) * sin(ry) * sin(rx) - sin(rz) * cos(rx), cos(rz) * sin(ry) * cos(rx) + sin(
        rz) * sin(rx)
    r21, r22, r23 = sin(rz) * cos(ry), sin(rz) * sin(ry) * sin(rx) + cos(rz) * cos(rx), sin(rz) * sin(ry) * cos(rx) - cos(
        rz) * sin(rx)
    r31, r32, r33 = -sin(ry), cos(ry) * sin(rx), cos(ry) * cos(rx)

    rotation_final = np.array([[r11, r12, r13],
                               [r21, r22, r23],
                               [r31, r32, r33]])

    return rotation_final


def img_to_point(img_path, camera_path, rotation_angle_set):
    '''
    :param img_path: 图像路径
    :param camera_path: 相机内参路径，为npz文件
    :return: 返回值为以赤道上照片的第一张读取的图片的中心点所确立的坐标, 以及色彩信息的集合
    '''

    img_ = cv.imread(img_path)
    gray_ = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
    print(gray_.shape)

    with np.load(camera_path) as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

    mtx_inv = np.linalg.inv(mtx)

    img_cor_ = np.mgrid[0:gray_.shape[0], 0:gray_.shape[1]]
    x_ = img_cor_[1, :, :]
    y_ = img_cor_[0, :, :]

    img_cor_ = np.dstack((x_, y_[np.lexsort(-y_.T)], np.ones_like(gray_))).reshape(-1, 3)
    img_cor_ = np.dot(mtx_inv, img_cor_.T).T
    img_cor_[:, [0, 1, 2]] = img_cor_[:, [2, 0, 1]]

    y_adj = (np.max(img_cor_[:, 1]) + np.min(img_cor_[:, 1])) / 2
    z_adj = (np.max(img_cor_[:, 2]) + np.min(img_cor_[:, 2])) / 2
    img_cor_ = img_cor_ + np.array([[0, -y_adj, -z_adj]])

    img_cor_ = np.dot(rotation_orla(rotation_angle_set), img_cor_.T).T

    r_ = np.sqrt(np.sum(img_cor_ ** 2, axis=1))
    lon_ = np.arctan2(img_cor_[:, 1], img_cor_[:, 0])
    lat_ = np.arcsin(img_cor_[:, 2] / r_)

    # ny = (np.max(lon_) - np.min(lon_))
    # nz = (np.max(lat_) - np.min(lat_))
    #
    # y_new_ = (lon_ / (ny / gray_.shape[1])).astype(np.int32)
    # z_new_ = (lat_ / (nz / gray_.shape[0])).astype(np.int32)

    y_new_ = (lon_ / 0.00021).astype(np.int32)
    z_new_ = (lat_ / 0.00021).astype(np.int32)

    new_img_cor_ = np.vstack((np.ones_like(y_new_), y_new_, z_new_)).T

    color_ = img_.reshape(-1, 3) / 255
    color_[:, [0, 2]] = color_[:, [2, 0]]

    return new_img_cor_, color_


def point_to_panorama():
    return


# cor_set1, color_set1 = img_to_point('img/middle/_DSC5698-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*0])
# cor_set2, color_set2 = img_to_point('img/middle/_DSC5701-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*1])
# cor_set3, color_set3 = img_to_point('img/middle/_DSC5704-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*2])
#
# cor_set4, color_set4 = img_to_point('img/middle/_DSC5707-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*3])
# cor_set5, color_set5 = img_to_point('img/middle/_DSC5710-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*4])
# cor_set6, color_set6 = img_to_point('img/middle/_DSC5713-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*5])

# cor_set1, color_set1 = img_to_point('img/top/_DSC5668-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*0])
# cor_set2, color_set2 = img_to_point('img/top/_DSC5671-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*1])
# cor_set3, color_set3 = img_to_point('img/top/_DSC5674-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*2])
#
# cor_set4, color_set4 = img_to_point('img/top/_DSC5677-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*3])
# cor_set5, color_set5 = img_to_point('img/top/_DSC5680-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*4])
# cor_set6, color_set6 = img_to_point('img/top/_DSC5683-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*5])
# cor_set7, color_set7 = img_to_point('img/top/_DSC5686-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*6])


cor_set1, color_set1 = img_to_point('img/middle/_DSC5719-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*0])
cor_set2, color_set2 = img_to_point('img/middle/_DSC5689-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*1])
cor_set3, color_set3 = img_to_point('img/middle/_DSC5692-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*2])
cor_set4, color_set4 = img_to_point('img/middle/_DSC5695-HDR.jpg', 'sony_16mm.npz', [0, 0, -36*2])

cor_set5, color_set5 = img_to_point('img/top/_DSC5686-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*0])
cor_set6, color_set6 = img_to_point('img/top/_DSC5668-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*1])
cor_set7, color_set7 = img_to_point('img/top/_DSC5671-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*2])
# cor_set7, color_set7 = img_to_point('img/top/_DSC5674-HDR.jpg', 'sony_16mm.npz', [0, -45, -51.428571428571*3])

cor_set = np.vstack((cor_set1, cor_set2, cor_set3, cor_set4, cor_set5, cor_set6, cor_set7))
color_set = np.vstack((color_set1, color_set2, color_set3, color_set4, color_set5, color_set6, color_set7))

# cor_set = np.vstack((cor_set4, cor_set5, cor_set6, cor_set7))
# color_set = np.vstack((color_set4, color_set5, color_set6, color_set7))


# cor_set = np.vstack((cor_set1, cor_set2, cor_set3, cor_set6, cor_set7))
# color_set = np.vstack((color_set1, color_set2, color_set3, color_set6, color_set7))

# cor_set = np.vstack((cor_set3, cor_set4, cor_set5))
# color_set = np.vstack((color_set3, color_set4, color_set5))
# cor_set = np.vstack((cor_set5, cor_set6))
# color_set = np.vstack((color_set5, color_set6))

print("等着显示图片")

pcd = o3d.geometry.PointCloud()
FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=6000, origin=[0, 0, 0])
pcd.points = o3d.utility.Vector3dVector(cor_set)
pcd.colors = o3d.utility.Vector3dVector(color_set)
o3d.visualization.draw_geometries([pcd, FOR1])
