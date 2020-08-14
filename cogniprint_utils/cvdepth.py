import numpy as np
import cv2
import math

def check_camera_pose(model_pts, image_pts, camera_matrix, dist_coefs, rvec, tvec):
    '''Checks the camera pose for... something. I guess.'''

    projected_pts, _ = cv2.projectPoints(model_pts, rvec, tvec, camera_matrix, dist_coefs)

    #print(projected_pts.reshape(-1,2).astype(np.int32) - image_pts.reshape(-1,2).astype(np.int32))

    rms = np.sum((projected_pts.reshape(-1,2).astype(np.int32) - image_pts.reshape(-1,2).astype(np.int32)) ** 2)

    return math.sqrt(rms / projected_pts.size)

def transform_point(pt, rvec, tvec):
    '''Transforms a single point using a given tvec and rvec'''

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    tvec = np.reshape(tvec, (-1, 1))
    matrix = np.append(rotation_matrix, tvec, axis=1)
    transformation_matrix = np.append(matrix, [[0,0,0,1]], axis=0)

    homogenous_pt = np.append(pt, [0])
    #print(pt)

    transformed_pt = np.matmul(transformation_matrix, homogenous_pt)[0:3]
    print("transformed: ",transformed_pt)

    return transformed_pt

def transform_point_inverse(pt, rvec, tvec):
    '''Inverse transforms a single point using a given tvec and rvec'''

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    rotation_matrix = rotation_matrix.T

    translation = np.matmul(-rotation_matrix, tvec)
    matrix = np.append(rotation_matrix, np.reshape(translation, (-1, 1)), axis=1)

    transformation_matrix = np.append(matrix, [[0,0,0,1]], axis=0)

    homogenous_pt = np.append(pt, [0])

    transformed_pt = np.matmul(transformation_matrix, homogenous_pt)[0:3]

    return transformed_pt

def compute_plane_equation(p0, p1, p2):
    '''Computes the plane equation given a set of input points'''
    #print(p0,p1,p2)
    # Vector p0_p1
    p0_p1 = p0 - p1

    # Vector p0_p2
    p0_p2 = p0 - p2


    print(p0_p1.shape, p0_p2.shape)
    # Normal vector
    n = np.cross(p0_p1, p0_p2)

    #abcd = np.append(n, [-np.sum(n * p0)], axis=0)

    d = -np.sum(n * p0)

    norm = math.sqrt(np.sum(n**2));

    n /= norm
    d /= norm

    a,b,c = n

    return a,b,c,d

def compute_3d_on_plane_from_2d(image_pt, camera_matrix, a,b,c,d):
    f = camera_matrix[[0,1],[0,1]]
    c_p = camera_matrix[0:2,2]

    normalized_image_pt = (image_pt - c_p) / f

    # Change to use distance from surface by default
    s = -431 # -d / (np.sum([a, b] * normalized_image_pt) + c)

    pt = s * np.append(normalized_image_pt, [1])

    return pt

def get_3D_pts(checkerboard_pts, camera_checkerboard_points, rvec, tvec, camera_matrix, dist_coefs):
    '''Get Real World 3D points from image points'''

    # Check camera pose
    rms = check_camera_pose(checkerboard_pts, camera_checkerboard_points, camera_matrix, dist_coefs, rvec, tvec)
    print("RMS error for camera pose = %s" % rms)

    # Transform model point (in object frame) to the camera frame
    checkerboard_pts_mat = checkerboard_pts.reshape((4,5,3))
    print("Points:", checkerboard_pts_mat[[0,1,-1],[0,3,-1]])
    plane_pts = [transform_point(pt, rvec, tvec) for pt in checkerboard_pts_mat[[0,1,-1],[0,3,-1]]]

    # Compute plane equation in the camera frame
    a,b,c,d = compute_plane_equation(*plane_pts)
    print("Plane equation = %s;%s;%s;%s" % (a,b,c,d))

    # Compute 3D from 2D
    pts_3d_camera_frame = [compute_3d_on_plane_from_2d(checkerboard_point, camera_matrix, a,b,c,d) for checkerboard_point in camera_checkerboard_points]
    pts_3d_object_frame = [transform_point_inverse(pt, rvec, tvec) for pt in pts_3d_camera_frame]

    rms_3D = np.sum((checkerboard_pts - pts_3d_object_frame) ** 2)
    print("RMS error for model points = %s" % math.sqrt(rms_3D / camera_checkerboard_points.size))

    return pts_3d_camera_frame, pts_3d_object_frame, rms_3D
