import numpy as np
import cv2

def create_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def compute_projection_matrix(camera_params, lidar_params):
    width = camera_params['Width']
    height = camera_params['Height']
    fov_degrees = camera_params['FOV_Degrees']
    
    fov_radians = np.deg2rad(fov_degrees)
    fx = width / (2.0 * np.tan(fov_radians / 2.0))
    fy = fx 
    cx = width / 2.0
    cy = height / 2.0
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    lidar_x = lidar_params['X']
    lidar_y = lidar_params['Y']
    lidar_z = lidar_params['Z']
    lidar_roll = np.deg2rad(lidar_params['Roll'])
    lidar_pitch = np.deg2rad(lidar_params['Pitch'])
    lidar_yaw = np.deg2rad(lidar_params['Yaw'])
    
    cam_x = camera_params['X']
    cam_y = camera_params['Y']
    cam_z = camera_params['Z']
    cam_pitch = np.deg2rad(camera_params['Pitch'])
    cam_roll = np.deg2rad(camera_params['Roll'])
    cam_yaw = np.deg2rad(camera_params['Yaw'])
    
    R_lidar = create_rotation_matrix(lidar_roll, lidar_pitch, lidar_yaw)
    T_lidar = np.array([[lidar_x],
                        [lidar_y],
                        [lidar_z]])
    
    R_cam = create_rotation_matrix(cam_roll, cam_pitch, cam_yaw)
    T_cam = np.array([[cam_x],
                      [cam_y],
                      [cam_z]])
    
    R = R_cam @ R_lidar.T
    T = T_cam - R @ T_lidar
    
    RT = np.hstack((R, T))
    
    P = K @ RT
    
    return P

def main():
    camera_params = {
        'Width': 1280,
        'Height': 1280,
        'FOV_Degrees': 90,
        'X': -0.3,
        'Y': 0,
        'Z': 0.8,
        'Pitch': 0.0,
        'Roll': 0.0,
        'Yaw': 0
    }
    
    lidar_params = {
        'X': 1.2,
        'Y': 0,
        'Z': 0.4,
        'Roll': 0,
        'Pitch': 0,
        'Yaw': 0
    }
    
    P = compute_projection_matrix(camera_params, lidar_params)
    print("Projection matrix:")
    print(P)

if __name__ == "__main__":
    main()
