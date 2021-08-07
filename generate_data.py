import argparse
import sys
import cv2
import numpy as np
from numpy import random
import os
import pybullet as p
import pybullet_data
import time

import yaml


# Setup to print out all value of matrix
np.set_printoptions(threshold=sys.maxsize)


def check_directory(directory_list, display=True):
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            if display:
                print(f"Create '{directory}'")


def load_objects_into_env(object_names, min_place, max_place):
    env_obj_info = {}
    random.shuffle(object_names)
    num_objs = random.randint(min_place, max_place)
    for i in range(num_objs):
        # Select one object and random initialize pose
        obj_name = object_names[i]
        pose = random_set_object_pose()
        pose.update({'object_name': obj_name})

        # Load the object into environment
        body_id = p.loadURDF(f"./objects/{obj_name}/{obj_name}.urdf",
                             pose['pos'], pose['qua_rot'])
        # Use `base_pos` and `base_qua_rot` to recover the object's location
        p_pos, p_qua_rot = p.getBasePositionAndOrientation(body_id)
        pose['base_pos'], pose['base_qua_rot'] = p_pos, p_qua_rot
        env_obj_info[body_id] = pose
    return env_obj_info, num_objs


def random_set_object_pose():
    # Rotation (Euler angles in radian)
    a = random.choice([-np.pi/2, 0, np.pi/2], 1, p=[0.15, 0.7, 0.15])
    b = random.choice([-np.pi/2, 0, np.pi/2], 1, p=[0.15, 0.7, 0.15])
    c = random.uniform(-np.pi, np.pi, size=1)[0]

    # Position
    x, y = random.uniform(-0.3, 0.3, size=2)
    z = 0 if (a == 0 and b == 0) else random.uniform(0.05, 0.2, size=1)[0]

    obj_pos, obj_rot = list(map(float, [x, y, z])), list(map(float, [a, b, c]))

    obj_q_rot = p.getQuaternionFromEuler(obj_rot)
    return {'pos': obj_pos, 'euler_rot': obj_rot, 'qua_rot': obj_q_rot}


def generate_one_round(round_idx, env_obj_info, pose_dir, projection_matrix):
    global NUM_X_STEP, NUM_Y_STEP, NUM_Z_STEP, MIN_X_DISTANCE, MIN_Y_DISTANCE
    global MIN_Z_DISTANCE, X_STEP, Y_STEP, Z_STEP, CAM_PARAM
    global IMAGE_DIR, DEPTH_DIR

    img_index = 0
    for x in range(0, NUM_X_STEP):
        for y in range(0, NUM_Y_STEP):
            for z in range(0, NUM_Z_STEP):
                img_index += 1
                # if img_index > 20:
                #     print(f'\n{time.process_time() - start_time} (20)')
                #     return

                camera_position = [MIN_X_DISTANCE + x * X_STEP,
                                   MIN_Y_DISTANCE + y * Y_STEP,
                                   MIN_Z_DISTANCE + z * Z_STEP]
                print(f"[{round_idx}][{img_index:03d}/500] " +
                      f"x: [{camera_position[0]:.4f}/{MAX_X_DISTANCE}], " +
                      f"y: [{camera_position[1]:.4f}/{MAX_Y_DISTANCE}], " +
                      f"z: [{camera_position[2]:.4f}/{MAX_Z_DISTANCE}]",
                      end='\r')

                view_matrix = p.computeViewMatrix(
                    cameraEyePosition=camera_position,
                    cameraTargetPosition=[0.0, 0, 0.02],
                    cameraUpVector=[0, 0.08, 1.3])

                delta_r, delta_g, delta_b = random.normal(0, 0.05, 3)
                light_color = [1-float(delta_r), 1-float(delta_g), 1-float(delta_b)]
                lightDirection = [5, -5, 30]

                # Record pose
                save_poses(env_obj_info, camera_position, light_color,
                           view_matrix, round_idx, img_index, pose_dir)

                # Get bbox, image, and depth image of each object
                img_dir = f"{IMAGE_DIR}/{round_idx:04d}/image{str(img_index)}"
                depth_dir = f"{DEPTH_DIR}/{round_idx:04d}/image{str(img_index)}"
                check_directory([img_dir, depth_dir], display=False)

                bbox_dict = {}
                for target_id, obj_info in env_obj_info.items():
                    # Remove non-target object to edge (-5, -5, -5)
                    remain_object(target_id, env_obj_info)

                    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                        640, 480, view_matrix, projection_matrix, lightDirection,
                        light_color, shadow=True, renderer=p.ER_TINY_RENDERER,
                        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)

                    bbox = target_bbox_from_segmentation(target_id, segImg,
                                                         width, height)
                    bbox_dict[target_id] = bbox

                    save_target_images(rgbImg, depthImg, width, height, round_idx,
                                       img_index, obj_info['object_name'],
                                       img_dir, depth_dir)

                # Recover all object's position
                for body_id, obj_info in env_obj_info.items():
                    p.resetBasePositionAndOrientation(
                        body_id, posObj=obj_info['base_pos'], ornObj=obj_info['base_qua_rot'])

                # Data with whole objects
                width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                    640, 480, view_matrix, projection_matrix, lightDirection,
                    light_color, shadow=True, renderer=p.ER_TINY_RENDERER,
                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)

                record_data(rgbImg, segImg, depthImg, width, height, img_index,
                            env_obj_info, bbox_dict, round_idx, with_seg=True)


def save_poses(env_obj_info, camera_position, light_color, view_matrix, round_idx, img_index, data_dir):
    label_pose = {}
    label_pose['camera'] = {
        'transform_mtx': np.asarray(view_matrix).reshape(4, -1).T.reshape(-1).tolist(),
        'position': camera_position, 'lightColor': light_color}

    for body_id, info in env_obj_info.items():
        obj_name = info['object_name']
        label_pose[obj_name] = {'position': info['pos'],
                                'orientation': info['euler_rot']}

    with open(f"{data_dir}/image{str(img_index)}.yaml", 'w') as outfile:
        yaml.dump(label_pose, outfile)


def remain_object(target_id, env_obj_info):
    edge_pos = [-5, -5, -5]
    for body_id, obj_info in env_obj_info.items():
        if body_id == target_id:
            p.resetBasePositionAndOrientation(
                body_id, posObj=obj_info['base_pos'], ornObj=obj_info['base_qua_rot'])
            continue

        if p.getBasePositionAndOrientation(body_id)[0] != edge_pos:
            p.resetBasePositionAndOrientation(
                body_id, posObj=edge_pos, ornObj=obj_info['base_qua_rot'])


def target_bbox_from_segmentation(target_id, segArr, width, height):
    seg_label = np.asarray(segArr).reshape([height, width, -1])
    mask = (seg_label == target_id).squeeze(axis=2)

    row_min, col_min, row_max, col_max = 0, 0, 0, 0
    if not np.all(mask == False):
        row_min, col_min, row_max, col_max = segmented_mask_to_bbox(mask)
    return row_min, col_min, row_max, col_max


def segmented_mask_to_bbox(mask):
    """ Assume only one object in the mask.
        Args:
            mask (np.array): segmented mask with size (height, width)
        Return:
            The positions of min and max are belong to the object.
    """

    col = np.where(np.sum(mask, axis=0) != 0)
    col_min, col_max = col[0][0], col[0][-1]
    row = np.where(np.sum(mask, axis=1) != 0)
    row_min, row_max = row[0][0], row[0][-1]
    return row_min, col_min, row_max, col_max


def save_target_images(rgbArr, depArr, width, height, round_idx, img_index, obj_name, img_dir, depth_dir):
    # Image
    img = np.asarray(rgbArr).reshape([height, width, -1])[:, :, :3]
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{img_dir}/image_{obj_name}.png", img)

    # Depth
    depth_buffer = np.asarray(depArr).reshape([height, width])
    # Recover to original depth
    far, near = CAM_PARAM['far'], CAM_PARAM['near']
    depth_img = far * near / (far - (far - near) * depth_buffer)
    # Transfer to the range [0, 65535] (16-bit) (linear)
    depth_img = (depth_img / far) * 65535
    depth_img = np.around(depth_img).astype(np.uint16)
    cv2.imwrite(f"{depth_dir}/image_{obj_name}.png", depth_img)


def record_data(rgbArr, segArr, depArr, width, height, numImg, env_obj_info, bbox_dict, round_idx, with_seg=False):
    img_dir = f"{IMAGE_DIR}/{round_idx:04d}"
    depth_dir = f"{DEPTH_DIR}/{round_idx:04d}"
    bbox_dir = f"{LABEL_DIR}/{round_idx:04d}"
    seg_dir = f"{SEGMENT_IMG}/{round_idx:04d}"
    img_bbox_dir = f"{LABELED_IMG}/{round_idx:04d}"
    dir_list = [img_dir, depth_dir, bbox_dir, img_bbox_dir]
    dir_list = dir_list + [seg_dir] if with_seg else dir_list
    check_directory(dir_list, display=False)

    #######################################################
    # Process image
    #######################################################
    # Convert `rgbArr` into `img`
    img = np.asarray(rgbArr).reshape([height, width, -1])[:, :, :3]
    # `img` need BGR format in cv2
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)

    # Save image
    cv2.imwrite(f"{img_dir}/image{str(numImg)}.png", img)

    #######################################################
    # Process depth image
    #######################################################
    global CAM_PARAM
    depth_buffer = np.asarray(depArr).reshape([height, width])

    # Recover to original depth
    far, near = CAM_PARAM['far'], CAM_PARAM['near']
    depth_img = far * near / (far - (far - near) * depth_buffer)

    # Transfer to the range [0, 65535] (16-bit) (linear)
    depth_img = (depth_img / far) * 65535
    depth_img = np.around(depth_img).astype(np.uint16)

    # Save depth image
    cv2.imwrite(f"{depth_dir}/image{str(numImg)}.png", depth_img)

    #######################################################
    # Process semantic segmentation and bounding boxes
    #######################################################
    """ Assume no same object in the one image. """
    global COLOR_DICT

    seg_label = np.asarray(segArr).reshape([height, width, -1])
    seg_image = np.zeros([height, width, 3])

    f = open(f"{bbox_dir}/image{str(numImg)}.txt", "w")
    for body_id, obj_info in env_obj_info.items():
        obj_name = obj_info['object_name']

        # Record bounding box
        bbox = bbox_dict[body_id]
        row_min, col_min, row_max, col_max = bbox
        norm_center_x, norm_center_y, norm_width, norm_height = \
            norm_bounding_box(bbox, width, height, with_center=True)
        f.write(f"{obj_name} {norm_center_x} {norm_center_y} {norm_width} {norm_height}\n")

        # Draw labeled image(bounding boxes)
        cv2.rectangle(img, (col_min, row_min), (col_max, row_max), (0, 0, 255), 2)

        if with_seg:
            mask = (seg_label == body_id).squeeze(axis=2)
            color = COLOR_DICT[obj_name]

            # Caution: when there are some objects which are hidden, there is no label file for these objects.
            # We need to check if the object is hidden
            if np.all(mask == False):
                print(f"The object is hidden [round:{round_idx}/image: {numImg}]")
            else:
                seg_image[mask] = color
    f.close()

    # Save semantic segmentation result
    if with_seg:
        cv2.imwrite(f"{seg_dir}/image{str(numImg)}.png", seg_image)

    # Save labeled image (bounding boxes)
    cv2.imwrite(f"{img_bbox_dir}/bounding box image{str(numImg)}.jpg", img)


def norm_bounding_box(bbox, width, height, with_center):
    """
        bbox: row_min, col_min, row_max, col_max.
              The positions of min and max are belong to the object.
    """
    assert with_center, f"with_center is {with_center} is not support."

    row_min, col_min, row_max, col_max = bbox

    # Record bounding box
    norm_center_x = (col_min + col_max) / (2 * width)
    norm_center_y = (row_min + row_max) / (2 * height)
    norm_width = (col_max - col_min + 1) / width
    norm_height = (row_max - row_min + 1) / height
    return norm_center_x, norm_center_y, norm_width, norm_height


parser = argparse.ArgumentParser()
parser.add_argument('--gui', default=False, action='store_true')
parser.add_argument('--min_place', default=4, type=int,
                    help='Minimum number of placed objects')
parser.add_argument('--max_place', default=8, type=int,
                    help='Maximum number of placed objects (not include)')
parser.add_argument('--max_round', default=1, type=int,
                    help='Maximum Number of round (load objects again)')
parser.add_argument('--init_round', default=1, type=int,
                    help='Initial round')
args = parser.parse_args()

##################################################
# Setup simulation environment
##################################################
# `p.GUI` for GUI and `p.DIRECT` for non-graphical version
physicsClient = p.connect(p.GUI) if args.gui else p.connect(p.DIRECT)

# Use floor (optionally)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Set gravity
p.setGravity(0, 0, -10)

# Camera arameters
CAM_PARAM = {'fov': 62.0, 'aspect': 1.275, 'near': 0.1, 'far': 3.5}

# Data directory
IMAGE_DIR = './data_generate/images'
DEPTH_DIR = './data_generate/depths'
LABELED_IMG = './data_generate/images_bbox'
LABEL_DIR = './data_generate/labels/bounding_box'
SEGMENT_IMG = './data_generate/labels/segmentation'
POSE_DIR = './data_generate/labels/pose'
check_directory([IMAGE_DIR, DEPTH_DIR, LABEL_DIR, LABELED_IMG, SEGMENT_IMG,
                 POSE_DIR])

##################################################
# Setup backgorund floor
##################################################
# Setup background floor (Texture supports png format not jpg format)
background_dir = "./objects/back_ground_model"
bg_pos = [0, 0, 0]
bg_ori = p.getQuaternionFromEuler([3.14/2, 0, 0])
p.loadURDF(f"{background_dir}/Paper/Paper.urdf", bg_pos, bg_ori, globalScaling=0.1)

bg_pos = [0, 0, 0]
bg_ori = p.getQuaternionFromEuler([0, 0, 3.14/2])
planeUid = p.loadURDF(f"{background_dir}/rug/rug.urdf", bg_pos, bg_ori, globalScaling=1)
texUid = p.loadTexture(f"{background_dir}/rug/rug 4.png")
p.changeVisualShape(planeUid, -1, rgbaColor=[0.57, 0.42, 0.36, 1])

############################################################
# Define the moving angle of virtual camera
############################################################
MIN_X_DISTANCE, MAX_X_DISTANCE = -0.5, 0.5
NUM_X_STEP = 10
X_STEP = (MAX_X_DISTANCE - MIN_X_DISTANCE) / NUM_X_STEP

MIN_Y_DISTANCE, MAX_Y_DISTANCE = -0.5, 0.5
NUM_Y_STEP = 10
Y_STEP = (MAX_Y_DISTANCE - MIN_Y_DISTANCE) / NUM_Y_STEP

MIN_Z_DISTANCE, MAX_Z_DISTANCE = 0.25, 0.45
NUM_Z_STEP = 5
Z_STEP = (MAX_Z_DISTANCE - MIN_Z_DISTANCE) / NUM_Z_STEP

##################################################
# Generate synthetic dataset
##################################################
start_time = time.process_time()

# Load object's information
with open("./objects/object_list.txt", 'r') as f:
    object_names = f.read().split('\n')[:-1]
# Be used in the semantic segmentation mask
COLOR_DICT = {name: i for i, name in enumerate(object_names)}

"""
`env_obj_info` is a dictionary which records object's
`pos`, `euler_rot`, `qua_rot`, `object_name`, `base_pos`, and `base_qua_rot`
and the key is the `bodyUniqueId` from PyBullet.
"""
projection_matrix = p.computeProjectionMatrixFOV(
    CAM_PARAM['fov'], CAM_PARAM['aspect'], CAM_PARAM['near'], CAM_PARAM['far'])

for round_idx in range(args.init_round, args.max_round+1):
    print(f"\nStart a new Round: {round_idx}")
    # Load objects into the simulated environment
    env_obj_info, num_objs = load_objects_into_env(
        object_names, args.min_place, args.max_place)
    print(f"Number of objects: {num_objs}")

    # Generate synthrtic data
    pose_dir = f"{POSE_DIR}/{round_idx:04d}"
    check_directory([pose_dir])
    generate_one_round(round_idx, env_obj_info, pose_dir, projection_matrix)

    # Remove all objects in the simulated environment
    for body_id, _ in env_obj_info.items():
        p.removeBody(body_id)

end_time = time.process_time()
print(f"Generate {NUM_X_STEP*NUM_Y_STEP*NUM_Z_STEP} images in {end_time-start_time} s")
p.disconnect()
