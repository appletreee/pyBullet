import sys
import cv2
import numpy
import os
import pybullet as p
import pybullet_data
from skimage import measure
import time

import yaml
# Setup to print out all value of matrix
numpy.set_printoptions(threshold=sys.maxsize)


def check_directory(directory_list):
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Create '{directory}'")


def segmented_mask_to_bbox(mask):
    """ Assume only one object in the mask. """

    # Create bounding boxes from segmented mask. [min, max)
    label_mask = measure.label(mask)
    props = measure.regionprops(label_mask)

    min_row, min_col, max_row, max_col = props[0].bbox

    # Record bounding box
    norm_center_x = (min_col + max_col - 1) / (2 * width)
    norm_center_y = (min_row + max_row - 1) / (2 * height)
    norm_width = (max_col - min_col) / width
    norm_height = (max_row - min_row) / height

    bbox = (min_row, min_col, max_row - 1, max_col - 1)
    return norm_center_x, norm_center_y, norm_width, norm_height, bbox


def data_generate_v2(rgbArr, segArr, depArr, width, height, numImg):
    #######################################################
    # Process image
    #######################################################
    # Convert `rgbArr` into `img`
    img = numpy.asarray(rgbArr).reshape([height, width, -1])[:, :, :3]
    # `img` need BGR format in cv2
    img = cv2.cvtColor(img.astype(numpy.float32), cv2.COLOR_RGB2BGR)

    # Save image
    # cv2.imwrite(os.path.join(IMAGE_DIR, 'image' + str(numImg) + '.jpg'), img)
    cv2.imwrite(f"{IMAGE_DIR}/image{str(numImg)}.jpg", img)

    #######################################################
    # Process depth image
    #######################################################
    global CAM_PARAM
    if len(depArr) != 0:
        depth_buffer = numpy.asarray(depArr).reshape([height, width])

        # Recover to original depth
        far, near = CAM_PARAM['far'], CAM_PARAM['near']
        depth_img = far * near / (far - (far - near) * depth_buffer)

        # if depth_img < 0.1:
        #     print("false")

        # Transfer to the range [0, 65535] (16-bit) (linear)
        depth_img = (depth_img / far) * 65535
        depth_img = numpy.around(depth_img).astype(numpy.uint16)

        # Save depth image
        # cv2.imwrite(os.path.join(DEPTH_DIR, 'image' + str(numImg) + '.png'), depth_img)
        cv2.imwrite(f"{DEPTH_DIR}/image{str(numImg)}.png", depth_img)

    #######################################################
    # Process semantic segmentation and bounding boxes
    #######################################################
    """ Assume no same object in the one image. """

    # In BGR format (key is the order of object setting)
    # 0, 1, 2, 3 are IDs of objects. If you want to add more target object, just add and define 4,5,6....
    object2color = {0: (0, 0, 255), 1: (0, 255, 0),
                    2: (255, 0, 0), 3: (255, 255, 0)}
    seg_label = numpy.asarray(segArr).reshape([height, width, -1])
    seg_image = numpy.zeros([height, width, 3])

    # f = open(os.path.join(LABEL_DIR, 'image' + str(numImg) + '.txt'), "w")
    f = open(f"{LABEL_DIR}/image{str(numImg)}.txt", "w")
    for obj_order, color in object2color.items():
        mask = (seg_label == obj_order).squeeze(axis=2)

        # Caution: when there are some objects which are hidden, there is no label file for these objects
        # We need to check if the object is hidden
        check = numpy.all(mask == False)
        if check:
            print("There is som objects are hidden")
        else:
            # Segmentation result
            seg_image[mask] = color

            # Get bounding box from segmented mask
            norm_center_x, norm_center_y, norm_width, norm_height, bbox = \
                segmented_mask_to_bbox(mask)
            min_row, min_col, max_row, max_col = bbox

            # Record bounding box
            f.write(f"{obj_order} {norm_center_x} {norm_center_y} {norm_width} {norm_height}\n")

            # Draw labeled image(bounding boxes)
            cv2.rectangle(img, (min_col, min_row), (max_col, max_row), (0, 0, 255), 2)

    f.close()

    # Save semantic segmentation result
    # cv2.imwrite(os.path.join(SEGMENT_IMG, 'image' + str(numImg) + '.jpg'), Seg_img)
    cv2.imwrite(f"{SEGMENT_IMG}/image{str(numImg)}.jpg", seg_image)

    # Save labeled image(bounding boxes)
    # cv2.imwrite(os.path.join(LABELED_IMG, 'bounding box image' + str(numImg) + '.jpg'), img)
    cv2.imwrite(f"{LABELED_IMG}/bounding box image{str(numImg)}.jpg", img)
    return


def setup_parameter_generate(Obj1_Pos, Obj1_Ori, Obj2_Pos, Obj2_Ori, Obj3_Pos, Obj3_Ori, Obj4_Pos, Obj4_Ori, cameraPosition, lightColor, numImg):
    f = open(f"{SETUP_PARAMETER}/image{str(numImg)}.txt", "w")
    f.write(f"Object 1 Position: {Obj1_Pos}, Object 1 Orientation: {Obj1_Ori}\n")
    f.write(f"Object 2 Position: {Obj2_Pos}, Object 1 Orientation: {Obj2_Ori}\n")
    f.write(f"Object 3 Position: {Obj3_Pos}, Object 1 Orientation: {Obj3_Ori}\n")
    f.write(f"Object 4 Position: {Obj4_Pos}, Object 1 Orientation: {Obj4_Ori}\n")
    f.write(f"Camera position: {cameraPosition}\n")
    f.write(f"Light color: Red:{lightColor[0]}, Green:{lightColor[1]}, Blue:{lightColor[2]}\n")
    return


def save_pose(Obj1_Pos, Obj1_Ori, Obj2_Pos, Obj2_Ori, Obj3_Pos, Obj3_Ori, Obj4_Pos, Obj4_Ori,
              cameraPosition, lightColor, numImg, viewMatrix):

    label_pose = {}
    label_pose['camera'] = {'transform_mtx': numpy.asarray(viewMatrix).reshape(4, -1).T.reshape(-1).tolist(),
                            'position': cameraPosition, 'lightColor': lightColor}

    obj_pos = [Obj1_Pos, Obj2_Pos, Obj3_Pos, Obj4_Pos]
    obj_ori = [list(Obj1_Ori), list(Obj2_Ori), list(Obj3_Ori), list(Obj4_Ori)]

    for j, (po, o) in enumerate(zip(obj_pos, obj_ori), 1):
        label_pose[j] = {'position': po, 'orientation': o}

    with open(f"{SETUP_PARAMETER}/image{str(numImg)}.yaml", 'w') as outfile:
        yaml.dump(label_pose, outfile)
    return

# Start simulation environment
# `p.GUI` for GUI and `p.DIRECT` for non-graphical version
# physicsClient = p.connect(p.GUI)
physicsClient = p.connect(p.DIRECT)

# Use floor
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# Set gravity
p.setGravity(0, 0, -10)

NUM_OBJ = 4

# Whether generate depth images
include_depth = True

IMAGE_DIR = './data_generate/images/train'
DEPTH_DIR = './data_generate/depths/train'
LABEL_DIR = './data_generate/labels/train'
LABELED_IMG = './data_generate/labeled_images'
SEGMENT_IMG = './data_generate/segmented_images'
SETUP_PARAMETER = './data_generate/setup_parameters'

check_directory([IMAGE_DIR, DEPTH_DIR, LABEL_DIR, LABELED_IMG, SEGMENT_IMG, SETUP_PARAMETER])

# ###.... define the moving angle of virtual camera.......#####
MIN_X_DISTANCE = -0.4
MAX_X_DISTANCE = 0.4
NUM_X_STEP = 7
X_STEP = (MAX_X_DISTANCE - MIN_X_DISTANCE) / NUM_X_STEP

MIN_Y_DISTANCE = -0.4
MAX_Y_DISTANCE = 0.4
NUM_Y_STEP = 7
Y_STEP = (MAX_Y_DISTANCE - MIN_Y_DISTANCE) / NUM_Y_STEP

MIN_Z_DISTANCE = 0.25
MAX_Z_DISTANCE = 0.45
NUM_Z_STEP = 5
Z_STEP = (MAX_Z_DISTANCE - MIN_Z_DISTANCE) / NUM_Z_STEP

# ###...Define the changing of light color...###
RED_MIN = 0
RED_MAX = 1
NUM_RED_STEP = 3
RED_STEP = (RED_MAX - RED_MIN) / NUM_RED_STEP

GREEN_MIN = 0
GREEN_MIN = 1
NUM_GREEN_STEP = 3
GREEN_STEP = (RED_MAX - RED_MIN) / NUM_RED_STEP

BLUE_MIN = 0
BLUE_MIN = 1
NUM_BLUE_STEP = 3
BLUE_STEP = (RED_MAX - RED_MIN) / NUM_RED_STEP

# Camera arameters
CAM_PARAM = {'fov': 62.0, 'aspect': 1.275, 'near': 0.1, 'far': 3.5}


def dataGenerate(rgbArr, segArr, width, height, numImg):
    img = numpy.zeros([height, width, 3])
    Seg_img = numpy.zeros([height, width, 3])
    for y in range(0, height):
        for x in range(0, width):
            idx = (y * width + x) * 4
            img[y, x, 2] = rgbArr[idx + 0]
            img[y, x, 1] = rgbArr[idx + 1]
            img[y, x, 0] = rgbArr[idx + 2]

    b = numpy.zeros([height, width])
    for y in range(0, height):
        for x in range(0, width):
            b[y, x] = segArr[y*width + x]

    x_list_1 = []
    y_list_1 = []

    x_list_2 = []
    y_list_2 = []

    x_list_3 = []
    y_list_3 = []

    x_list_4 = []
    y_list_4 = []

    for j in range(0, height):
        for i in range(0, width):
            if b[j, i] >= 0:
                obuid = segArr[j*width + i] & ((1 << 24) - 1)
                if obuid == 0:
                    x_list_1.append(i)
                    y_list_1.append(j)

                    Seg_img[j, i, 2] = 255
                    Seg_img[j, i, 1] = 0
                    Seg_img[j, i, 0] = 0
                if obuid == 1:
                    x_list_2.append(i)
                    y_list_2.append(j)

                    Seg_img[j, i, 2] = 0
                    Seg_img[j, i, 1] = 255
                    Seg_img[j, i, 0] = 0
                if obuid == 2:
                    x_list_3.append(i)
                    y_list_3.append(j)

                    Seg_img[j, i, 2] = 0
                    Seg_img[j, i, 1] = 0
                    Seg_img[j, i, 0] = 255
                if obuid == 3:
                    x_list_4.append(i)
                    y_list_4.append(j)

                    Seg_img[j, i, 2] = 0
                    Seg_img[j, i, 1] = 255
                    Seg_img[j, i, 0] = 255

    f = open(os.path.join(LABEL_DIR, 'image' + str(numImg) + '.txt'), "w")

    cv2.imwrite(os.path.join(SEGMENT_IMG, 'image' + str(numImg) + '.jpg'), Seg_img)

    cv2.imwrite(os.path.join(IMAGE_DIR, 'image' + str(numImg) + '.jpg'), img)
    if len(x_list_1) > 0:
        # create image including bounding boxes
        cv2.rectangle(img, (min(x_list_1), min(y_list_1)), (max(x_list_1), max(y_list_1)), (0, 0, 255), 2)
        # write label data
        x_center_nor_1 = (min(x_list_1) + (max(x_list_1) - min(x_list_1)) / 2) / width
        y_center_nor_1 = (min(y_list_1) + (max(y_list_1) - min(y_list_1)) / 2) / height
        bb_width_nor_1 = (1 + max(x_list_1) - min(x_list_1)) / width
        bb_height_nor_1 = (1 + max(y_list_1) - min(y_list_1)) / height
        f.write(str(0) + " " + str(x_center_nor_1) + " " + str(y_center_nor_1) + " " + str(bb_width_nor_1) + " " + str(bb_height_nor_1) + '\n')

    if len(x_list_2) > 0:
        cv2.rectangle(img, (min(x_list_2), min(y_list_2)), (max(x_list_2), max(y_list_2)), (0, 0, 255), 2)
        # write label data
        x_center_nor_2 = (min(x_list_2) + (max(x_list_2) - min(x_list_2)) / 2) / width
        y_center_nor_2 = (min(y_list_2) + (max(y_list_2) - min(y_list_2)) / 2) / height
        bb_width_nor_2 = (1 + max(x_list_2) - min(x_list_2)) / width
        bb_height_nor_2 = (1 + max(y_list_2) - min(y_list_2)) / height
        f.write(str(1) + " " + str(x_center_nor_2) + " " + str(y_center_nor_2) + " " + str(bb_width_nor_2) + " " + str(bb_height_nor_2) + '\n')

    if len(x_list_3) > 0:
        cv2.rectangle(img, (min(x_list_3), min(y_list_3)), (max(x_list_3), max(y_list_3)), (0, 0, 255), 2)
        # write label data
        x_center_nor_3 = (min(x_list_3) + (max(x_list_3) - min(x_list_3)) / 2) / width
        y_center_nor_3 = (min(y_list_3) + (max(y_list_3) - min(y_list_3)) / 2) / height
        bb_width_nor_3 = (1 + max(x_list_3) - min(x_list_3)) / width
        bb_height_nor_3 = (1 + max(y_list_3) - min(y_list_3)) / height
        f.write(str(2) + " " + str(x_center_nor_3) + " " + str(y_center_nor_3) + " " + str(bb_width_nor_3) + " " + str(bb_height_nor_3) + '\n')

    if len(x_list_4) > 0:
        cv2.rectangle(img, (min(x_list_4), min(y_list_4)), (max(x_list_4), max(y_list_4)), (0, 0, 255), 2)
        # write label data
        x_center_nor_4 = (min(x_list_4) + (max(x_list_4) - min(x_list_4)) / 2) / width
        y_center_nor_4 = (min(y_list_4) + (max(y_list_4) - min(y_list_4)) / 2) / height
        bb_width_nor_4 = (1 + max(x_list_4) - min(x_list_4)) / width
        bb_height_nor_4 = (1 + max(y_list_4) - min(y_list_4)) / height
        f.write(str(3) + " " + str(x_center_nor_4) + " " + str(y_center_nor_4) + " " + str(bb_width_nor_4) + " " + str(bb_height_nor_4) + '\n')

    cv2.imwrite(os.path.join(LABELED_IMG, 'bounding box image' + str(numImg) + '.jpg'), img)

    return

# ### ......The angle is measured in Radian

object1_startPos = [0, 0, 0]
obj1_rot = [0, 0, 0]
object1_startOrientation = p.getQuaternionFromEuler([0, 0, 0])

object2_startPos = [0.1, 0.1, 0]
obj2_rot = [0, 0, 1.2*3.14/4]
object2_startOrientation = p.getQuaternionFromEuler([0, 0, 1.2*3.14/4])

object3_startPos = [-0.03, -0.15, 0]
obj3_rot = [0, 0, 0.6*3.14]
object3_startOrientation = p.getQuaternionFromEuler([0, 0, 0.6*3.14])

object4_startPos = [0.2, -0.16, 0]
obj4_rot = [0, 0, 0.28*3.14]
object4_startOrientation = p.getQuaternionFromEuler([0, 0, 0.28*3.14])

# ######........... Setup objects...........####
object1 = p.loadURDF("./objects/002_master_chef_can/002_master_chef_can.urdf", object1_startPos, object1_startOrientation)
object2 = p.loadURDF("./objects/003_cracker_box/003_cracker_box.urdf", object2_startPos, object2_startOrientation)
object3 = p.loadURDF("./objects/004_sugar_box/004_sugar_box.urdf", object3_startPos, object3_startOrientation)
object4 = p.loadURDF("./objects/005_tomato_soup_can/005_tomato_soup_can.urdf", object4_startPos, object4_startOrientation)

# #####..............SETUP BACKGROUND.......####
# Setup background floor (Texture supports png format not jpg format)
BG1_startPos = [0, 0, 0]
BG1_startOri = p.getQuaternionFromEuler([3.14/2, 0, 0])
p.loadURDF("./objects/back_ground_model/Paper/Paper.urdf", BG1_startPos, BG1_startOri, globalScaling=0.1)

BG2_startPos = [0, 0, 0]
BG2_startOri = p.getQuaternionFromEuler([0, 0, 3.14/2])
planeUid = p.loadURDF("./objects/back_ground_model/rug/rug.urdf", BG2_startPos, BG2_startOri, globalScaling=1)
texUid = p.loadTexture("./objects/back_ground_model/rug/rug 4.png")
p.changeVisualShape(planeUid, -1, rgbaColor=[0.57, 0.42, 0.36, 1])

start_time = time.process_time()

# set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

imgIndex = 0

for x in range (0, NUM_X_STEP):
    for y in range (0, NUM_Y_STEP):
        for z in range (0, NUM_Z_STEP):
            for r in range (0, NUM_RED_STEP):
                for g in range (0, NUM_GREEN_STEP):
                    for b in range (0, NUM_BLUE_STEP):
                        # ###... Setup camera position ...###
                        camera_position = [MIN_X_DISTANCE + x * X_STEP,
                                           MIN_Y_DISTANCE + y * Y_STEP,
                                           MIN_Z_DISTANCE + z * Z_STEP]

                        # camera_position = [-0.4, 0.258714, 0.25]

                        # ###... Setup light color ... ###
                        Light_color = [RED_MIN + r * RED_STEP, GREEN_MIN + g * GREEN_STEP, BLUE_MIN + b * BLUE_STEP]

                        print(f"[{imgIndex}] Camera position: {camera_position}")

                        viewMatrix = p.computeViewMatrix(cameraEyePosition=camera_position,
                                                         cameraTargetPosition=[0.0, 0, 0.02],
                                                         cameraUpVector=[0, 0.08, 1.3])
                        projectionMatrix = p.computeProjectionMatrixFOV(fov=62.0, aspect=1.275, nearVal=0.1, farVal=3.5)

                        width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=640, height=480,
                                                                                   viewMatrix=viewMatrix,
                                                                                   projectionMatrix=projectionMatrix,
                                                                                   lightDirection=[5, -5, 30],
                                                                                   lightColor= Light_color,
                                                                                   shadow=True,
                                                                                   renderer=p.ER_TINY_RENDERER,
                                                                                   flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)

                        imgIndex = imgIndex + 1
                        # dataGenerate(rgbImg, segImg, width, height, imgIndex)
                        if include_depth:
                            data_generate_v2(rgbImg, segImg, depthImg, width, height, imgIndex)
                        else:
                            data_generate_v2(rgbImg, segImg, [], width, height, imgIndex)

                        save_pose(object1_startPos, obj1_rot,
                                  object2_startPos, obj2_rot,
                                  object3_startPos, obj3_rot,
                                  object4_startPos, obj4_rot,
                                  camera_position, Light_color, imgIndex,
                                  viewMatrix)

end_time = time.process_time()
print("generate " + str(NUM_Y_STEP * NUM_Y_STEP * NUM_Z_STEP) + " images in " + str(end_time - start_time) + "s")

p.disconnect()
