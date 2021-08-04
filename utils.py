import pybullet as p


def bbox_vertex(aabb):
    aabbMin, aabbMax = aabb[0], aabb[1]

    p1 = [aabbMax[0], aabbMax[1], aabbMax[2]]
    p2 = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p3 = [aabbMin[0], aabbMin[1], aabbMax[2]]
    p4 = [aabbMax[0], aabbMin[1], aabbMax[2]]

    p5 = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p6 = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p7 = [aabbMin[0], aabbMin[1], aabbMin[2]]
    p8 = [aabbMax[0], aabbMin[1], aabbMin[2]]
    return [p1, p2, p3, p4, p5, p6, p7, p8]


def draw_bbox(bbox_vertexes):
    v = bbox_vertexes

    p.addUserDebugLine(v[0], v[1], [1, 1, 1])
    p.addUserDebugLine(v[1], v[2], [1, 1, 1])
    p.addUserDebugLine(v[2], v[3], [1, 1, 1])
    p.addUserDebugLine(v[3], v[0], [1, 1, 1])

    p.addUserDebugLine(v[4], v[5], [1, 1, 1])
    p.addUserDebugLine(v[5], v[6], [1, 1, 1])
    p.addUserDebugLine(v[6], v[7], [1, 1, 1])
    p.addUserDebugLine(v[7], v[4], [1, 1, 1])

    p.addUserDebugLine(v[0], v[4], [1, 1, 1])
    p.addUserDebugLine(v[1], v[5], [1, 1, 1])
    p.addUserDebugLine(v[2], v[6], [1, 1, 1])
    p.addUserDebugLine(v[3], v[7], [1, 1, 1])




'''
Try to get 3D bounding boxes (Failed)

# aabb = p.getAABB(object1)
# bbox_vertexes = utils.bbox_vertex(aabb)
# utils.draw_bbox(bbox_vertexes)
# print(bbox_vertexes)

# mesh = p.getMeshData(object2, flags=p.MESH_DATA_SIMULATION_MESH)
mesh = p.getMeshData(object2)
points = numpy.asarray(mesh[1])
xs = points[:, 0]
ys = points[:, 1]
zs = points[:, 2]
center_x = np.mean(xs)
center_y = np.mean(ys)
center_z = np.mean(zs)
xs -= center_x
ys -= center_y
zs -= center_z
zs -= min(zs)

xs_min, xs_max = min(xs), max(xs)
ys_min, ys_max = min(ys), max(ys)
zs_min, zs_max = min(zs), max(zs)

# aabbmin = np.array([xs_min, ys_min, zs_min])
# aabbmax = np.array([xs_max, ys_max, zs_max])
aabb = np.array([[xs_min, ys_min, zs_min], [xs_max, ys_max, zs_max]])
mtx = p.getMatrixFromQuaternion(object2_startOrientation)
mtx = np.asarray(mtx).reshape(3, 3).T
aabb = np.matmul(aabb, mtx.T) + object2_startPos

bbox_vertexes = utils.bbox_vertex(aabb)
utils.draw_bbox(bbox_vertexes)
# print(xs_min, xs_max)
# print(ys_min, ys_max)
# print(zs_min, zs_max)
input()
exit()



def visualize_point_cloud(points):
    """ points (numpy.array): shape is (num_points, 3) """

    import open3d
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points)
    open3d.visualization.draw_geometries([point_cloud])


visualize_point_cloud(points)
exit()
aabb = p.getAABB(object2)
bbox_vertexes = utils.bbox_vertex(aabb)
utils.draw_bbox(bbox_vertexes)
print(bbox_vertexes)
# utils.drawAABB(aabb)
input()
exit()
'''