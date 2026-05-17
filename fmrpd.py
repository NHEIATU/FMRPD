import numpy as np
import heapq
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. 基础工具：归一化
# ============================================================

def normalize_feature(x, mask, eps=1e-9):
    """
    只对非空网格进行归一化。
    """
    y = np.zeros_like(x, dtype=float)

    if not np.any(mask):
        return y

    valid = x[mask]
    min_v = np.min(valid)
    max_v = np.max(valid)

    y[mask] = (valid - min_v) / (max_v - min_v + eps)
    return y


# ============================================================
# 2. kNN 图 + Dijkstra 测地距离
# ============================================================

def build_knn_graph(points, k=10):
    """
    在局部点云上构建 kNN 图。
    图的边权为欧氏距离。
    """
    k = min(k, len(points))

    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)

    graph = [[] for _ in range(len(points))]

    for i in range(len(points)):
        for j, d in zip(indices[i], distances[i]):
            if i != j:
                graph[i].append((j, d))

    return graph


def dijkstra(graph, start_idx):
    """
    在 kNN 图上计算从关键点到其他局部邻域点的最短路径距离。
    该距离作为测地距离近似。
    """
    dist = np.full(len(graph), np.inf)
    dist[start_idx] = 0.0

    pq = [(0.0, start_idx)]

    while pq:
        cur_dist, u = heapq.heappop(pq)

        if cur_dist > dist[u]:
            continue

        for v, w in graph[u]:
            new_dist = cur_dist + w

            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    return dist


# ============================================================
# 3. 曲率计算
# ============================================================

def compute_curvature(points, k=20):
    """
    根据局部协方差矩阵特征值计算曲率：
        curvature = lambda_min / (lambda1 + lambda2 + lambda3)
    """
    n = len(points)
    k = min(k, n)

    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)

    curvature = np.zeros(n)

    for i in range(n):
        neighbors = points[indices[i]]
        center = np.mean(neighbors, axis=0)
        cov = np.cov((neighbors - center).T)

        eigvals = np.linalg.eigvalsh(cov)
        eig_sum = np.sum(eigvals)

        curvature[i] = eigvals[0] / (eig_sum + 1e-9)

    return curvature


# ============================================================
# 4. 法向一致性计算
# ============================================================

def compute_normal_consistency(points, normals, k=20):
    """
    计算每个点与其邻域点法向量之间的平均余弦相似度。
    """
    n = len(points)
    k = min(k, n)

    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)

    consistency = np.zeros(n)

    for i in range(n):
        n_i = normals[i]
        n_neighbors = normals[indices[i]]

        dot_vals = np.abs(n_neighbors @ n_i)
        norm_vals = np.linalg.norm(n_neighbors, axis=1) * np.linalg.norm(n_i)

        cos_vals = dot_vals / (norm_vals + 1e-9)
        consistency[i] = np.mean(cos_vals)

    return consistency


# ============================================================
# 5. 旋转矩阵
# ============================================================

def rotation_matrix_xyz(angle):
    """
    绕 x、y、z 三轴旋转相同角度。
    angle 为弧度。
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])

    Ry = np.array([
        [ np.cos(angle), 0, np.sin(angle)],
        [0,              1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])

    return Rz @ Ry @ Rx


# ============================================================
# 6. FMRPD 描述子核心构建
# ============================================================

def build_fmrpd_descriptor(
    points,
    normals,
    keypoint,
    external_lrf,
    radius,
    grid_size,
    k_geodesic=10,
    k_curvature=20,
    k_normal=20,
    rotation_angles=None
):


    if rotation_angles is None:
        rotation_angles = [0.0]

    x_axis, y_axis, z_axis = external_lrf

    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    z_axis = np.asarray(z_axis, dtype=float)

    # LRF 矩阵，来自外部结果，不在本代码中计算
    U = np.column_stack([x_axis, y_axis, z_axis])

    # ------------------------------------------------------------
    # 1. 选取关键点邻域
    # ------------------------------------------------------------

    dist_to_keypoint = np.linalg.norm(points - keypoint, axis=1)
    local_indices = np.where(dist_to_keypoint <= radius)[0]

    local_points = points[local_indices]
    local_normals = normals[local_indices]

    if len(local_points) < 3:
        return []

    # 将关键点加入局部点集首位，便于测地距离计算
    local_points_with_key = np.vstack([keypoint.reshape(1, 3), local_points])

    # ------------------------------------------------------------
    # 2. 测地距离：kNN 图 + Dijkstra
    # ------------------------------------------------------------

    graph = build_knn_graph(local_points_with_key, k=k_geodesic)
    geodesic_all = dijkstra(graph, start_idx=0)

    # 去掉首位关键点自身，只保留邻域点测地距离
    geodesic_values = geodesic_all[1:]

    # ------------------------------------------------------------
    # 3. 曲率、法向一致性
    # ------------------------------------------------------------

    curvature_values = compute_curvature(local_points, k=k_curvature)
    normal_consistency_values = compute_normal_consistency(
        local_points,
        local_normals,
        k=k_normal
    )

    # ------------------------------------------------------------
    # 4. 球面网格参数
    # ------------------------------------------------------------

    num_theta = int(np.ceil(2 * np.pi / grid_size))
    num_phi = int(np.ceil(np.pi / grid_size))

    descriptors = []

    # ------------------------------------------------------------
    # 5. 多视角旋转投影
    # ------------------------------------------------------------

    for angle in rotation_angles:

        R_local = rotation_matrix_xyz(angle)

        # 在 LRF 坐标系中旋转
        R_global = U @ R_local @ U.T

        rotated_points = (local_points - keypoint) @ R_global.T + keypoint
        rotated_normals = local_normals @ R_global.T

        grid_curvature = np.zeros((num_phi, num_theta))
        grid_geodesic = np.zeros((num_phi, num_theta))
        grid_density = np.zeros((num_phi, num_theta))
        grid_normal = np.zeros((num_phi, num_theta))
        grid_count = np.zeros((num_phi, num_theta))

        # --------------------------------------------------------
        # 6. 点投影到球面网格
        # --------------------------------------------------------

        for i, p in enumerate(rotated_points):

            rel = p - keypoint

            # 转换到外部 LRF 坐标系
            local_coord = U.T @ rel

            r = np.linalg.norm(local_coord)

            if r < 1e-9:
                continue

            theta = np.arctan2(local_coord[1], local_coord[0])
            phi = np.arccos(np.clip(local_coord[2] / r, -1.0, 1.0))

            theta_idx = int((theta + np.pi) / (2 * np.pi) * num_theta)
            phi_idx = int(phi / np.pi * num_phi)

            theta_idx = np.clip(theta_idx, 0, num_theta - 1)
            phi_idx = np.clip(phi_idx, 0, num_phi - 1)

            grid_curvature[phi_idx, theta_idx] += curvature_values[i]
            grid_geodesic[phi_idx, theta_idx] += geodesic_values[i]
            grid_density[phi_idx, theta_idx] += 1.0
            grid_normal[phi_idx, theta_idx] += normal_consistency_values[i]
            grid_count[phi_idx, theta_idx] += 1.0

        # --------------------------------------------------------
        # 7. 网格内求平均
        # --------------------------------------------------------

        mask = grid_count > 0

        grid_curvature[mask] /= grid_count[mask]
        grid_geodesic[mask] /= grid_count[mask]
        grid_normal[mask] /= grid_count[mask]

        # density 本身就是点数，不需要除以 count

        # --------------------------------------------------------
        # 8. RGBA 多通道融合
        # --------------------------------------------------------

        R = normalize_feature(grid_curvature, mask)
        G = normalize_feature(grid_geodesic, mask)
        B = normalize_feature(grid_density, mask)
        A = normalize_feature(grid_normal, mask)

        rgba_descriptor = np.stack([R, G, B, A], axis=-1)

        # 空网格置零
        rgba_descriptor[~mask] = 0.0

        descriptors.append(rgba_descriptor)

    return descriptors