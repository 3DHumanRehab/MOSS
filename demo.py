# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # 定义一个简化的骨骼权重类
# class BoneWeight:
#     def __init__(self, bone_index, weight):
#         self.bone_index = bone_index
#         self.weight = weight

# # 假设的顶点，带有位置和骨骼权重列表
# class Vertex:
#     def __init__(self, position):
#         self.position = position
#         self.bone_weights = []

# # 假设的骨骼，带有旋转和位移
# class Bone:
#     def __init__(self, rotation, translation):
#         self.rotation = rotation
#         self.translation = translation

# def dual_quaternion_skinning(vertices, bones):
#     # 对每个顶点应用变换
#     for vertex in vertices:
#         # 初始化累计双四元数
#         dq_accum = np.zeros(8)
#         total_weight = 0

#         # 累加影响该顶点的每个骨骼的双四元数
#         for bw in vertex.bone_weights:
#             bone = bones[bw.bone_index]
#             weight = bw.weight
#             total_weight += weight

#             # 计算双四元数
#             q_rotation = R.from_quat(bone.rotation).as_quat()
#             q_translation = np.array([0, bone.translation[0], bone.translation[1], bone.translation[2]])
#             dual_quat = np.concatenate((q_rotation, q_translation)) * weight
            
#             # 累加
#             dq_accum += dual_quat

#         # 归一化双四元数
#         dq_accum /= total_weight

#         # 应用双四元数变换到顶点（留给读者实现，需要将双四元数转换为变换矩阵或直接应用）

#     return vertices

# # 创建一些假设的数据来演示
# vertices = [Vertex(np.array([1, 0, 0]))]  # 一个顶点在(1, 0, 0)
# vertices[0].bone_weights.append(BoneWeight(0, 1))  # 完全被一个骨骼影响
# bones = [Bone([1, 0, 0, 0], [0, 0, 0])]  # 一个位于原点，无旋转的骨骼

# # 运行双四元数蒙皮
# skinned_vertices = dual_quaternion_skinning(vertices, bones)

# # 打印结果，需要实现顶点变换后的位置打印
# for v in skinned_vertices:
#     print(v.position)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CrossAttention(nn.Module):
#     def __init__(self, feature_dim, num_heads=1):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         self.feature_dim = feature_dim
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key = nn.Linear(9, feature_dim)
#         self.value = nn.Linear(9, feature_dim)

#     def forward(self, query, key, value):
#         # 这里仅为示例，实际应用中可能需要根据query, key, value的实际情况调整维度
#         Q = self.query(query)  # [batch_size, seq_len, feature_dim]
#         K = self.key(key)      # [batch_size, seq_len, feature_dim]
#         V = self.value(value)  # [batch_size, seq_len, feature_dim]

#         # 计算注意力分数
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
#         attention = F.softmax(attention_scores, dim=-1)

#         # 应用注意力分数到value上
#         output = torch.matmul(attention, V)

#         return output

# # 假设数据
# A = torch.rand(6850, 24)  # LBS特征
# B = torch.rand(24, 9)    # 合并后的关节特征和关节旋转特征

# # 将B调整为和A相同的第一维度

# # 定义模型并应用
# feature_dim = 24  # 或者其它根据实际情况定义的特征维度
# num_heads = 1     # 根据需要设置头数
# model = CrossAttention(feature_dim, num_heads)

# # 将A, B作为query, key和value输入到模型
# output = model(A, B, B)

# print(output.shape)  # 查看输出形状，理解数据流动



# import torch

# def normalize_quaternion(q):
#     return q / q.norm(p=2, dim=-1, keepdim=True)

# def dual_quaternion_skinning(vertices, bone_indices, bone_weights, bone_dual_quaternions):
#     # vertices: (num_vertices, 3) - The positions of the vertices
#     # bone_indices: (num_vertices, num_influences) - The indices of the bones affecting each vertex
#     # bone_weights: (num_vertices, num_influences) - The weights of the bones affecting each vertex
#     # bone_dual_quaternions: (num_bones, 8) - The dual quaternions of the bones

#     # Normalize the dual quaternions
#     dq_norm = normalize_quaternion(bone_dual_quaternions[:, :4])
#     bone_dual_quaternions = torch.cat([dq_norm, bone_dual_quaternions[:, 4:]], dim=-1)

#     # Prepare the vertices
#     num_vertices = vertices.shape[0]
#     extended_vertices = torch.cat([vertices, torch.ones(num_vertices, 1)], dim=1)

#     # Apply the skinning
#     transformed_vertices = torch.zeros_like(vertices)
#     for i in range(num_vertices):
#         vertex = extended_vertices[i]
#         dq_acc = torch.zeros(8)
#         for j, (bone_idx, weight) in enumerate(zip(bone_indices[i], bone_weights[i])):
#             bone_dq = bone_dual_quaternions[bone_idx]
#             dq_acc += bone_dq * weight  # Weighted sum of dual quaternions

#         # Convert the accumulated dual quaternion back to affine transformation
#         dq_acc = normalize_quaternion(dq_acc)  # Normalize the accumulated dual quaternion
#         transformed_vertex = apply_dual_quaternion_to_point(dq_acc, vertex)  # Apply to the vertex
#         transformed_vertices[i] = transformed_vertex[:3]  # Ignore the added 1 from earlier

#     return transformed_vertices

# def apply_dual_quaternion_to_point(dq, point):
#     # This function applies the dual quaternion to a point
#     # dq: (8,) - The dual quaternion
#     # point: (4,) - The point (in homogeneous coordinates)

#     # Split the dual quaternion
#     qr = dq[:4]  # Real part
#     qd = dq[4:]  # Dual part

#     # Compute the translation vector
#     t = 2 * quaternion_multiply(qd, quaternion_conjugate(qr))[:3]

#     # Apply rotation and translation
#     rotated_point = quaternion_rotate_point(qr, point[:3])
#     translated_point = rotated_point + t

#     return torch.cat([translated_point, point[3:]])

# # Other necessary quaternion functions (to be implemented)
# def quaternion_multiply(a, b):
#     # Multiplies two quaternions
#     # a, b: (4,) - The quaternions
#     pass

# def quaternion_conjugate(q):
#     # Conjugates a quaternion
#     # q: (4,) - The quaternion
#     pass

# def quaternion_rotate_point(q, p):
#     # Rotates a point using a quaternion
#     # q: (4,) - The quaternion
#     # p: (3,) - The point
#     pass

# # Simulated data (for demonstration)
# num_vertices = 6850
# num_bones = 24
# vertices = torch.rand(num_vertices, 3)
# bone_indices = torch.randint(0, num_bones, (num_vertices, 4))
# bone_weights = torch.rand(num_vertices, 4)
# bone_dual_quaternions = torch.rand(num_bones, 8)

# # Perform dual quaternion skinning
# transformed_vertices = dual_quaternion_skinning(vertices, bone_indices, bone_weights, bone_dual_quaternions)
# print(transformed_vertices.shape)



# import numpy as np

# # 假设函数，实际应用中你需要实现或获取对偶四元数和其操作
# def dual_quaternion_from_bone(bone):
#     # 根据骨骼信息生成对偶四元数
#     # 实际应用中应该根据骨骼的旋转和位移来计算
#     return np.random.rand(8)  # 假设的对偶四元数

# def transform_point_by_dual_quaternion(dq, point):
#     # 使用对偶四元数变换一个点
#     # 这里需要实现对偶四元数和点的乘法操作
#     return point  # 返回变换后的点（这里仅为示例，没有实际变换）

# # 假设的骨骼、顶点和权重数据
# bones = [np.random.rand(3, 3) for _ in range(10)]  # 假设有10个骨骼
# vertices = [np.random.rand(3) for _ in range(100)]  # 假设有100个顶点
# weights = np.random.rand(100, 10)  # 每个顶点对每个骨骼的权重

# # 使用对偶四元数蒙皮
# transformed_vertices = []
# for vertex in vertices:
#     # 对于每个顶点，计算所有骨骼影响后的新位置
#     new_pos = np.zeros(3)
#     for bone, weight in zip(bones, weights):
#         dq = dual_quaternion_from_bone(bone)  # 获取骨骼的对偶四元数
#         transformed = transform_point_by_dual_quaternion(dq, vertex)  # 变换顶点
#         new_pos += transformed * weight  # 加权叠加
#     transformed_vertices.append(new_pos)

# # 输出变换后的顶点（这里仅为示例，没有实际变换）
# print(transformed_vertices)


import numpy as np

# 对偶四元数的共轭
def dual_quat_conjugate(dq):
    q0, qe = dq[:4], dq[4:]
    q0_conj = q0 * np.array([1, -1, -1, -1])
    qe_conj = qe * np.array([1, -1, -1, -1])
    return np.concatenate([q0_conj, qe_conj])

# 四元数乘法
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

# 将点转换为纯四元数
def point_to_quat(point):
    return np.concatenate([[0], point])

# 将四元数转换为点（丢弃四元数的实部）
def quat_to_point(quat):
    return quat[1:]

# 使用对偶四元数变换一个点
def transform_point_by_dual_quaternion(dq, point):
    q, qe = dq[:4], dq[4:]
    p = point_to_quat(point)
    
    # 计算对偶四元数的共轭
    dq_star = dual_quat_conjugate(dq)
    q_star, qe_star = dq_star[:4], dq_star[4:]

    # 变换公式 p' = QpQ*
    # 注意：实际变换中，涉及到四元数与对偶四元数的乘法
    translated = quat_multiply(quat_multiply(q, p), q_star)
    translated_point = quat_to_point(translated)
    return translated_point

# 测试数据
dq = np.random.rand(8)  # 假设的对偶四元数
point = np.array([1, 0, 0])  # 一个测试点

# 进行变换
transformed_point = transform_point_by_dual_quaternion(dq, point)
print("Original point:", point)
print("Transformed point:", transformed_point)
