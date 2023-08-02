import torch 
import numpy as np


sigmoid_func = torch.nn.Sigmoid()



def basic_matrix(K, R, t):
    """
    Args:
        K: camera intrinsic matrix
        R: camera rotation matrix
        t: camera translation matrix
    Returns:
        F: basic matrix (3, 3)
    """
    #本质矩阵E = t cross R
    t_cross = torch.tensor([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=torch.float32)
    E = t_cross @ R
    #基础矩阵
    K_inv_transpose = torch.inverse(K).transpose(0, 1)
    F = K_inv_transpose @ E @ torch.inverse(K)

    return F

def epipolar_line_compute(K, R, t, pi):
    """
    given a point pi from view i, compute the epipolar line l in source view j

    Args:
        K: relative camera intrinsic matrix
        R: relative camera rotation matrix
        t: relative camera translation matrix
        pi: point in source view i
    """
    F = basic_matrix(K, R, t)
    #变为齐次坐标
    pi_homogeneous = torch.cat((pi, torch.tensor([1], dtype=torch.float32)))
    l = F @ pi_homogeneous
    l_normalized = l / torch.norm(l[:2])
    return l_normalized

def epipolar_line_distance(pj, l):
    """
    given a point j and epipolar line l, compute the distance between them
    """
    #pj_homogeneous = torch.cat((pj, torch.tensor([1], dtype=torch.float32)))
    distance = torch.abs(torch.matmul(pj, l) / torch.norm(l[:2]))
    return distance


def epipolar_weight_Mat(resolution, K, R, t, threshold=0.7):
    """
    given feature map resolution, compute the epipolar weight matrix M

    Args:
        K: relative camera intrinsic matrix (3 * 3)
        R: relative camera rotation matrix  (3 * 3)
        t: relative camera translation matrix (3,)
        resolution: feature map resolution H * W
        
    return: weight matrix M(HW * HW)
    """
    #version 1 
    # F = basic_matrix(K, R, t)
    # H = resolution[0]
    # W = resolution[1]
    # weight_Mat = torch.zeros(resolution[0]*resolution[1], resolution[0]*resolution[1], dtype=torch.float32)
    
    # # TO be optimized
    # for i in range(H * W):
    #     for j in range(H * W):
    #         # 计算点 i 和点 j 对应的像素坐标 (xi, yi) 和 (xj, yj)
    #         xi, yi = i // W, i % W
    #         xj, yj = j // W, j % W
            
    #         # 构建点 i 和点 j 对应的齐次坐标
    #         pi = torch.tensor([xi, yi, 1], dtype=torch.float32)
    #         pj = torch.tensor([xj, yj, 1], dtype=torch.float32)
    #         l = F @ pi
    #         l_normalized = l / torch.norm(l[:2])
    #         # 计算点 j 到极线 l 的距离
    #         distance = epipolar_line_distance(pj, l_normalized)
    #         # 存储距离到距离矩阵
    #         weight_Mat[i, j] = distance
            
    # return weight_Mat

    # version 2
    F = basic_matrix(K, R, t)
    H, W = resolution
    
    # Generate pixel coordinates
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    pixel_coordinates = torch.stack([y, x, torch.ones_like(x)], dim=2).view(-1, 3).float()  #pixel_cor(i,j, :)为[i, j, 1]
    #print(pixel_coordinates.shape)
    lines =  F@pixel_coordinates.t()        #[3, 128*128]
    lines = lines / torch.norm(lines[:2], dim=0) 
    
    pixel_coordinates = pixel_coordinates.view(-1, W, 3)  #(128, 128, 3)
    lines = lines.view(3, W, -1)     #(3, 128, 128), lines[:, i, j] = pij 对应的极线参数
    pixel_coordinates_flat = pixel_coordinates.view(-1, 3).permute(1,0) #([3, 128*128])
    lines_flat = lines.view(3, -1).permute(1,0) #(128*128, 3)
    W_Mat = torch.abs(torch.matmul(lines_flat, pixel_coordinates_flat)) #[16384, 16384]
    
    W_Mat = 1. - sigmoid_func(50.0*(W_Mat- threshold))
    return W_Mat

def epipolar_Affinity_Mat(key, query):
    """
        Args:
            key: source view feature, shape: (B, channel, height, width)
            query : intermedia UNet feature, shape: (B, channel, height, width)
        Returns:
            affinity_matrix: affinity matrix, shape: (B, height * width, height * width)
    """
    
    key_channel = key.shape[1]
    query_channel = query.shape[1]
    key_flat = key.view(key.size(0), key_channel, -1)           # (B, channel, height * width)
    query_flat = query.view(query.size(0), query_channel, -1)   # (B, channel, height * width)
    # 计算相似度得分,并归一化
    scores = torch.bmm(query_flat.transpose(1, 2), key_flat)
    softmax_func = torch.nn.Softmax(dim= -1)
    affinity_matrix = softmax_func(scores)
    return affinity_matrix    


def batch_epipolar_weight_Mat(resolution, K, R, t, threshold=0.7):
    """
    modified to take in batch inputs

    Args:
        K: relative camera intrinsic matrix (B, 3,  3)
        R: relative camera rotation matrix  (B, 3,  3)
        t: relative camera translation matrix (B, 3,)
        resolution: feature map resolution H * W
        
    return: weight matrix M(HW * HW)
    """
    B = K.shape[0]
    F = torch.zeros(B, 3, 3)

    for i in range(B):
        F[i] = basic_matrix(K[i], R[i], t[i])
    
    H, W = resolution
    
    # Generate pixel coordinates
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    pixel_coordinates = torch.stack([y, x, torch.ones_like(x)], dim=2).view(-1, 3).float()  #pixel_cor(i,j, :)为[i, j, 1]
    
    pixel_coordinates = pixel_coordinates.unsqueeze(0).repeat(B, 1, 1)  #(B, 128*128, 3)
    cor_t = pixel_coordinates.permute(0, 2, 1)
    
    #print(pixel_coordinates.shape)
    lines =  F@cor_t       #[B, 3, 128*128]

    lines = lines / torch.norm(lines[:, :2, :], dim=1).unsqueeze(1)
    
    pixel_coordinates = pixel_coordinates.view(B, -1, W, 3)  #(128, 128, 3)
    lines = lines.view(B, 3, W, -1)     #(B, 3, 128, 128), lines[:, i, j] = pij 对应的极线参数
    pixel_coordinates_flat = pixel_coordinates.view(B, -1, 3).permute(0, 2, 1) #([B, 3, 128*128])
    lines_flat = lines.view(B, 3, -1).permute(0, 2, 1) #(B, 128*128, 3)
    W_Mat = torch.abs(torch.matmul(lines_flat, pixel_coordinates_flat)) #[B, 16384, 16384]
    
    W_Mat = 1. - sigmoid_func(50.0*(W_Mat- threshold))
    return W_Mat