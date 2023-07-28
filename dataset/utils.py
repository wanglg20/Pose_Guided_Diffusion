import numpy as np


def Calculate_Rel_Mat(R1, t1, R2, t2):
    T1 = np.array([[R1[0][0], R1[0][1], R1[0][2], t1[0]],
               [R1[1][0], R1[1][1], R1[1][2], t1[1]],
               [R1[2][0], R1[2][1], R1[2][2], t1[2]],
               [0, 0, 0, 1]])
    T2 = np.array([[R2[0][0], R2[0][1], R2[0][2], t2[0]],
                [R2[1][0], R2[1][1], R2[1][2], t2[1]],
                [R2[2][0], R2[2][1], R2[2][2], t2[2]],
                [0, 0, 0, 1]])
    T_rel = np.linalg.inv(T1) @ T2
    R_rel = T_rel[:3, :3]
    t_rel = T_rel[:3, 3]
    return R_rel, t_rel