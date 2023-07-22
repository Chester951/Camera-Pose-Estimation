import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation


def trilateration(p1, p2, p3, r1, r2, r3):

    ex = (p2 - p1)/(np.linalg.norm(p2 - p1))
    i = np.dot(ex, p3 - p1)
    ey = (p3 - p1 - i*ex)/(np.linalg.norm(p3 - p1 - i*ex))
    ez = np.cross(ex, ey)
    d = np.linalg.norm(p2 - p1)
    j = np.dot(ey, p3 - p1)

    x = (r1**2 - r2**2 + d**2)/(2*d)
    y = ((r1**2 - r3**2 + i**2 + j**2)/(2*j)) - ((i/j)*x)

    z = np.sqrt(r1**2 - x**2 - y**2)

    t1 = p1 + x*ex + y*ey + z*ez
    t2 = p2 + x*ex + y*ey + z*ez

    return t1, t2


def undistortImg(points, distCoeffs, size):

    points /= np.array([size[1], size[0]]).reshape((2, 1))

    center = np.array([0.5, 0.5]).reshape((2, 1))
    r = np.linalg.norm((points - center), axis=0)

    xc, yc = center[0], center[1]
    x, y = points[0], points[1]
    k1, k2, p1, p2 = distCoeffs[0], distCoeffs[1], distCoeffs[2], distCoeffs[3]

    xu = x + (x - xc) * (k1 * (r**2) + k2 * (r**4)) + \
	    (p1 * (r**2 + 2*((x - xc)**2)) + 2 * p2 * (x - xc) * (y - yc))
    yu = y + (y - yc) * (k1 * (r**2) + k2 * (r**4)) + \
	    (p2 * (r**2 + 2*((y - yc)**2)) + 2 * p1 * (x - xc) * (y - yc))
    undistorted_points = np.vstack((xu, yu)) * np.array([size[1], size[0]]).reshape((2, 1))

    return undistorted_points


def myP3P(points3D, points2D, cameraMatrix):

    # Step 1 transform 2D image points from image coordinate sys to camera coordinate sys
    inv_cameraMatrix = np.linalg.pinv(cameraMatrix)    
    homo_points2D = np.concatenate((points2D, np.array([[1, 1, 1, 1]])), axis=0)  
    ccs_points2D = inv_cameraMatrix @ homo_points2D                                
    ccs_points2D = ccs_points2D / np.linalg.norm(ccs_points2D, axis=0)   # 3x4 matrix
    
    # Step 2  G0 G1 G2 G3 G4 feom intrinsic matrix
    v1, v2, v3 = ccs_points2D[:, 0], ccs_points2D[:, 1], ccs_points2D[:, 2]    
       
    Cab, Cac, Cbc = np.dot(v1, v2), np.dot(v1, v3), np.dot(v2, v3)

    x1, x2, x3 = points3D[:, 0], points3D[:, 1], points3D[:, 2]
    Rab, Rac, Rbc = euclidean(x1, x2), euclidean(x1, x3), euclidean(x2, x3)
    
	# Step 3 Find the root
    K1, K2 = (Rbc / Rac) ** 2, (Rbc / Rab) ** 2
	
    G4 = (K1 * K2 - K1 - K2) ** 2 - 4 * K1 * K2 * (Cbc ** 2)
    G3 = 4 * (K1 * K2 - K1 - K2) * K2 * (1 - K1) * Cab \
		+ 4 * K1 * Cbc * ((K1 * K2 - K1 + K2) * Cac + 2 * K2 * Cab * Cbc)
    G2 = (2 * K2 * (1 - K1) * Cab) ** 2 \
        + 2 * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) \
        + 4 * K1 * ((K1 - K2) * (Cbc**2) + K1 * (1 - K2) * (Cac ** 2) - 2 * (1 + K1) * K2 * Cab * Cac * Cbc)
    G1 = 4 * (K1 * K2 + K1 - K2) * K2 * (1 - K1) * Cab \
        + 4 * K1 * ((K1 * K2 - K1 + K2) * Cac * Cbc \
		+ 2 * K1 * K2 * Cab * (Cac ** 2))
    G0 = (K1 * K2 + K1 - K2) ** 2 \
        - 4 * (K1 ** 2) * K2 * (Cac ** 2)

    quartic_param = [G4, G3, G2, G1, G0]
    poly_root = np.roots(quartic_param)
    x = np.array([np.real(r) for r in poly_root if np.isreal(r)])

	# Step 4 Compute y, a, b and c
    m1, p1, q1 = (1 - K1), 2 * (K1 * Cac - x * Cbc), (x ** 2 - K1)
    m2, p2, q2 = 1, 2 * (-x * Cbc), (x ** 2) * (1 - K2) + 2 * x * K2 * Cab - K2
    y = -1 * (m2 * q1 - m1 * q2) / (p1 * m2 - p2 * m1)

    a = np.sqrt((Rab ** 2) / (1 + (x ** 2) - 2 * x * Cab))
    b = x * a
    c = y * a

	# Step 5 trilateration
    intersects = []
    for i in range(len(a)):
        tmp_c1, tmp_c2 = trilateration(points3D[:, 0], points3D[:, 1], points3D[:, 2], a[i], b[i], c[i])
        intersects.append(tmp_c1)
        intersects.append(tmp_c2)
	

	# Step 6 Calculate lambda and R
    possible_sol = []	
    for intersect in intersects:
        intersect = intersect.reshape((3,1))
        for sign in [1,-1]:
            lambda_ = sign * np.linalg.norm((points3D[:, :3] - intersect), axis=0)
            R = (lambda_ * ccs_points2D[:, :3]) @ np.linalg.pinv(points3D[:, :3] - intersect )
            possible_sol.append([R, intersect, lambda_])		
			
    best_R = possible_sol[0][0]
    best_T = possible_sol[0][1]
    min_error = np.inf

	# Step 7 Use 4th point to determine best result
    for R, T, lambda_ in possible_sol:
        projected_2D = cameraMatrix @ R @ (points3D[:, 3].reshape(3, 1) - T)
        projected_2D /= projected_2D[-1]
        error = np.linalg.norm(projected_2D[:2, :] - points2D[:, 3].reshape(2, 1))
        
        if error < min_error:
            best_R = R
            best_T = T
            min_error = error

    return best_R, best_T


def mypnpRansac(points3D, points2D, cameraMatrix, distCoeffs):
       
    img_size = [1920, 1080]
    N = np.log((1-0.99)) / np.log(1 - (1 - 0.5)**3)
    N = round(N)
    min_outlier = np.inf
    best_R, best_T = None, None
    for i in range(N):
        # image undistortion
        mask = np.random.randint(points2D.shape[0], size = 4)
        chosen_points2D = points2D[ mask, :].T
        chosen_points3D = points3D[ mask, :].T 
        chosen_points2D = undistortImg(chosen_points2D, distCoeffs, img_size)
        try:
            R, T = myP3P(chosen_points3D, chosen_points2D, cameraMatrix)

            projected2D = cameraMatrix @ (R @ (points3D.T - T.reshape(3, 1)))
            projected2D /= projected2D[-1, :].reshape((1,-1))
            error = np.linalg.norm(projected2D[:2, :] - points2D.T, axis=0)

            outliers = len(error[np.where(error > 10)])
            if outliers < min_outlier:
                best_R = R
                best_T = T
                min_outlier = outliers
        except:
            print("No solution")
    best_R = Rotation.from_matrix(best_R).as_quat()
    best_T = best_T.reshape(-1)

    return best_R, best_T



def mypnpsolver(query,model):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    return mypnpRansac(points3D, points2D, cameraMatrix, distCoeffs)
