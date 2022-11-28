import numpy as np
import struct

BRDF_SAMPLING_RES_THETA_H = 90
BRDF_SAMPLING_RES_THETA_D = 90
BRDF_SAMPLING_RES_PHI_D   = 360

RED_SCALE   = (1.0/1500.0)
GREEN_SCALE = (1.15/1500.0)
BLUE_SCALE  = (1.66/1500.0)
M_PI = 3.1415926535897932384626433832795

def read_brdf(filename):
    with open(filename, 'rb') as f:
        bdata = f.read()
    dims = []
    dims.append(int.from_bytes(bdata[:4], 'little'))
    dims.append(int.from_bytes(bdata[4:8], 'little'))
    dims.append(int.from_bytes(bdata[8:12], 'little'))
    n = np.prod(dims)
    if n != (BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2):
        print('Dimensions don\'t match')
        return None
    print(dims)
    print(n)
    brdf = struct.unpack(str(3*n)+'d', bdata[12:])

    brdf = np.reshape(np.array(brdf), (3,BRDF_SAMPLING_RES_THETA_H,BRDF_SAMPLING_RES_THETA_D,BRDF_SAMPLING_RES_PHI_D//2))
    brdf = np.transpose(brdf,(1,2,3,0))
    brdf *= np.array([RED_SCALE, GREEN_SCALE, BLUE_SCALE])
    print(brdf.shape)
    return brdf

def normalize(v, axis=None):
    return v / np.linalg.norm(v,axis=axis, keepdims=True)

def rotate_vector(vector, axis, angle):
    out = vector * np.cos(angle)

    temp = np.sum(axis*vector)
    temp = temp*(1.0-np.cos(angle))

    out += axis * temp

    cross = np.cross(axis,vector)
    out += cross * np.sin(angle)

    return out
	


def std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out):
	# out : double& theta_half,double& fi_half,double& theta_diff,double& fi_diff
    v_in = np.array([np.sin(theta_in)*np.cos(fi_in), np.sin(theta_in)*np.sin(fi_in), np.cos(theta_in)])
    v_out = np.array([np.sin(theta_out)*np.cos(fi_out), np.sin(theta_out)*np.sin(fi_out), np.cos(theta_out)])
    v_half = normalize(v_in+v_out)
    
    theta_half = np.arccos(np.clip(v_half[2],-1,1))
    fi_half = np.arctan2(v_half[1], v_half[0])

    bi_normal = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    temp = rotate_vector(v_in, normal , -fi_half)
    diff = rotate_vector(temp, bi_normal, -theta_half)

    theta_diff = np.arccos(np.clip(diff[2],-1,1))
    fi_diff = np.arctan2(diff[1], diff[0])

    return theta_half, fi_half, theta_diff, fi_diff

# Lookup theta_half index
# This is a non-linear mapping!
# In:  [0 .. pi/2]
# Out: [0 .. 89]
def theta_half_index(theta_half):
    if (theta_half <= 0.0):
        return 0
    theta_half_deg = ((theta_half / (M_PI/2.0))*BRDF_SAMPLING_RES_THETA_H)
    temp = theta_half_deg*BRDF_SAMPLING_RES_THETA_H
    temp = np.sqrt(temp)
    ret_val = int(temp)
    if (ret_val < 0):
        ret_val = 0
    if (ret_val >= BRDF_SAMPLING_RES_THETA_H):
        ret_val = BRDF_SAMPLING_RES_THETA_H-1
    return ret_val

# Lookup theta_half index
# This is a non-linear mapping!
# In:  [0 .. pi/2]
# Out: [0,1]
def theta_half_index_float(theta_half):
    theta_half = np.clip(theta_half,0,None)
    theta_half_deg = ((theta_half / (M_PI/2.0))*BRDF_SAMPLING_RES_THETA_H)
    temp = theta_half_deg*BRDF_SAMPLING_RES_THETA_H
    temp = np.sqrt(temp)
    ret_val = temp
    ret_val = np.clip(ret_val,0,BRDF_SAMPLING_RES_THETA_H)
    return ret_val

# Lookup theta_diff index
# In:  [0 .. pi/2]
# Out: [0 .. 89]
def theta_diff_index(theta_diff):
    tmp = int(theta_diff / (M_PI * 0.5) * BRDF_SAMPLING_RES_THETA_D)
    if (tmp < 0):
        return 0
    elif (tmp < BRDF_SAMPLING_RES_THETA_D - 1):
        return tmp
    else:
        return BRDF_SAMPLING_RES_THETA_D - 1

# Lookup phi_diff index
def phi_diff_index(phi_diff):
    # Because of reciprocity, the BRDF is unchanged under
    # phi_diff -> phi_diff + M_PI
    if (phi_diff < 0.0):
        phi_diff += M_PI

    # In: phi_diff in [0 .. pi]
    # Out: tmp in [0 .. 179]
    tmp = int(phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2)
    if (tmp < 0):	
        return 0
    elif (tmp < BRDF_SAMPLING_RES_PHI_D // 2 - 1):
        return tmp
    else:
        return BRDF_SAMPLING_RES_PHI_D // 2 - 1

# Lookup phi_diff index
def phi_diff_index_float(phi_diff):
    # Because of reciprocity, the BRDF is unchanged under
    # phi_diff -> phi_diff + M_PI
    phi_diff += M_PI * (phi_diff<0.0)

    tmp = phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2

    # In: phi_diff in [0 .. pi]
    # Out: tmp in [0 .. 179]
    return np.clip(tmp, 0, BRDF_SAMPLING_RES_PHI_D / 2)

def lookup_brdf_val(brdf, theta_in, fi_in, theta_out, fi_out): 
	# out : double& red_val,double& green_val,double& blue_val
    theta_half, fi_half, theta_diff, fi_diff = std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out)
    return brdf[theta_half_index(theta_half), theta_diff_index(theta_diff), phi_diff_index(fi_diff)]


def main():
    import matplotlib.pyplot as plt
    
    brdf = read_brdf('BRDFDatabase/brdfs/white-fabric.binary')

    fi_in = 0.0
    theta_in = np.radians(10)
    fi_out = fi_in + np.pi
    list_theta_out = np.arange(0.0, 0.5*np.pi, np.pi/180)
    brdf_values = []
    for theta_out in list_theta_out:
        brdf_values.append(lookup_brdf_val(brdf, theta_in, fi_in, theta_out, fi_out))
    brdf_values = np.array(brdf_values)
    for brdf_value, color in zip(brdf_values.T, ['r','g','b']):
        plt.plot(np.degrees(list_theta_out), brdf_value, color=color)
    plt.ylim([0.0, 1.2*np.amax(brdf_values)])
    plt.show()

if __name__ == '__main__':
    main()