import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution):
    head_box = np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution - 1)
    head_channel = np.zeros((resolution, resolution), dtype=np.float32)
    head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)

    return head_channel

def get_head_mask_inv(x_min, y_min, x_max, y_max, width, height, resolution):
    head_box = np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution - 1)
    head_channel = np.ones((resolution, resolution), dtype=np.float32)
    head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 0
    head_channel = torch.from_numpy(head_channel)

    return head_channel



def get_label_map(img, pt, sigma, pdf="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:

        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if pdf == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif pdf == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])
    #print(g[g_y[0] : g_y[1], g_x[0] : g_x[1]])
    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / np.max(img)  # normalize heatmap so it has max value of 1

    return to_torch(img)

def get_label_full_map(img,depth,pt):
    img = to_numpy(img)
    p_x = np.linspace(0, 224 - 1, 224)
    p_y = np.linspace(0, 224 - 1, 224)
    p_xv, p_yv = np.meshgrid(p_x, p_y)
    vx = p_xv - pt[0]
    vy = p_yv - pt[1]
    depth = depth-depth[pt[0],pt[1]]
    norm = np.sqrt(vx * vx + vy * vy + depth * depth)
    norm = 1 - norm/np.max(norm)
    img = img+norm
    return to_torch(img)


def get_gaze_feild(img,depth,direction,eye_x,eye_y,size, d = False ):
    img = to_numpy(img)

    p_x = np.linspace(0, 224 - 1, 224)
    p_y = np.linspace(0, 224 - 1, 224)
    p_xv, p_yv = np.meshgrid(p_x, p_y)
    p_yv = p_yv-eye_y
    p_xv = p_xv-eye_x
    p_d = depth - depth[int(eye_x),int(eye_y)]

    p_xv = p_xv[np.newaxis,:,:]
    p_yv = p_yv[np.newaxis, :, :]
    p_d = p_d[np.newaxis, :, :]

    i = to_torch(np.concatenate((p_xv,p_yv),0).transpose(2,1,0).reshape(224*224,2))
    #i = to_torch(np.concatenate((np.concatenate((p_xv, p_yv), 0),p_d),0).transpose(2,1,0).reshape(224 * 224, 3))
    print(to_torch(direction).shape)
    si = F.cosine_similarity(to_torch(direction), i).view(224,224).transpose(1,0).numpy()
    #si = 1/(1+np.exp(-si))

    si = np.int64(si > 0.8).astype(float)*si
    img = img+si
    #img = img / np.max(img)
    return to_torch(img)


def get_multi_hot_map(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int, [p[0] * w.float(), p[1] * h.float()])
            x = min(x, w - 1)
            y = min(y, h - 1)
            target_map[y, x] = 1

    return target_map


def get_auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(np.reshape(onehot_im, onehot_im.size), np.reshape(heatmap, heatmap.size))
    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score


def get_ap(label, pred):
    return average_precision_score(label, pred)


def get_heatmap_peak_coords(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float, idx)
    return pred_x, pred_y


def get_l2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_angular_error(p1, p2):
    norm_p1 = (p1[0] ** 2 + p1[1] ** 2) ** 0.5
    norm_p2 = (p2[0] ** 2 + p2[1] ** 2) ** 0.5
    cos_sim = (p1[0] * p2[0] + p1[1] * p2[1]) / (norm_p2 * norm_p1 + 1e-6)
    cos_sim = np.maximum(np.minimum(cos_sim, 1.0), -1.0)

    return np.arccos(cos_sim) * 180 / np.pi


def get_memory_format(config):
    if config.channels_last:
        return torch.channels_last

    return torch.contiguous_format
