####################################
### ZouJiu
### 20240429
### 1069679911@qq.com
# https://zoujiu.blog.csdn.net/
# https://zhihu.com/people/zoujiu1
# https://github.com/ZouJiu1
####################################

import os
import cv2
import numpy as np
from PIL import Image
abspath = os.path.abspath(__file__)
filename = abspath.split(os.sep)[-1]
abspath = abspath.replace(filename, "")
'''
https://fourcc.org/fccyvrgb.php
The following 2 sets of formulae are taken from information from Keith Jack's excellent book "Video Demystified" (ISBN 1-878707-09-4).
https://wiki.videolan.org/YUV/
https://video.matrox.com/en/media/guides-articles/introduction-color-spaces-video
'''

def I420_YV12(Y, U, V, YV12=True):
    u = []
    v = []
    YC = Y.copy()
    for h in range(height):
        for w in range(width):
            if w % 2==0 and h % 2==0:
                u.append(U[h, w])
                v.append(V[h, w])
    Y = list(Y.flatten())
    if YV12:
        ret_tuple = (Y, v, u)
    else:
        ret_tuple = (Y, u, v)
    
    picture = np.zeros((height // 2 * 3, width), dtype = np.uint8)
    picture_YV12 = np.zeros((height // 2 * 3, width), dtype = np.uint8)
    picture[:height, :] = YC
    picture_YV12[:height, :] = YC
    for h in range(height):
        for w in range(width):
            if w % 2==0 and h % 2==0:
                right = width//2 if ((h//2)%2)==1 else 0
                picture[height + h//4, w//2 + right] = U[h, w]
                picture[height + height//4 + h//4, w//2 + right] = V[h, w]
                
                picture_YV12[height + h//4, w//2 + right] = V[h, w]
                picture_YV12[height + height//4 + h//4, w//2 + right] = U[h, w]
                
    # if savebin:
    #     np.array(ret[0]).tofile(os.path.join(abspath, 'I420.bin'))
    return ret_tuple, picture, picture_YV12

def I422(Y, U, V):
    u = []
    v = []
    for h in range(height):
        for w in range(width):
            if w % 2==0:
                u.append(U[h, w])
                v.append(V[h, w])
    Y = list(Y.flatten())
    ret = (Y, u, v)
    return ret

def I444(Y, U, V):
    u = []
    v = []
    Y = list(Y.flatten())
    u = list(U.flatten())
    v = list(V.flatten())
    ret = (Y, u, v)
    return ret

def UYVY_YUY2_YVYU_VYUV_422(Y, U, V):
    UC = U.copy()
    VC = V.copy()
    for h in range(height):
        for w in range(width):
            if w % 2==1:
                UC[h, w] = VC[h, w - 1]
                VC[h, w] = UC[h, w - 1]

    UYVY = np.zeros((Y.shape[0], Y.shape[1], 2), dtype=np.uint8)
    UYVY[:, :, 0] = UC
    UYVY[:, :, 1] = Y

    YUY2 = np.zeros((Y.shape[0], Y.shape[1], 2), dtype=np.uint8)
    YUY2[:, :, 0] = Y
    YUY2[:, :, 1] = UC

    YVYU = np.zeros((Y.shape[0], Y.shape[1], 2), dtype=np.uint8)
    YVYU[:, :, 0] = Y
    YVYU[:, :, 1] = VC
    
    u = []
    v = []
    Y = list(Y.flatten())
    for h in range(height):
        for w in range(width):
            if w % 2==0:
                u.append(U[h, w])
                v.append(V[h, w])
    UYVY_plane = []
    lu = 0
    lv = 0
    ly = 0
    pre = 0
    while True:
        if lu < len(u):
            UYVY_plane.append(u[lu])
            lu += 1
        if ly < len(Y):
            UYVY_plane.append(Y[ly])
            ly += 1
        if lv < len(v):
            UYVY_plane.append(v[lv])
            lv += 1
        if ly < len(Y):
            UYVY_plane.append(Y[ly])
            ly += 1

        if pre==len(UYVY_plane):
            break
        pre = len(UYVY_plane)

    YUY2_plane = []
    lu = 0
    lv = 0
    ly = 0
    pre = 0
    while True:
        if ly < len(Y):
            YUY2_plane.append(Y[ly])
            ly += 1
        if lu < len(u):
            YUY2_plane.append(u[lu])
            lu += 1
        if ly < len(Y):
            YUY2_plane.append(Y[ly])
            ly += 1
        if lv < len(v):
            YUY2_plane.append(v[lv])
            lv += 1

        if pre==len(YUY2_plane):
            break
        pre = len(YUY2_plane)

    YVYU_plane = []
    lu = 0
    lv = 0
    ly = 0
    pre = 0
    while True:
        if ly < len(Y):
            YVYU_plane.append(Y[ly])
            ly += 1
        if lv < len(v):
            YVYU_plane.append(v[lv])
            lv += 1
        if ly < len(Y):
            YVYU_plane.append(Y[ly])
            ly += 1
        if lu < len(u):
            YVYU_plane.append(u[lu])
            lu += 1

        if pre==len(YVYU_plane):
            break
        pre = len(YVYU_plane)

    # if savebin:
    #     np.array(UYVY_plane).tofile(os.path.join(abspath, 'UYVY_plane.bin'))
    return (UYVY_plane, YUY2_plane, YVYU_plane, UYVY, YUY2, YVYU)

def NV12_NV21_420(Y, U, V):
    u = []
    v = []
    for h in range(height):
        for w in range(width):
            if w % 2==0 and h % 2==0:
                u.append(U[h, w])
                v.append(V[h, w])
    Y = list(Y.flatten())
    
    tmp12 = []
    lu = 0
    lv = 0
    pre = 0
    while True:
        if lu < len(u):
            tmp12.append(u[lu])
            lu += 1
        if lv < len(v):
            tmp12.append(v[lv])
            lv += 1

        if pre==len(tmp12):
            break
        pre = len(tmp12)

    NV12 = [Y, tmp12]

    tmp21 = []
    lu = 0
    lv = 0
    pre = 0
    while True:
        if lv < len(v):
            tmp21.append(v[lv])
            lv += 1
        if lu < len(u):
            tmp21.append(u[lu])
            lu += 1

        if pre==len(tmp21):
            break
        pre = len(tmp21)
    
    NV21 = (Y, tmp21)
    if savebin:
        np.array(NV12[0] + NV12[1], dtype=np.uint8).tofile(os.path.join(abspath, 'NV12.bin'))
        np.array(NV21[0] + NV21[1], dtype=np.uint8).tofile(os.path.join(abspath, 'NV21.bin'))
    return (NV12, NV21)

def NV16_NV61_422(Y, U, V):
    u = []
    v = []
    for h in range(height):
        for w in range(width):
            if w % 2==0:
                u.append(U[h, w])
                v.append(V[h, w])
    Y = list(Y.flatten())
    
    tmp16 = []
    lu = 0
    lv = 0
    pre = 0
    while True:
        if lu < len(u):
            tmp16.append(u[lu])
            lu += 1
        if lv < len(v):
            tmp16.append(v[lv])
            lv += 1

        if pre==len(tmp16):
            break
        pre = len(tmp16)

    NV16 = [Y, tmp16]

    tmp61 = []
    lu = 0
    lv = 0
    pre = 0
    while True:
        if lv < len(v):
            tmp61.append(v[lv])
            lv += 1
        if lu < len(u):
            tmp61.append(u[lu])
            lu += 1

        if pre==len(tmp61):
            break
        pre = len(tmp61)
    
    NV61 = (Y, tmp61)

    return (NV16, NV61)

def NV24_NV42_444(Y, U, V):
    u = list(U.flatten())
    v = list(V.flatten())
    Y = list(Y.flatten())
    
    tmp24 = []
    lu = 0
    lv = 0
    pre = 0
    while True:
        if lu < len(u):
            tmp24.append(u[lu])
            lu += 1
        if lv < len(v):
            tmp24.append(v[lv])
            lv += 1

        if pre==len(tmp24):
            break
        pre = len(tmp24)

    NV24 = [Y, tmp24]

    tmp42 = []
    lu = 0
    lv = 0
    pre = 0
    while True:
        if lv < len(v):
            tmp42.append(v[lv])
            lv += 1
        if lu < len(u):
            tmp42.append(u[lu])
            lu += 1

        if pre==len(tmp42):
            break
        pre = len(tmp42)
    
    NV42 = (Y, tmp42)

    return (NV24, NV42)

def rgb2yuv444():
    Y = np.zeros((height, width), dtype = np.float32)
    U = np.zeros_like(img, dtype = np.float32)
    V = np.zeros_like(img, dtype = np.float32)
    for h in range(height):
        for w in range(width):
            R, G, B = img[h, w, :]
            Y[h, w] = round( 0.257 * R + 0.504 * G + 0.098 * B) + 16
            U[h, w, 0] = round(-0.148 * R - 0.291 * G + 0.439 * B) + 128   # blue projection
            V[h, w, 2] = round( 0.439 * R - 0.368 * G - 0.071 * B) + 128    # red projection
    Y[Y > 255] = 255; Y[Y < 0] = 0
    U[U > 255] = 255; U[U < 0] = 0
    V[V > 255] = 255; V[V < 0] = 0
    Y = Y.astype(np.uint8); U = U.astype(np.uint8); V = V.astype(np.uint8)
    
    cv2.imwrite(os.path.join(savepath, r'444Y_rgb2yuv.jpg'), Y)
    cv2.imwrite(os.path.join(savepath, r'444U_blue_projection.png'), U)
    cv2.imwrite(os.path.join(savepath, r'444V_red_projection.png'), V)

    list_I444 = I444(Y, U[:, :, 0], V[:, :, 2])
    NV24, NV42 = NV24_NV42_444(Y, U[:, :, 0], V[:, :, 2])
    
    return list_I444, NV24, NV42

def rgb2yuv422():
    # https://github.com/opencv/opencv/blob/4.x/modules/imgproc/test/test_cvtyuv.cpp
    Y = np.zeros((height, width), dtype = np.float32)
    U = np.zeros_like(img, dtype = np.float32)
    V= np.zeros_like(img, dtype = np.float32)
    for h in range(height):
        for w in range(0, width, 2):
            R, G, B = img[h, w, :]
            Rn, Gn, Bn = img[h, w + 1, :]
            Y[h, w] = round(0.257 * R + 0.504 * G + 0.098 * B + 16)
            Y[h, w + 1] = round(0.257 * Rn + 0.504 * Gn + 0.098 * Bn + 16)
            U[h, w, 0] = round(-0.148 * (R + Rn) / 2.0 - 0.291 * (G + Gn) / 2.0 + 0.439 * (B + Bn) / 2.0 + 128.0)   # blue projection
            V[h, w, 2] = round(0.439 * (R + Rn) / 2.0 - 0.368 * (G + Gn) / 2.0 - 0.071 * (B + Bn) / 2.0 + 128.0)    # red projection
    Y[Y > 255] = 255; Y[Y < 0] = 0
    U[U > 255] = 255; U[U < 0] = 0
    V[V > 255] = 255; V[V < 0] = 0
    Y = Y.astype(np.uint8); U = U.astype(np.uint8); V = V.astype(np.uint8)
    
    cv2.imwrite(os.path.join(savepath, r'422Y_rgb2yuv.jpg'), Y)
    cv2.imwrite(os.path.join(savepath, r'422U_blue_projection.png'), U)
    cv2.imwrite(os.path.join(savepath, r'422V_red_projection.png'), V)

    list_I422 = I422(Y, U[:, :, 0], V[:, :, 2])
    UYVY_plane, YUY2_plane, YVYU_plane, UYVY, YUY2, YVYU = UYVY_YUY2_YVYU_VYUV_422(Y, U[:, :, 0], V[:, :, 2])
    NV16, NV61 = NV16_NV61_422(Y, U[:, :, 0], V[:, :, 2])

    return list_I422, UYVY_plane, YUY2_plane, YVYU_plane, UYVY, YUY2, YVYU, NV16, NV61

def rgb2yuv420():
    Y = np.zeros((height, width), dtype = np.float32)
    U = np.zeros_like(img, dtype = np.float32)
    V= np.zeros_like(img, dtype = np.float32)
    for h in range(height):
        for w in range(width):
            R, G, B = img[h, w, :]
            Y[h, w] = round( 0.257 * R + 0.504 * G + 0.098 * B) + 16
            if w % 2==0 and h % 2==0:
                U[h, w, 0] = round(-0.148 * R - 0.291 * G + 0.439 * B) + 128   # blue projection
                V[h, w, 2] = round( 0.439 * R - 0.368 * G - 0.071 * B) + 128    # red projection
    Y[Y > 255] = 255; Y[Y < 0] = 0
    U[U > 255] = 255; U[U < 0] = 0
    V[V > 255] = 255; V[V < 0] = 0
    Y = Y.astype(np.uint8); U = U.astype(np.uint8); V = V.astype(np.uint8)

    cv2.imwrite(os.path.join(savepath, r'420Y_rgb2yuv.jpg'), Y)
    cv2.imwrite(os.path.join(savepath, r'420U_blue_projection.png'), U)
    cv2.imwrite(os.path.join(savepath, r'420V_red_projection.png'), V)

    I420_YV12_tuple, I420, YV12 = I420_YV12(Y, U[:, :, 0], V[:, :, 2], YV12=False)
    NV12, NV21 = NV12_NV21_420(Y, U[:, :, 0], V[:, :, 2])
    
    return I420_YV12_tuple, I420, YV12, NV12, NV21

def yuv444_2_rgb():
    Y = cv2.imread(os.path.join(savepath, r'444Y_rgb2yuv.jpg'), 0)
    U = cv2.imread(os.path.join(savepath, r'444U_blue_projection.png'))[:, :, 0]
    V = cv2.imread(os.path.join(savepath, r'444V_red_projection.png'))[:, :, 2]
    imgnow = np.zeros_like(img, dtype=np.float32)
    for h in range(height):
        for w in range(width):
            y, u, v = Y[h, w], U[h, w], V[h, w]
            y = max(0, y - 16)
            u = u - 128
            v = v - 128
            R = 1.164 * y + 1.596 * v
            G = 1.164 * y - 0.813 * v - 0.391 * u
            B = 1.164 * y + 2.018 * u
            imgnow[h, w, :] = [B, G, R]
    imgnow[imgnow > 255] = 255
    imgnow[imgnow < 0] = 0
    imgnow = imgnow.astype(np.uint8)
    cv2.imwrite(os.path.join(savepath, r'yuv444_2_rgb.jpg'), imgnow)
    if Togray:
        return Y
    return None
    
def yuv422_2_rgb():
    Y = cv2.imread(os.path.join(savepath, r'422Y_rgb2yuv.jpg'), 0)
    U = cv2.imread(os.path.join(savepath, r'422U_blue_projection.png'))[:, :, 0]
    V = cv2.imread(os.path.join(savepath, r'422V_red_projection.png'))[:, :, 2]
    imgnow = np.zeros_like(img, dtype=np.float32)
    for h in range(height):
        for w in range(width):
            y, u, v = Y[h, w], U[h, w], V[h, w]
            y = max(0, y - 16)
            if w % 2==1:
                u = U[h, w - 1]
                v = V[h, w - 1]
            u = u - 128
            v = v - 128
            R = 1.164 * y + 1.596 * v
            G = 1.164 * y - 0.813 * v - 0.391 * u
            B = 1.164 * y + 2.018 * u
            imgnow[h, w, :] = [B, G, R]
    imgnow[imgnow > 255] = 255
    imgnow[imgnow < 0] = 0
    imgnow = imgnow.astype(np.uint8)
    cv2.imwrite(os.path.join(savepath, r'yuv422_2_rgb.jpg'), imgnow)
    if Togray:
        return Y
    return None

def yuv420_2_rgb():
    Y = cv2.imread(os.path.join(savepath, r'420Y_rgb2yuv.jpg'), 0)
    U = cv2.imread(os.path.join(savepath, r'420U_blue_projection.png'))[:, :, 0]
    V = cv2.imread(os.path.join(savepath, r'420V_red_projection.png'))[:, :, 2]
    imgnow = np.zeros_like(img, dtype=np.float32)
    for h in range(height):
        h_sub = (1 if h%2!=0 else 0)
        for w in range(width):
            y, u, v = Y[h, w], U[h, w], V[h, w]
            y = max(0, y - 16)
            if w % 2!=0 or h % 2!= 0:
                w_sub = (1 if w%2!=0 else 0)
                u = U[h - h_sub, w - w_sub]
                v = V[h - h_sub, w - w_sub]
            u = u - 128
            v = v - 128
            R = 1.164 * y + 1.596 * v
            G = 1.164 * y - 0.813 * v - 0.391 * u
            B = 1.164 * y + 2.018 * u
            imgnow[h, w, :] = [B, G, R]
    imgnow[imgnow > 255] = 255
    imgnow[imgnow < 0] = 0
    imgnow = imgnow.astype(np.uint8)
    cv2.imwrite(os.path.join(savepath, r'yuv420_2_rgb.jpg'), imgnow)
    if Togray:
        return Y
    return None

def generate():
    img = np.zeros((2, 4, 2+1), dtype=np.uint8)
    img[1, 2, :] = img[0, 3, :] = img[0, 0, :] = [200, 0, 0]
    img[1, 3, :] = img[1, 0, :] = img[0, 1, :] = [0, 0, 200]
    img[1, 1, :] = img[0, 2, :] = [0, 200, 0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def readbin_2_image():
    pth = os.path.join(abspath, 'NV21.bin')

def create_blue_red_projection_422(UV_cross, VU_cross):
    UV = np.zeros_like(img, dtype=np.uint8)
    VU = np.zeros_like(img, dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            if w%2==0:
                UV[h, w, 0] = UV_cross[h, w]
                VU[h, w, 2] = VU_cross[h, w]
            else:
                UV[h, w, 2] = UV_cross[h, w]
                VU[h, w, 0] = VU_cross[h, w]
    cv2.imwrite(os.path.join(savepath, '422_UV.jpg'), UV)
    cv2.imwrite(os.path.join(savepath, '422_VU.jpg'), VU)

def create_blue_red_projection_420(I420, YV12):
    out_I420 = np.zeros((height//2*3, width, 3), dtype=np.uint8)
    out_YV12 = np.zeros((height//2*3, width, 3), dtype=np.uint8)
    out_I420[:height, :, :] = np.stack([I420[:height, :], I420[:height, :], I420[:height, :]], axis = 2)
    out_YV12[:height, :, :] = np.stack([YV12[:height, :], YV12[:height, :], YV12[:height, :]], axis = 2)
    for h in range(height, height//2*3):
        for w in range(width):
            if h < (height + height//4):
                out_I420[h, w, 0] = I420[h, w]
                out_YV12[h, w, 2] = YV12[h, w]
            else:
                out_I420[h, w, 2] = I420[h, w]
                out_YV12[h, w, 0] = YV12[h, w]
    cv2.imwrite(os.path.join(savepath, '420_I420.jpg'), out_I420)
    cv2.imwrite(os.path.join(savepath, '420_YV12.jpg'), out_YV12)

def create_fromNV12_NV21_420():
    nv12pth = os.path.join(abspath, 'NV12.bin')
    nv21pth = os.path.join(abspath, 'NV21.bin')
    
    nv12 = np.fromfile(nv12pth, dtype=np.uint8)
    nv21 = np.fromfile(nv21pth, dtype=np.uint8)
    
    nv12_img = np.zeros((height//2*3, width, 3), dtype=np.uint8)
    nv21_img = nv12_img.copy()
    Y_12 = nv12[:height * width].reshape((height, width))
    UV_12 = nv12[height*width:].reshape((height//2, width))
    gray_12 = np.concatenate([Y_12, UV_12], axis = 0)
    
    Y_21 = nv21[:height * width].reshape((height, width))
    VU_21 = nv21[height*width:].reshape((height//2, width))
    gray_21 = np.concatenate([Y_21, VU_21], axis = 0)
    
    nv12_img[:height, :, :] = np.stack([Y_12, Y_12, Y_12], axis = 2)
    nv21_img[:height, :, :] = np.stack([Y_21, Y_21, Y_21], axis = 2)
    
    for h in range(height, height//2*3):
        hi = h - height
        for w in range(0, width, 2):
            nv12_img[h, w, 0] = UV_12[hi, w]
            nv12_img[h, w+1, 2] = UV_12[hi, w+1]
            nv21_img[h, w, 2] = VU_21[hi, w]
            nv21_img[h, w+1, 0] = VU_21[hi, w+1]

    cv2.imwrite(os.path.join(savepath, '420_NV12.jpg'), nv12_img)
    cv2.imwrite(os.path.join(savepath, '420_NV21.jpg'), nv21_img)
    return gray_12, gray_21

if __name__=='__main__':
    imgpath = os.path.join(abspath, "picture", 'sunoray.png')
    # imgpath = os.path.join(abspath, "picture", 'cgi0.png')
    savepath = os.path.join(abspath, "picture")
    savebin = True
    Togray = True
    
    img = Image.open(imgpath).convert("RGB")
    img = np.asarray(img).copy().astype(np.float32)
    # img = generate()
    # cv2.imwrite(os.path.join(abspath, "picture", "gen.png"), img)
    height, width, channel = img.shape
    num_pixel = height * width * channel
    UYVY = True
    YUV = True
    # https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color.simd_helpers.hpp  # CvtHelper
    if YUV:
        assert height % 2 == 0 and width % 2 == 0
    elif UYVY:
        assert width % 2 == 0

    ############################ rgb 2 yuv444 ######################################
    I444_, NV24, NV42 = rgb2yuv444()

    ############################ rgb 2 yuv422 ######################################
    '''
        https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp
        //! RGB to YUV 4:2:2 family
        COLOR_RGB2YUV_UYVY = 143,
        COLOR_RGB2YUV_Y422 = COLOR_RGB2YUV_UYVY,
        COLOR_RGB2YUV_UYNV = COLOR_RGB2YUV_UYVY,

        COLOR_RGB2YUV_YUY2 = 147,
        COLOR_RGB2YUV_YVYU = 149,
        COLOR_RGB2YUV_YUYV = COLOR_RGB2YUV_YUY2,
        COLOR_RGB2YUV_YUNV = COLOR_RGB2YUV_YUY2,

        //! YUV 4:2:2 family to RGB
        COLOR_YUV2RGB_UYVY = 107,
        //COLOR_YUV2RGB_VYUY = 109,
        COLOR_YUV2RGB_Y422 = COLOR_YUV2RGB_UYVY,
        COLOR_YUV2RGB_UYNV = COLOR_YUV2RGB_UYVY,

        COLOR_YUV2RGB_YUY2 = 115,
        COLOR_YUV2RGB_YVYU = 117,
        COLOR_YUV2RGB_YUYV = COLOR_YUV2RGB_YUY2,
        COLOR_YUV2RGB_YUNV = COLOR_YUV2RGB_YUY2,

        COLOR_YUV2RGBA_YUY2 = 119,
        COLOR_YUV2RGBA_YVYU = 121,
        COLOR_YUV2RGBA_YUYV = COLOR_YUV2RGBA_YUY2,
        COLOR_YUV2RGBA_YUNV = COLOR_YUV2RGBA_YUY2,
    '''
    CV_UYVY = cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_RGB2YUV_UYVY)
    CV_YUY2 = cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_RGB2YUV_YUY2)
    CV_YVYU = cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_RGB2YUV_YVYU)

    I422_, UYVY_plane, YUY2_plane, YVYU_plane, UYVY, YUY2, YVYU, NV16, NV61 = rgb2yuv422()
    
    recover_UYVY = cv2.cvtColor(UYVY, cv2.COLOR_YUV2BGR_UYVY)
    recover_YUY2 = cv2.cvtColor(YUY2, cv2.COLOR_YUV2BGR_YUY2)
    recover_YVYU = cv2.cvtColor(YVYU, cv2.COLOR_YUV2BGR_YVYU)
    cv2.imwrite(os.path.join(savepath, '422_UYVY_recover.jpg'), recover_UYVY)
    cv2.imwrite(os.path.join(savepath, '422_YUY2_recover.jpg'), recover_YUY2)
    cv2.imwrite(os.path.join(savepath, '422_YVYU_recover.jpg'), recover_YVYU)
    
    UYVY_f = np.array(UYVY, dtype=np.float32)
    YUY2_f = np.array(YUY2, dtype=np.float32)
    YVYU_f = np.array(YVYU, dtype=np.float32)
    CV_UYVY_f = np.array(CV_UYVY, dtype=np.float32)
    CV_YUY2_f = np.array(CV_YUY2, dtype=np.float32)
    CV_YVYU_f = np.array(CV_YVYU, dtype=np.float32)
    assert (np.max(UYVY_f - CV_UYVY_f) <= 1.0 and np.sum(UYVY != CV_UYVY) / num_pixel < 0.006), \
        (np.max(UYVY_f - CV_UYVY_f), np.sum(UYVY != CV_UYVY) / num_pixel)
    assert (np.max(YUY2_f - CV_YUY2_f) <= 1.0 and np.sum(YUY2 != CV_YUY2) / num_pixel < 0.006), \
        (np.max(YUY2_f - CV_YUY2_f), np.sum(YUY2 != CV_YUY2) / num_pixel)
    assert (np.max(YVYU_f - CV_YVYU_f) <= 1.0 and np.sum(YVYU != CV_YVYU) / num_pixel < 0.006), \
        (np.max(YVYU_f - CV_YVYU_f), np.sum(YVYU != CV_YVYU) / num_pixel)
    del UYVY_f, YUY2_f, YVYU_f, CV_UYVY_f, CV_UYVY, CV_YUY2, CV_YUY2_f, CV_YVYU, CV_YVYU_f
    
    create_blue_red_projection_422(UYVY[:, :, 0], YVYU[:, :, 1])

    ############################ rgb 2 yuv420 ######################################
    '''
        //! RGB to YUV 4:2:0 family
        COLOR_RGB2YUV_I420  = 127,
        COLOR_RGB2YUV_IYUV  = COLOR_RGB2YUV_I420,
        COLOR_RGB2YUV_YV12  = 131,
    '''
    CV_I420 = cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_RGB2YUV_I420)
    CV_YV12 = cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_RGB2YUV_YV12)
    
    I420_YV12_tuple, I420, YV12, NV12, NV21 = rgb2yuv420()
    
    I420_f = np.array(I420, dtype=np.float32)
    YV12_f = np.array(YV12, dtype=np.float32)
    CV_I420_f = np.array(CV_I420, dtype=np.float32)
    CV_YV12_f = np.array(CV_YV12, dtype=np.float32)
    num_pixel = (height//2*3) * width * 3
    assert (np.max(I420_f - CV_I420_f) <= 1.0 and np.sum(I420 != CV_I420) / num_pixel < 0.006), \
        (np.max(I420_f - CV_I420_f), np.sum(I420 != CV_I420) / num_pixel)
    assert (np.max(YV12_f - CV_YV12_f) <= 1.0 and np.sum(YV12 != CV_YV12) / num_pixel < 0.006), \
        (np.max(YV12_f - CV_YV12_f), np.sum(YV12 != CV_YV12) / num_pixel)
    del CV_I420, CV_I420_f, CV_YV12, CV_YV12_f, I420_f, YV12_f
    
    create_blue_red_projection_420(I420, YV12)
    
    gray_12, gray_21 = create_fromNV12_NV21_420()
    # https://github.com/opencv/opencv/blob/4.x/modules/imgproc/test/test_cvtyuv.cpp
    recover12 = cv2.cvtColor(gray_12, cv2.COLOR_YUV2BGR_NV12)
    recover21 = cv2.cvtColor(gray_21, cv2.COLOR_YUV2BGR_NV21)
    recover_I420 = cv2.cvtColor(I420, cv2.COLOR_YUV2BGR_I420)
    recover_YV12 = cv2.cvtColor(YV12, cv2.COLOR_YUV2BGR_YV12)
    cv2.imwrite(os.path.join(savepath, '420_NV12_recover.jpg'), recover12)
    cv2.imwrite(os.path.join(savepath, '420_NV21_recover.jpg'), recover21)
    cv2.imwrite(os.path.join(savepath, '420_I420_recover.jpg'), recover_I420)
    cv2.imwrite(os.path.join(savepath, '420_YV12_recover.jpg'), recover_YV12)



    ############################ yuv 2 rgb ######################################
    Img_gray0 = yuv444_2_rgb()
    Img_gray1 = yuv422_2_rgb()
    Img_gray2 = yuv420_2_rgb()