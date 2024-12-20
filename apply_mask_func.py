from blend_modes import multiply
import cv2
import numpy as np

def make_black_transparent(img):
    # Convert image to image gray
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Applying thresholding technique
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    
    # Using cv2.split() to split channels 
    # of coloured image
    b, g, r = cv2.split(img)
    
    # Making list of Red, Green, Blue
    # Channels and alpha
    rgba = [b, g, r, alpha]
    
    # Using cv2.merge() to merge rgba
    # into a coloured/multi-channeled image
    dst = cv2.merge(rgba, 4)
    return dst

def blend_images(img1, img2):
    print( 'debug before blend' )
    print( img1.shape )
    print( img2.shape )
    result = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    return result

def add_alpha_channel(img):
    b_channel, g_channel, r_channel = cv2.split(img)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.

    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA

def align_image_size(bm_img, t_img):
    bm_dimension = bm_img.shape
    t_dimension = t_img.shape

    print('body mask dimension: width {}, height {}'.format(str(bm_dimension[1]), str(bm_dimension[0]) ))
    print('tattoo dimension: width {}, height {}'.format( str(t_dimension[1]), str(t_dimension[0]) ))

    if t_dimension[1] == bm_dimension[1] and t_dimension[0] == bm_dimension[0]:
        pass
    elif t_dimension[1] > bm_dimension[1] and t_dimension[0] > bm_dimension[0]:
        # Scale down tattoo image
        scaler = bm_dimension[1] / t_dimension[1]
        new_dimension = (int(t_dimension[1]*scaler), int(t_dimension[0]*scaler))
        print('Debug new dimension: ')
        print( new_dimension )
        t_img = cv2.resize(t_img, new_dimension, interpolation=cv2.INTER_AREA)
        t_img = t_img[0:int(bm_dimension[1]), 0:int(bm_dimension[0])]
    else:
        # Scale down body mask image
        scaler = t_dimension[1] / bm_dimension[1]
        new_dimension = (int(bm_dimension[1]*scaler), int(bm_dimension[0]*scaler))
        bm_img = cv2.resize(bm_img, new_dimension, interpolation=cv2.INTER_AREA)
        bm_img = bm_img[0:int(t_dimension[1]), 0:int(t_dimension[0])]

    print('body mask adjusted dimension: width {}, height {}'.format(str(bm_img.shape[1]), str(bm_img.shape[0])))
    print('tattoo adjusted dimension: width {}, height {}'.format(str(t_img.shape[1]), str(t_img.shape[0])))

    return bm_img, t_img

def create_body_mask(bm_img):
    hsv = cv2.cvtColor(bm_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,168]), np.array([172,111,255]))
    return mask


def apply_mask(body_mask_image, tattoo_image, body_image=None):
    if body_mask_image.shape[2]==3:
        body_mask_image = add_alpha_channel(body_mask_image)
    
    if tattoo_image.shape[2]==3:
        tattoo_image = add_alpha_channel(tattoo_image)
    
    if body_image is not None:
        if body_image.shape[2] == 3:
                body_image = add_alpha_channel(body_image)
    # print( body_mask_image )
    # print( tattoo_image )
    # final_img = None
    body_mask_image, tattoo_image = align_image_size(body_mask_image, tattoo_image)

    body_mask = create_body_mask(body_mask_image)
    print('body mask shape')
    print(body_mask.shape)
    masked_tattoo_img = cv2.bitwise_and(tattoo_image, tattoo_image, mask=body_mask)
    print('masked tattoo img')
    print(masked_tattoo_img.shape)

    if body_image is not None:
        body_image, masked_tattoo_img = align_image_size(body_image, masked_tattoo_img)

    # Produce a black and white of the tattoo	
    gray_masked_tattoo_img = cv2.cvtColor(masked_tattoo_img, cv2.COLOR_BGRA2GRAY)
    gray_masked_tattoo_img = cv2.cvtColor(gray_masked_tattoo_img, cv2.COLOR_GRAY2RGB)
    gray_masked_tattoo_img = make_black_transparent(gray_masked_tattoo_img)
    print('debug gray masked tattoo image')
    print( gray_masked_tattoo_img.shape )

    if body_image is not None:
        final_img = blend_images(body_image, masked_tattoo_img)

        gray_final_img = blend_images(body_image, gray_masked_tattoo_img)
        
        body_image_float = body_image.astype(float)
        masked_tattoo_img_float = masked_tattoo_img.astype(float)
        gray_masked_tattoo_img_float = gray_masked_tattoo_img.astype(float)
        multi_mode_final_img = multiply(body_image_float, masked_tattoo_img_float, 1)
        multi_mode_final_img = multi_mode_final_img.astype(np.uint8)

        multi_mode_gray_final_img = multiply(body_image_float, gray_masked_tattoo_img_float, 0.9)
        multi_mode_gray_final_img = multi_mode_gray_final_img.astype(np.uint8)

        return final_img, masked_tattoo_img, gray_masked_tattoo_img, multi_mode_final_img, multi_mode_gray_final_img
    else:
        return masked_tattoo_img, gray_masked_tattoo_img

def pure_apply_mask(body_mask_image, tattoo_image):
    if body_mask_image.shape[2]==3:
        body_mask_image = add_alpha_channel(body_mask_image)
    
    if tattoo_image.shape[2]==3:
        tattoo_image = add_alpha_channel(tattoo_image)
    
    body_mask_image, tattoo_image = align_image_size(body_mask_image, tattoo_image)

    body_mask = create_body_mask(body_mask_image)
    print('body mask shape')
    print(body_mask.shape)
    masked_tattoo_img = cv2.bitwise_and(tattoo_image, tattoo_image, mask=body_mask)
    print('masked tattoo img')
    print(masked_tattoo_img.shape)

    return masked_tattoo_img