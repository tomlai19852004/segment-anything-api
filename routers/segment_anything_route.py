import torch
import cv2
import supervision as sv
import numpy as np
import time

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from fastapi import APIRouter, Request
from pydantic import BaseModel
from common_func import \
     download_img, generate_local_path, \
     encode_img_to_base64, data_uri_to_cv2_img, \
     decode_base64_to_img
from apply_mask_func import apply_mask, pure_apply_mask
from math import floor
from PIL import Image, ImageOps, ExifTags
from PIL.ExifTags import TAGS
# from urllib.request import Request, urlopen

router = APIRouter()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Price device')
print( DEVICE )
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)
mask_annotator = sv.MaskAnnotator()
mask_predictor = SamPredictor(sam)

debug_mode = True

class SegmentTestInput(BaseModel):
    body_img_url: str | None = None
    tattoo_img_url: str | None = None
    bounding_box: list | None = None
    input_labels: list | None = None
    input_points: list | None = None
    body_image: str | None = None
    debug_path: str | None = None
    extension: str | None = None
    

@router.post('/segment-anything-predict')
async def segment_anything_predict(segment_input: SegmentTestInput):
    start_time = time.time()
    global mask_predictor
    # print('Debug input')
    # print( segment_input )
    has_data = False
    if segment_input.body_img_url:
        file_name, local_file_path = generate_local_path(segment_input.body_img_url)
        # file_name_arr = file_name.split('.')
        download_img(segment_input.body_img_url, local_file_path)
        image_bgr = cv2.imread(local_file_path)[:,:,::-1]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
       
        has_data = True    
    elif segment_input.body_image:
        
        # Only this is actually being used
        image_bgr = decode_base64_to_img(segment_input.body_image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # pil_img = Image.open(segment_input.body_image)
        # exif = {
        #     TAGS[k]: v
        #     for k, v in pil_img._getexif().items()
        #     if k in TAGS
        # }
        # print( exif )
        has_data = True

    if segment_input.debug_path and has_data:
        debug_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        file_name_arr = segment_input.debug_path.split('.')
        for ii, point in enumerate(segment_input.input_points):
            color = (0,255,255) if segment_input.input_labels[ii]==1 else (255,255,0)
            # print( point )
            # print( color )
            debug_img = cv2.circle(debug_img, (point[0], point[1]), radius=3, color=color, thickness=-1)
        cv2.imwrite('./cache/{}-debug.{}'.format( file_name_arr[0], segment_input.extension ), debug_img)

    payload = { "results": [], 'scores': [] }
    if has_data and segment_input.input_labels and segment_input.input_points:
        mask_predictor.set_image(image_rgb)
        mask, scores, logits = mask_predictor.predict(
            point_coords=np.array(segment_input.input_points),
            point_labels=np.array(segment_input.input_labels),
            multimask_output=True,
        )

        print( scores )
        # print( logits )
        target_index = np.where(scores == max(scores))[0][0]
        body_mask_image = None
        for ii, m in enumerate(mask):
            if ii == target_index:
                body_mask_image = m.astype(np.uint8) * 255
                body_mask_image = cv2.cvtColor(body_mask_image, cv2.COLOR_GRAY2RGBA)
                base64_result_mask = encode_img_to_base64(body_mask_image)
                if segment_input.debug_path and len(file_name_arr)>1:
                    cv2.imwrite('./cache/{}-{}.{}'.format(file_name_arr[0], str(scores[ii]), file_name_arr[-1]), body_mask_image)
                payload['results'].append(base64_result_mask)
                payload['scores'].append(float(scores[ii]))

        if segment_input.debug_path and debug_img is not None:
            payload['debug'] = encode_img_to_base64(debug_img)
    print('Time needed: {}'.format(str(time.time() - start_time)))
    return payload


class SegmentInput(BaseModel):
    body_img_url: str | None = None
    tattoo_img_url: str | None = None
    bounding_box: list | None = None
    show_result: bool | None = None
    show_mask_tattoo: bool | None = None
    body_img: str | None = None

@router.post("/segment-anything")
def segment_anything(segment_input: SegmentInput):
    start_time = time.time()
    print('debug input')
    print( segment_input )
    global mask_annotator    
    global mask_generator
    global mask_predictor

    has_data = False
    if segment_input.body_img_url:
        file_name, local_file_path = generate_local_path(segment_input.body_img_url)
        file_name_arr = file_name.split('.')
        download_img(segment_input.body_img_url, local_file_path)
        image_bgr = cv2.imread(local_file_path)[:,:,::-1]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        has_data = True
    elif segment_input.body_img:
        image_bgr = data_uri_to_cv2_img(segment_input.body_img)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        has_data = True

    payload = { "results": [] }
    if has_data:
        if segment_input.tattoo_img_url:
            tat_file_name, tat_local_file_path = generate_local_path(segment_input.tattoo_img_url)
            tat_file_name_arr = tat_file_name.split('.')
            download_img(segment_input.tattoo_img_url, tat_local_file_path)
            tat_image = cv2.imread(tat_local_file_path, cv2.IMREAD_GRAYSCALE)
            print( "first debug tat image")
            print( tat_image.shape )

            edged = cv2.Canny(tat_image, 30, 200)
            # cv2.waitKey(0)

            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            print("Number of Contours found = " + str(len(contours)))           
            # Draw all contours
            # -1 signifies drawing all contours
            cv2.drawContours(tat_image, contours, -1, (0, 255, 0), 3)
            # print( zip(contours, hierarchy) )
            min_x, min_y, max_x, max_y = False, False, False, False
            for contour in contours:
                (x,y,w,h) = cv2.boundingRect(contour)
                # print('debug bounding box')
                # print( (x,y,w,h) )
                if w > 30 and h > 30:
                    if min_x == False or x < min_x:
                        min_x = x
                    if min_y == False or y < min_y:
                        min_y = y
                    if max_x == False or (w+x) > max_x:
                        max_x = w+x
                    if max_y == False or (h+y) > max_y:
                        max_y = h+y
                
            cv2.rectangle(tat_image, (min_x,min_y), (max_x,max_y), (255, 0, 0), 2)
            cv2.imwrite('./cache/{}-mask.{}'.format(file_name_arr[0], file_name_arr[-1]), tat_image)

            mask_predictor.set_image(image_rgb)
            box = np.array([min_x, min_y, max_x, max_y]) 
            print(box)
            print( [floor((max_x+min_x)/2), floor((max_y+min_y)/2)])
            sam_points = np.array([[floor((max_x+min_x)/2), floor((max_y+min_y)/2)]])
            sam_labels = np.array([True])
            mask, scores, logits = mask_predictor.predict(
                box=box,
                point_coords=sam_points,
                point_labels=sam_labels,
                multimask_output=True,
            )

            print( scores )
            print( max(scores))
            target_index = np.where(scores == max(scores))[0][0]
            body_mask_image = None
            for ii, m in enumerate(mask):
                if ii == target_index:
                    body_mask_image = m.astype(np.uint8) * 255
                    body_mask_image = cv2.cvtColor(body_mask_image, cv2.COLOR_GRAY2RGBA)
                    base64_result_mask = encode_img_to_base64(body_mask_image)
                    cv2.imwrite('./cache/{}-{}.{}'.format(file_name_arr[0], str(ii), file_name_arr[-1]), body_mask_image)
                    payload['results'].append(base64_result_mask)
            
            if segment_input.show_result and not segment_input.show_mask_tattoo:
                print( 'Debug segment input show result')
                tattoo_image = cv2.imread(tat_local_file_path, cv2.IMREAD_UNCHANGED)
                body_image = cv2.imread(local_file_path, cv2.IMREAD_UNCHANGED)
                print( tattoo_image.shape )
                print( body_mask_image.shape )
                final_img, masked_tattoo_img, gray_masked_tattoo_img, multi_mode_final_img, multi_mode_gray_final_img = apply_mask(body_mask_image, tattoo_image, body_image)

                base64_final_img = encode_img_to_base64(final_img)
                base64_masked_tattoo_img = encode_img_to_base64(masked_tattoo_img)
                base64_gray_masked_tattoo_img = encode_img_to_base64(gray_masked_tattoo_img)
                base64_multi_mode_final_img = encode_img_to_base64(multi_mode_final_img)
                base64_multi_mode_gray_final_img = encode_img_to_base64(multi_mode_gray_final_img)

                payload["masked_tattoo_img"] = base64_masked_tattoo_img.decode('utf-8')
                payload["gray_masked_tattoo_img"] = base64_gray_masked_tattoo_img.decode('utf-8')
                payload["final_img"] = base64_final_img.decode('utf-8')
                payload["multi_mode_final_img"] = base64_multi_mode_final_img.decode('utf-8')
                payload["multi_mode_gray_final_img"] = base64_multi_mode_gray_final_img.decode('utf-8')
            elif segment_input.show_mask_tattoo:
                print( 'Debug segment input show result')
                tattoo_image = cv2.imread(tat_local_file_path, cv2.IMREAD_UNCHANGED)
                # body_image = cv2.imread(local_file_path, cv2.IMREAD_UNCHANGED)
                print( tattoo_image.shape )
                print( body_mask_image.shape )
                
                masked_tattoo_img = pure_apply_mask(body_mask_image, tattoo_image)

                base64_masked_tattoo_img = encode_img_to_base64(masked_tattoo_img)
                payload["masked_tattoo_img"] = base64_masked_tattoo_img.decode('utf-8')

        elif segment_input.bounding_box and len(segment_input.bounding_box) == 4:
            mask_predictor.set_image(image_rgb)
            box = np.array(segment_input.bounding_box)
            mask, scores, logits = mask_predictor.predict(
                box=box,
                multimask_output=True
            )
            print( scores )
            for ii, m in enumerate(mask):
                m_uint8 = m.astype(np.uint8) * 255
                m_uint8 = cv2.cvtColor(m_uint8, cv2.COLOR_GRAY2RGBA)
                base64_result_mask = encode_img_to_base64(m_uint8)
                cv2.imwrite('./cache/{}-{}.{}'.format(file_name_arr[0], str(ii), file_name_arr[-1]), m_uint8)
                payload['results'].append(base64_result_mask)
        else:
            result = mask_generator.generate(image_rgb)
            print("--- Model finished running %s seconds ---" % (time.time() - start_time))
            for ii, res in enumerate(result):
                # print( type(res['segmentation']) )
                # print( res['segmentation'] )
                m_uint8 = res['segmentation'].astype(np.uint8) * 255
                m_uint8 = cv2.cvtColor(m_uint8, cv2.COLOR_GRAY2RGBA)
                if not ii:
                    print(m_uint8.shape)
                
                base64_result_mask = encode_img_to_base64(m_uint8)
                # cv2.imwrite('./cache/{}-{}.{}'.format(file_name_arr[0], str(ii), file_name_arr[-1]), m_uint8)
                payload['results'].append(base64_result_mask)

            # detections = sv.Detections.from_sam(result)
            # print('Debug detections')
            # print( detections )
            # annotated_image = mask_annotator.annotate(image_bgr, detections)
            # print( annotated_image.shape )
            # cv2.imwrite('./cache/{}-annotated-image{}.{}'.format(file_name_arr[0], str(0), file_name_arr[-1]), annotated_image)
            print("--- Ready to respond %s seconds ---" % (time.time() - start_time))
    return payload