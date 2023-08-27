import torch
import cv2
import numpy as np
import gc
from depth_estimation.tf.run_onnx import process_onnx
from depth_estimation.run import process


def process_prediction(prediction, grayscale, bits=1):
    if not grayscale:
        bits = 1

    if not np.isfinite(prediction).all():
        prediction=np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = prediction.min()
    depth_max = prediction.max()
    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (prediction - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(prediction.shape, dtype=prediction.dtype)
    
    out_for_volume_calc = cv2.merge([out,out,out])
    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)
        print('INFERNO')
    else:
        #out = cv2.cvtColor(np.uint8(out), cv2.COLOR_BGR2RGB)
        #gray = cv2.cvtColor(np.uint8(out), cv2.COLOR_BGR2GRAY)
        out = cv2.merge([out,out,out])
        print('BONE')
    
    if bits == 1:
        return out.astype("uint8"), out_for_volume_calc
    elif bits == 2:
        return out.astype("uint16"), out_for_volume_calc

def process_image(image_bgr, mask, cropped = True, input_size_for_model = 512):

    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) 
    if mask is not None: 
        gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        try: hierarchy = hierarchy[0]
        except: hierarchy = []

        height, width = gray.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        # computes the bounding box for the contour
        for contour, hier in zip(contours, hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
        
        if max_x - min_x > 0 and max_y - min_y > 0 and cropped == True:
            img = img[min_y:max_y, min_x:max_x]

        x = min_x
        y = min_y
        h = max_y - min_y
        w = max_x - min_x
        coords = x,y
        img_size = w,h
    
        img = img / 255.0

    return img, coords, img_size

def add_borders(coords, crop_image_size, image, crop_image):
        x = coords[0]
        y = coords[1]
        w = crop_image_size[0]
        h = crop_image_size[1]
        crop_image_with_borders = np.zeros_like(image)
        crop_image_with_borders[y:y+h, x:x+w] = crop_image
        return crop_image_with_borders


def get_depth_image(device,  model, model_type, image_bgr, mask ,image_bgr_size, transform, optimize,  crop_and_process=True, bits=2, grayscale=False):    

    gc.collect()
    torch.cuda.empty_cache()
    prediction_for_volume_calc_with_borders = None
    
    processed_image, coords, crop_image_size = process_image(image_bgr, mask, crop_and_process)

    transformed_image = transform({"image": processed_image})["image"]
    with torch.no_grad():
        prediction = process_onnx( model, model_type, transformed_image, image_bgr_size, processed_image.shape[1::-1])
        #prediction = process(device, model,model_type, image, image_size, original_image_rgb.shape[1::-1], optimize, False)
        #  process(device, model, model_type, image, input_size, target_size, optimize, use_camera)

    # model.eval()
    # img_input = np.zeros(image.shape, np.float32)
    # sample = torch.from_numpy(image).to(device).unsqueeze(0)
    
    # torch.onnx.export(model, sample, 'dpt_beit_large_512.onnx', input_names = ['input'],   # the model's input names
    #               output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3 : 'width'},    # variable length axes
    #                             'output' : {0 : 'batch_size', 2 : 'height', 3 : 'width'}})
    if crop_and_process:
        processed_prediction, prediction_for_volume_calc = process_prediction(prediction, grayscale)
        prediction_for_volume_calc = 1 - prediction_for_volume_calc
        prediction = add_borders(coords, crop_image_size, image_bgr, processed_prediction)
        prediction_for_volume_calc_with_borders = add_borders(coords, crop_image_size, image_bgr, prediction_for_volume_calc)

    
    return prediction, prediction_for_volume_calc_with_borders, processed_image, coords, crop_image_size
    






