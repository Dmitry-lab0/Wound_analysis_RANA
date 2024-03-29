from flask import Flask, render_template, request, jsonify, send_file,redirect
from flask_cors import CORS
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from typing import Any, Dict, List
from arg_parse import parser
from utils import mkdir_or_exist
from collections import deque
import threading
import queue
import cv2
import numpy as np
import io, os
import time
import base64
import argparse
import torch
import torchvision
import calculate_area
import calculate_depth
import gc
import open3d as o3d


from depth_estimation.midas.model_loader import default_models, load_model
from depth_estimation.tf.run_onnx import load_model_onnx
import onnx

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

class Mode:
    def __init__(self) -> None:
        self.IAMGE = 1
        self.MASKS = 2
        self.CLEAR = 3
        self.P_POINT = 4
        self.N_POINT = 5
        self.BOXES = 6
        self.INFERENCE = 7
        self.UNDO = 8
        self.COLOR_MASKS = 9
        self.DEPTH_MASKS = 10 
        self.MODEL_3D = 12
        self.CALCULATE_AREA_PERIMETER_VOLUME = 0 
        

MODE = Mode()

class SamAutoMaskGen:
    def __init__(self, model, args) -> None:
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
        self.amg_kwargs = self.get_amg_kwargs(args)
        self.generator = SamAutomaticMaskGenerator(model, output_mode=output_mode, **self.amg_kwargs)

    def get_amg_kwargs(self, args):
        amg_kwargs = {
            "points_per_side": args.points_per_side,
            "points_per_batch": args.points_per_batch,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "stability_score_offset": args.stability_score_offset,
            "box_nms_thresh": args.box_nms_thresh,
            "crop_n_layers": args.crop_n_layers,
            "crop_nms_thresh": args.crop_nms_thresh,
            "crop_overlap_ratio": args.crop_overlap_ratio,
            "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
            "min_mask_region_area": args.min_mask_region_area,
        }
        amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
        return amg_kwargs

    def generate(self, image) -> np.ndarray:
        masks = self.generator.generate(image)
        np_masks = []
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            np_masks.append(mask)

        return np.array(np_masks, dtype=bool)

class SAM_Web_App:
    def __init__(self, args):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.args = args
         
        #clear cache
        gc.collect()
        torch.cuda.empty_cache()
            
        # load model SAM
        print("Loading model SAM...\n", end="")
        self.device_1 = args.device

        
        print(f"using {self.device_1}...\n", end="")
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        sam.to(device=self.device_1)

        self.predictor = SamPredictor(sam)
        self.autoPredictor = SamAutoMaskGen(sam, args)
        
        # load model for depth estimation
        self.model_for_depth_path = "depth_estimation/weights/dpt_beit_large_512_dymanic_axes_square.onnx"
        self.model_for_depth_type = "dpt_beit_large_512"
        print("Loading model for depth estimation...\n", end="")
        #self.model_for_depth_path = "depth_estimation/weights/dpt_swin2_large_384.pt"
        #self.model_for_depth_type = "dpt_swin2_large_384"
        
        self.optimize = True
        self.height = None
        self.square = True
        self.device_2 = 'cpu'

       
        
        self.model_for_depth, self.transform_for_depth, self.net_w, self.net_h = load_model_onnx(self.model_for_depth_path, self.model_for_depth_type, self.square)
        


        #self.model_for_depth, self.transform_for_depth, self.net_w, self.net_h = load_model(self.device_2, self.model_for_depth_path, self.model_for_depth_type, square=self.square)

        print("Done")




        # Store the image globally on the server
        self.origin_image = None
        self.processed_img = None
        self.masked_img = None
        self.image_with_square = None
        self.colorMasks = None
        self.depthMasks = None
        self.imgSize = None
        self.imgIsSet = False           # To run self.predictor.set_image() or not

        self.area = 0
        self.perimeter = 0
        self.volume = 0 

        self.mode = "p_point"           # p_point / n_point / box
        self.curr_view = "image"
        self.queue = deque(maxlen=1000)  # For undo list
        self.prev_inputs = deque(maxlen=500)

        self.points = []
        self.points_label = []
        self.boxes = []
        self.masks = []

        # Set the default save path to the Downloads folder
        home_dir = os.path.expanduser("~")
        self.save_path = os.path.join(home_dir, "Downloads")

        self.app.route('/', methods=['GET'])(self.home)
        self.app.route('/get_area_perimeter_volume', methods=['GET'])(self.get_area_perimeter_volume)
        self.app.route('/clear_wound_properties', methods=['GET'])(self.clear_wound_properties)
        self.app.route('/set_height', methods=['POST'])(self.set_height)
        self.app.route('/upload_image', methods=['POST'])(self.upload_image)
        self.app.route('/button_click', methods=['POST'])(self.button_click)
        self.app.route('/point_click', methods=['POST'])(self.handle_mouse_click)
        self.app.route('/box_receive', methods=['POST'])(self.box_receive)
        self.app.route('/set_save_path', methods=['POST'])(self.set_save_path)
        self.app.route('/save_image', methods=['POST'])(self.save_image)
        self.app.route('/send_stroke_data', methods=['POST'])(self.handle_stroke_data)

        
        
    def home(self): 
         return render_template('index.html', default_save_path=self.save_path)
    
    def set_height(self):
        height = request.form.get('height') 
        print(height)
        if height is not None:
            height = float(height)  # in mm
            if height > 0:
                self.height = height
                print(f"Set height: {self.height}")
                return jsonify({"status": "success", "message": "Height set successfully", 'height':height})
      
        return jsonify({"status": "error", "message": "Invalid height"}), 400
            

    
    def set_save_path(self):
        self.save_path = request.form.get("save_path")

        # Perform your server-side checks on the save_path here
        # e.g., check if the path exists, if it is writable, etc.
        if os.path.isdir(self.save_path):
            print(f"Set save path to: {self.save_path}")
            return jsonify({"status": "success", "message": "Save path set successfully"})
        else:
            return jsonify({"status": "error", "message": "Invalid save path"}), 400
        
    def save_image(self):
        # Save the colorMasks
        filename = request.form.get("filename")
        if filename == "":
            return jsonify({"status": "error", "message": "No image to save"}), 400
        print(f"Saving: {filename} ...", end="")
        dirname = os.path.join(self.save_path, filename)
        mkdir_or_exist(dirname)
        # Get the number of existing files in the save_folder
        num_files = len([f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))])
        # Create a unique file name based on the number of existing files
        savename = f"{num_files}.png"
        save_path = os.path.join(dirname, savename)
        try:
            encoded_img = cv2.imencode(".png", self.colorMasks)[1]
            encoded_img.tofile(save_path)
            print("Done!")
            return jsonify({"status": "success", "message": f"Image saved to {save_path}"})
        except:
            return jsonify({"status": "error", "message": "Imencode error"}), 400

    def upload_image(self):
        if 'image' not in request.files:
            return jsonify({'error': 'No image in the request'}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Store the image globally
        self.origin_image = image
        self.processed_img = image
        self.masked_img = np.zeros_like(image)
        self.colorMasks = np.zeros_like(image)
        self.depthMasks = np.zeros_like(image)
        self.imgSize = image.shape

        # Create image imbedding
        # self.predictor.set_image(image, image_format="RGB")   # Move to first inference

        # Reset inputs and masks and image ebedding
        self.imgIsSet = False
        self.reset_inputs()
        self.reset_masks()
        self.queue.clear()
        self.prev_inputs.clear()
        torch.cuda.empty_cache()

        self.clear_wound_properties()
        return "Uploaded image, successfully initialized"

    def button_click(self):
        if self.processed_img is None:
            return jsonify({'error': 'No image available for processing'}), 400

        data = request.get_json()
        button_id = data['button_id']
        print(f"Button {button_id} clicked")

        # Info
        info = {
            'event': 'button_click',
            'data': button_id
        }

        # Process and return the image
        return self.process_image(self.processed_img, info)

    def handle_mouse_click(self):
        if self.processed_img is None:
            return jsonify({'error': 'No image available for processing'}), 400

        data = request.get_json()
        x = data['x']
        y = data['y']
        print(f'Point clicked at: {x}, {y}')
        self.points.append(np.array([x, y], dtype=np.float32))
        self.points_label.append(1 if self.mode == 'p_point' else 0)

        # Add command to queue list
        self.queue.append("point")

        # Process and return the image
        return f"Click at image pos {x}, {y}"
    def get_area_perimeter_volume(self):
        if self.processed_img is None:
            return jsonify({'error': 'No image available for processing'}), 400
        area_mm = 0
        perimeter_mm = 0
        volume = 0
        height = 0
        if self.height is not None and self.height>0:

            print('CALCULATE_AREA_PERIMETER_VOLUME')   
            processed_image, area_mm, perim_mm, real_area_per_px = calculate_area.get_area_and_perim_in_mm(self.origin_image, self.masks)
            self.processed_img = processed_image
            self.area = area_mm
            self.perimeter = perim_mm
            area_mm =  "{:.2f}".format(self.area)
            perimeter_mm = "{:.2f}".format(self.perimeter)

            processed_prediction_with_borders, prediction_for_volume_calc_with_borders, _, _, _ = calculate_depth.get_depth_image(self.device_2, self.model_for_depth, self.model_for_depth_type, 
                self.origin_image , self.colorMasks, (self.net_w, self.net_h), self.transform_for_depth, self.optimize, grayscale=False)

            _, processed_prediction_with_borders = self.updateDepthMaskImg(processed_prediction_with_borders, self.masks)
            _, prediction_for_volume_calc_with_borders = self.updateDepthMaskImg(prediction_for_volume_calc_with_borders, self.masks)

            prediction_for_volume_calc_with_borders = cv2.cvtColor(prediction_for_volume_calc_with_borders, cv2.COLOR_RGB2GRAY) 
            max_val = prediction_for_volume_calc_with_borders.max().astype(float)
            prediction_for_volume_calc_with_borders = prediction_for_volume_calc_with_borders / max_val
            height_mm = self.height * prediction_for_volume_calc_with_borders
            volume_mm = np.sum(real_area_per_px*height_mm)
            self.depthMasks = processed_prediction_with_borders
            self.volume = volume_mm
            volume = "{:.2f}".format(self.volume)
            height = "{:.2f}".format(self.height)

            print(f"Area {area_mm}")
            print(f"Perimeter {perim_mm}")
            print(f"Volume: {volume_mm}")
            print(f"Height: {height}")
            self.reset_inputs()
            self.queue.append("calculate_area_perimeter_volume")
        return jsonify({'area': area_mm, 'perimeter': perimeter_mm, 'volume': volume, 'height': height})

    def clear_wound_properties(self):
        self.area =  0 
        self.perimeter = 0 
        self.volume = 0
        self.height = 0
        return jsonify({'area': self.area, 'perimeter': self.perimeter, 'volume': self.volume, 'height': self.height})


    
    def handle_stroke_data(self):
        data = request.get_json()
        stroke_data = data['stroke_data']

        print("Received stroke data")

        if len(stroke_data) == 0:
            pass
        else:
            # Process the stroke data here
            stroke_img = np.zeros_like(self.origin_image)
            print(f"stroke data len: {len(stroke_data)}")

            latestData = stroke_data[len(stroke_data) - 1]
            strokes, size = latestData['Stroke'], latestData['Size']
            BGRcolor = (latestData['Color']['b'], latestData['Color']['g'], latestData['Color']['r'])
            Rpos, Bpos = 2, 0
            stroke_data_cv2 = []
            for stroke in strokes:
                stroke_data_cv2.append((int(stroke['x']), int(stroke['y'])))
            for i in range(len(strokes) - 1):
                cv2.line(stroke_img, stroke_data_cv2[i], stroke_data_cv2[i + 1], BGRcolor, size)

            if BGRcolor[0] == 255:
                mask = np.squeeze(stroke_img[:, :, Bpos] == 0)
                opt = "negative"
            else: # np.where(BGRcolor == 255)[0] == Rpos
                mask = np.squeeze(stroke_img[:, :, Rpos] > 0)
                opt = "positive"

            self.masks.append({
                "mask": mask,
                "opt": opt
            })

        self.get_colored_masks_image()
        self.processed_img, maskedImage = self.updateMaskImg(self.origin_image, self.masks)
        self.masked_img = maskedImage
        self.queue.append("brush")

        if self.curr_view == "masks":
            print("view masks")
            processed_image = self.masked_img
        elif self.curr_view == "colorMasks":
            print("view color")
            processed_image = self.colorMasks
        elif self.curr_view == "depthMasks":
            print("view depth")
            processed_image = self.depthMasks
        else:   # self.curr_view == "image":
            print("view image")
            processed_image = self.processed_img

        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': img_base64})
    
    def box_receive(self):
        if self.processed_img is None:
            return jsonify({'error': 'No image available for processing'}), 400

        data = request.get_json()
        self.boxes.append(np.array([
            data['x1'], data['y1'],
            data['x2'], data['y2']
        ], dtype=np.float32))

        # Add command to queue list
        self.queue.append("box")

        return "server received boxes"

    def process_image(self, image, info):
        processed_image = image

        if info['event'] == 'button_click':
            id = info['data']
            if (id == MODE.IAMGE):
                self.curr_view = "image"
                processed_image = self.processed_img
            elif (id == MODE.MASKS):
                self.curr_view = "masks"
                processed_image = self.masked_img
            elif (id == MODE.COLOR_MASKS):
                self.curr_view = "colorMasks"
                processed_image = self.colorMasks
            elif (id == MODE.DEPTH_MASKS):
                self.curr_view = "depthMasks"
                processed_image = self.depthMasks
            elif (id == MODE.CLEAR):
                processed_image = self.origin_image
                self.processed_img = self.origin_image
                self.reset_inputs()
                self.reset_masks()  
                self.clear_wound_properties()
                self.queue.clear()
                self.prev_inputs.clear()
            elif (id == MODE.P_POINT):
                self.mode = "p_point"
            elif (id == MODE.N_POINT):
                self.mode = "n_point"
            elif (id == MODE.BOXES):
                self.mode = "box"
            elif (id == MODE.INFERENCE):
                print("INFERENCE")
                # self.reset_masks()
                
                #clear cache
                gc.collect()
                torch.cuda.empty_cache()
               
                points = np.array(self.points)
                labels = np.array(self.points_label)
                boxes = np.array(self.boxes)
                print(f"Points shape {points.shape}")
                print(f"Labels shape {labels.shape}")
                print(f"Boxes shape {boxes.shape}")
                if len(boxes) != 0 or len(labels) != 0 or len(boxes) != 0:
                    prev_masks_len = len(self.masks)
                    processed_image, self.masked_img = self.inference(self.origin_image, points, labels, boxes)
                    curr_masks_len = len(self.masks)
                    #print(type(self.masks[0]['mask']))
                    #print((self.masks[0]['mask'].shape))
                    #self.post_processing(self.masks)

                    self.get_colored_masks_image()
                    self.processed_img = processed_image
                    self.prev_inputs.append({
                        "points": self.points,
                        "labels": self.points_label,
                        "boxes": self.boxes
                    })
                    self.reset_inputs()
                    self.queue.append(f"inference-{curr_masks_len - prev_masks_len}")
   
            elif(id == MODE.CALCULATE_AREA_PERIMETER_VOLUME):
                processed_img = self.processed_img


            elif (id == MODE.MODEL_3D):
                prediction, _, image, coords, img_size = calculate_depth.get_depth_image(self.device_2, self.model_for_depth, self.model_for_depth_type, 
                    self.origin_image , self.colorMasks, (self.net_w, self.net_h), self.transform_for_depth, self.optimize, crop_and_process=False
                )
                 #prediction *= 1000
                x = coords[0]
                y = coords[1]
                width = img_size[0]
                height = img_size[1]
                #print("x", x, "y", y, "h", h, "w", w, "black_image shape", black_image.shape, "prediction shape", prediction.shape)
                prediction = prediction[y:y+height, x:x+width]
                cropped_image = image[y:y+height, x:x+width]
                cropped_image = cv2.flip(cropped_image, 1)
                prediction = cv2.flip(prediction, 1)
                depth_image = (prediction * 255 / np.max(prediction)).astype('uint8')
                image = (np.array(cropped_image)*255).astype('uint8')

                # create rgbd image
                depth_o3d = o3d.geometry.Image(depth_image)
                image_o3d = o3d.geometry.Image(image)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

                # camera settings
                camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
                camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

                # create point cloud
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
                
                o3d.visualization.draw_geometries([pcd])

            
            elif (id == MODE.UNDO):
                if len(self.queue) != 0:
                    command = self.queue.pop()
                    command = command.split('-')
                else:
                    command = None
                print(f"Undo {command}")

                if command is None:
                    pass
                elif command[0] == "point":
                    self.points.pop()
                    self.points_label.pop()
                elif command[0] == "box":
                    self.boxes.pop()
                elif command[0] == "inference":
                    # Calculate masks and image again
                    val = command[1]
                    self.masks = self.masks[:(len(self.masks) - int(val))]
                    self.processed_img, self.masked_img = self.updateMaskImg(self.origin_image, self.masks)
                    self.get_colored_masks_image()

                    # Load prev inputs
                    prev_inputs = self.prev_inputs.pop()
                    self.points = prev_inputs["points"]
                    self.points_label = prev_inputs["labels"]
                    self.boxes = prev_inputs["boxes"]
                elif command[0] == "brush":
                    self.masks.pop()
                    self.processed_img, self.masked_img = self.updateMaskImg(self.origin_image, self.masks)
                    self.get_colored_masks_image()
                elif command[0] == 'calculate_area':
                    self.processed_img, self.masked_img = self.updateMaskImg(self.origin_image, self.masks)
                
                if self.curr_view == "masks":
                    print("view masks")
                    processed_image = self.masked_img
                elif self.curr_view == "colorMasks":
                    print("view color")
                    processed_image = self.colorMasks
                elif self.curr_view == "depthMasks":
                    print("view color")
                    processed_image = self.depthMasks
                else:   # self.curr_view == "image":
                    print("view image")
                    processed_image = self.processed_img
            

        _, buffer = cv2.imencode('.jpg', processed_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': img_base64})
    
    
    def post_processing(self, masks):
        for mask in masks:
        #print(mask)
            mask_u8 = mask['mask'].astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) 
            label = np.uint8(np.zeros((mask_u8.shape[0], mask_u8.shape[1])))
            if len(contours)!=1:
                for num, cnt in enumerate(contours):
                    contour_area = cv2.contourArea(cnt)
                    if contour_area > 130: 
                        if hierarchy[0, num][3] != -1:
                            label = cv2.drawContours(label, [cnt], -1, (0,0,0), -1)

                        else:
                            label = cv2.drawContours(label, [cnt], -1, (255,255,255), -1)
            else:
                label = cv2.drawContours(label, contours, -1, (255,255,255), -1)
                            
            mask['mask'] = label.astype(np.bool_)
    
            
           
    
    def inference(self, image, points, labels, boxes) -> np.ndarray:
        
        #clear cache
        gc.collect()
        torch.cuda.empty_cache()

        
        points_len, lables_len, boxes_len = len(points), len(labels), len(boxes)
        if (len(points) == len(labels) == 0):
            points = labels = None
        if (len(boxes) == 0):
            boxes = None

        # Image is set ?
        if self.imgIsSet == False:
            self.predictor.set_image(image, image_format="RGB")
            self.imgIsSet = True
            print("Image set!")

        #Auto 
        if (points_len == boxes_len == 0):
            masks = self.autoPredictor.generate(image)
            for mask in masks:
                self.masks.append({
                    "mask": mask,
                    "opt": "positive"
                })

        # One Object
        elif ((boxes_len == 1) or (points_len > 0 and boxes_len <= 1)):
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=boxes,
                multimask_output=True,
            )
            max_idx = np.argmax(scores)
            self.masks.append({
                "mask": masks[max_idx],
                "opt": "positive"
            })

        # Multiple Object
        elif (boxes_len > 1):
            boxes = torch.tensor(boxes, device=self.predictor.device)
            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
            masks, scores, logits = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            masks = masks.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            max_idxs = np.argmax(scores, axis=1)
            print(f"output mask shape: {masks.shape}")  # (batch_size) x (num_predicted_masks_per_input) x H x W
            for i in range(masks.shape[0]):
                self.masks.append({
                    "mask": masks[i][max_idxs[i]],
                    "opt": "positive"
                })
        
        self.post_processing(self.masks)
        
        # Update masks image to show
        overlayImage, maskedImage = self.updateMaskImg(self.origin_image, self.masks)
        # overlayImage, maskedImage = self.updateMaskImg(overlayImage, maskedImage, [self.brushMask])
        return overlayImage, maskedImage

    def updateMaskImg(self, image, masks):

        if (len(masks) == 0 or masks[0] is None):
            print(masks)
            return image, np.zeros_like(image)
        
        union_mask = np.zeros_like(image)[:, :, 0]
        np.random.seed(0)
        for i in range(len(masks)):
            if masks[i]['opt'] == "negative":
                image = self.clearMaskWithOriginImg(self.origin_image, image, masks[i]['mask'])
                union_mask = np.bitwise_and(union_mask, masks[i]['mask'])
            else:
                colored = True
                image = self.overlay_mask(image, masks[i]['mask'], 0.9, colored)
                union_mask = np.bitwise_or(union_mask, masks[i]['mask'])
        
        # Cut out objects using union mask
        masked_image = self.origin_image * union_mask[:, :, np.newaxis]
        
        return image, masked_image
    
    def updateDepthMaskImg(self, image, masks):

        # if (len(masks) == 0 or masks[0] is None):
        #     return image, np.zeros_like(image)
        
        union_mask = np.zeros_like(image)[:, :, 0]
        np.random.seed(0)
        for i in range(len(masks)):
            if masks[i]['opt'] == "negative":
                image = self.clearMaskWithOriginImg(self.origin_image, image, masks[i]['mask'])
                union_mask = np.bitwise_and(union_mask, masks[i]['mask'])
            
            else:
                union_mask = np.bitwise_or(union_mask, masks[i]['mask'])
        
        # Cut out objects using union mask
        masked_image = image * union_mask[:, :, np.newaxis]

        return image, masked_image

    # Function to overlay a mask on an image
    def overlay_mask(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        alpha: float, 
        colored: bool = False,
    ) -> np.ndarray:
        """ Draw mask on origin image

        parameters:
        image:  Origin image
        mask:   Mask that have same size as image
        color:  Mask's color in BGR
        alpha:  Transparent ratio from 0.0-1.0

        return:
        blended: masked image
        """
        # Blend the image and the mask using the alpha value
        if colored:
            color = np.array([0.5, 0.25, 0.25])
        else:
            color = np.array([1.0, 1.0, 1.0])    # BGR
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask *= 255 * alpha
        mask = mask.astype(dtype=np.uint8)
        blended = cv2.add(image, mask)
        
        return blended
    
    def get_colored_masks_image(self):
        masks = self.masks
        darkImg = np.zeros_like(self.origin_image)
        image = darkImg.copy()

        np.random.seed(0)
        if (len(masks) == 0):
            self.colorMasks = image
            return image
        for mask in masks:
            if mask['opt'] == "negative":
                image = self.clearMaskWithOriginImg(darkImg, image, mask['mask'])
            else:
                colored = False
                image = self.overlay_mask(image, mask['mask'], 1, colored)

        self.colorMasks = image
        return image

    # def process_image_for_depth(self, image):
    #     masked_img_white_background = np.ones_like(image)
    #     #if np.any(self.masked_img is None and self.colorMasks is None):
    #     inverse_colorMasks = np.logical_not(self.colorMasks).astype(int)*255
    #     masked_img_white_bg = self.masked_img + inverse_colorMasks
    #     return masked_img_white_bg
         
    def clearMaskWithOriginImg(self, originImage, image, mask):
        originImgPart = originImage * np.invert(mask)[:, :, np.newaxis]
        image = image * mask[:, :, np.newaxis]
        image = cv2.add(image, originImgPart)
        return image
    
    def reset_inputs(self):
        self.points = []
        self.points_label = []
        self.boxes = []

    def reset_masks(self):
        self.masks = []
        self.masked_img = np.zeros_like(self.origin_image)
        self.colorMasks = np.zeros_like(self.origin_image)
        self.depthMasks = np.zeros_like(self.origin_image)
        
        
    def run(self, debug=True):
        self.app.run(debug=debug, port=8989)


if __name__ == '__main__':
    
    args = parser().parse_args()
    app = SAM_Web_App(args)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    app.run(debug=True)