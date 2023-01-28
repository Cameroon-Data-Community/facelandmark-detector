''' Pytorch script for model inferecing'''
from __future__ import print_function, division
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from math import *
from imutils import face_utils
import torchvision.transforms.functional as TF
import time
plt.ion()
import dlib
from network import *
import numpy as np
from skimage import io, color

def load_model(model_path):
  model=XceptionNet()
  model.load_state_dict(torch.load(model_path, map_location='cpu'))
  #model=torch.load(model_path, map_location='cpu')
  model.eval()
  return model

model=load_model('path to model')

def transform_img(image):
  image=TF.to_pil_image(image)
  image = TF.resize(image, (224, 224))
  image = TF.to_tensor(image)
  image = (image - image.min())/(image.max() - image.min())
  image = (2 * image) - 1
  return image.unsqueeze(0)


def landmarks_draw(image,img_landmarks):
  image=image.copy()
  for landmarks,(left,top,height, width) in img_landmarks:
    landmarks=landmarks.view(-1,2)
    landmarks=(landmarks+0.5)
    landmarks=landmarks.numpy()

    for i, (x,y) in enumerate(landmarks, 1):
      try:
        cv2.circle(image, (int((x * width) + left), int((y * height) + top)), 2, [40, 117, 255], -1)
      except:
        pass
  return image



detector=dlib.get_frontal_face_detector()

@torch.no_grad()


# def inference(frame):
#   gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  
#   faces=detector(gray,1)
  

#   all_faces_outputs=[]

  
#   for (i, face) in enumerate(faces):
#     (x, y, w, h) = face_utils.rect_to_bb(face)
    
#     crop_img=gray[y:y+h, x:x+w]
#     transformed_img= transform_img(crop_img)

#     landmarks_predictions = model(transformed_img.cpu())
#     landmarks_predictions=landmarks_predictions.reshape(68,2)
    
    
   
#     landmarks_predictions=torch.nn.functional.linear(landmarks_predictions,landmarks_predictions)
    
#     landmarks_predictions=landmarks_predictions.cpu().detach().numpy()
#     face_out=[]
#     for i in range(landmarks_predictions.shape[0]):
#       landmark_pt=landmarks_predictions[i]
#       landmark_dict={
#         "x":landmark_pt[0],
#         "y":landmark_pt[1]
#       }
      
#       face_out.append(landmark_dict)
      
#     all_faces_outputs.append(face_out)
    
    

#     return all_faces_outputs
def inference(frame):
  gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  faces=detector(gray, 1)
  outputs=[]
  for (i, face) in enumerate(faces):
    (x,y,w,h)=face_utils.rect_to_bb(face)
    crop_img=gray[y:y+h, x:x+w]
    transformed_img=transform_img(crop_img)
    landmarks_predictions=model(transformed_img.cpu())
    outputs.append((landmarks_predictions.cpu(), (x,y,w,h)))
  return landmarks_draw(frame, outputs) #get rid of this


# ''' inference on video'''

# def output_video(video, name, seconds = None):
#     start_time=time.time()
#     total = int(video.fps * seconds) if seconds else int(video.fps * video.duration)
#     print('Will read', total, 'images...')
    
#     outputs = []

#     writer = cv2.VideoWriter(name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.fps, tuple(video.size))

#     for i, frame in enumerate(tqdm(video.iter_frames(), total = total), 1):    
#         if seconds:
#             if (i + 1) == total:
#                 break
                
#         output = inference(frame)
#         outputs.append(output)

        

#         writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
#     end_time=time.time()
#     print("Model inference Time taken: {:.2f} seconds".format(end_time-start_time))

    

#     writer.release()

#     return outputs


''' inference on image'''

def image_output(image, name):
  print("Analysing image")
 

  # outputs=[]



  output=inference(image)
  # outputs.append(output)

  # vec=np.empty((68,2), dtype=int)
  # for _ in range(68):
  #   x=output[0]
  #   y=output[1]

  #   print(x,y)
  # print(output)
  
  # disp_img=cv2.imwrite(name + '.jpg', output)


  #writer.release()

  
  return  print(output)

# img=cv2.imread('D:/Dev Projects/DeepStack/proto.facelandmarkdetector/data/faces/1198_0_861.jpg')
# img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# outputs=[]

# transformed_img=transform_img(img)

# landmark_predictions = model(transformed_img.cpu())
# outputs.append(landmark_predictions.cpu())

# print(outputs)



# for i in range (68):
#   x=outputs[0][i][0]
#   y=outputs[0][i][1]
  
#   print(x,y)
  




img_path='../img'#replace with  path to image for inference
img=cv2.imread(img_path)
outputs=image_output(img,'') #image output path
