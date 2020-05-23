# import the necessary packages
from face_blur.face_blurring import anonymize_face_pixelate
from face_blur.face_blurring import anonymize_face_simple
import numpy as np
import os
import argparse
import cv2

# GPU support
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to import image.")
ap.add_argument("-m", "--method", type=str, default="simple", choices=["simple", "pixelated"],
                help='face bulrring/anonymizing method.')
ap.add_argument('-f', "--face", required=True,
                help="path to face detector model directory.")
ap.add_argument('-b', '--blocks', type=int, default=20,
                help='# of blocks for the pixelated blurring method.')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
                help='minimum probability to filter weak detections.')
ap.add_argument('-s', '--save', default='save.png',
                help='path to save file.')

args = vars(ap.parse_args())


# load our mserialized face detector model for disk
print('[INFO] loading face detector model...')
# transfer learning
prototxtPath = os.path.sep.join([args['face'], 'deploy.prototxt'])
weightPath = os.path.sep.join([args['face'], 'res10_300x300_ssd_iter_140000.caffemodel'])
net = cv2.dnn.readNet(prototxtPath, weightPath)

# load the input image frim disk, clone it, and grab the image spatial dimensions
image = cv2.imread(args['image'])
# image = np.array(image)
orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))

# parse the blob through the network and obtain the face detections
print('[INFO] computing face detections...')
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) assocaiate with the detection
    confidence = detections[0,0,i,2]

    # filter out weak detections by ensuring the confience is greadter than the minimun confidence
    if confidence > args['confidence']:
        # compare the (x,y)-coordinates of teh bounding box for the object
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (x_start, y_start, x_end, y_end) = box.astype('int')

        # extract the face ROI (region of interest)
        face = image[y_start:y_end, x_start:x_end]

        # check to see if we are applying the 'sample' face bulrring method
        if args['method'] == 'simple':
            face = anonymize_face_simple(face, factor=3.0)
            

        # otherwise, we must be applying the 'pixelated' ace anonymization method
        else:
            face = anonymize_face_pixelate(face, blocks=args['blocks'])

        # store the blurred face in the ouput image
        image[y_start:y_end, x_start:x_end] = face

# display the original image and the output image with the blurred face side by side
output = np.hstack([orig, image])

# save the image 
print('[INFO] svaing the image...')
cv2.imwrite(args['save'], image)

# show the original-image and blurred-image
cv2.imshow("Output", output)
cv2.waitKey(0)

