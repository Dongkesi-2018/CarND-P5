from P4 import p4_init
from P5 import p5_init
from utils import resize_image
import numpy as np
import cv2


f1 = 'project_video.mp4'
f2 = 'test_video.mp4'
input_file = f1

lane, lane_verify, p4_pipeline = p4_init("./CarND-Advanced-Lane-Lines/", input_file)
tracker = p5_init("./CarND-Vehicle-Detection/")
# Change Input File Here

w_name = input_file
cv2.namedWindow(w_name, cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture("./CarND-Vehicle-Detection/" + input_file)
def progress_bar_cb(x):
    cap.set(cv2.CAP_PROP_POS_FRAMES, x)

cv2.createTrackbar('Frame', w_name, 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), progress_bar_cb)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
prefix = input_file.split('.')[0]
out_debug = cv2.VideoWriter(prefix + '_debug.avi', fourcc, 20.0, (1536, 864))
out_project = cv2.VideoWriter(prefix + '_output.avi', fourcc, 20.0, (1280, 720))
delay = 1
pause = False
frame_idx = 0
ret = False
while (cap.isOpened()):
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        pause = not pause

    if not pause:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('Frame', w_name, frame_idx)
        ret, frame = cap.read()

    if not ret:
        break
    print("--------current frame:", frame_idx)

    final_img, project, undist_img = p4_pipeline(frame, lane, lane_verify)
    layer1_input_boxes_img, layer1_output_img, \
    layer2_input_boxes_img, layer2_output_img, \
    layer1_heatmap, layer2_heatmap, \
    slide_boxes_img, _, parameter_img\
        = tracker.pipeline(np.copy(undist_img))
    vehicles_img = tracker.draw_vehicles_img(project)

    scale = 0.4
    resize_shape = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    layer1_input_boxes_img_resized = resize_image(layer1_input_boxes_img, resize_shape, "layer1_input_boxes_img")
    layer1_output_img_resized = resize_image(layer1_output_img, resize_shape, "layer1_output_img")
    layer2_input_boxes_img_resized = resize_image(layer2_input_boxes_img, resize_shape, 'layer2_input_boxes_img')
    layer2_output_img_resized = resize_image(layer2_output_img, resize_shape, 'layer2_output_img')
    layer1_heatmap_resized = resize_image(layer1_heatmap, resize_shape, 'layer1_heatmap')
    layer2_heatmap_resized = resize_image(layer2_heatmap, resize_shape, 'layer2_heatmap')
    slide_boxes_img_resized = resize_image(slide_boxes_img, resize_shape, 'slide_window')
    vehicles_img_resized = resize_image(np.copy(vehicles_img), resize_shape, 'vehicles_img')
    parameter_img_resized = resize_image(parameter_img, resize_shape, 'parameters')
    project_resized = resize_image(project, resize_shape, 'project')

    img_h1 = np.hstack((layer1_input_boxes_img_resized, layer1_heatmap_resized, layer1_output_img_resized, ))
    img_h2 = np.hstack((layer2_input_boxes_img_resized, layer2_heatmap_resized, layer2_output_img_resized, ))
    img_h3 = np.hstack((slide_boxes_img_resized, parameter_img_resized, vehicles_img_resized,))
    img = np.vstack((img_h1, img_h2, img_h3))
    # print(img.shape, layer2_output_img.shape)
    cv2.imshow(w_name, img)

    # cv2.imshow('1', vehicles_img)
    out_debug.write(img)
    out_project.write(vehicles_img)

cap.release()
cv2.destroyAllWindows()