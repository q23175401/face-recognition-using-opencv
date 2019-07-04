import cv2
import os
import numpy as np
import pickle

RESOLUTION_DICT = {
	'480p' : (640, 480),
	'720p' : (1280, 720),
	'1080p' : (1920, 1080),
	'4k' : (3840, 2160)
}

VIDEO_CODE_TYPE = {
	'avi' : cv2.VideoWriter_fourcc(*'XVID'),
	'mp4' : cv2.VideoWriter_fourcc(*'XVID')
}

def get_video_type(filename):
	_, ext = os.path.split(filename)
	if ext in VIDEO_CODE_TYPE:
		return VIDEO_CODE_TYPE[ext]
	return VIDEO_CODE_TYPE['avi']

def set_resolution(cap, resolution='1080p'):
	if resolution in RESOLUTION_DICT:
		(width, height) = RESOLUTION_DICT[resolution]
	else:
		(width, height) = RESOLUTION_DICT['720p']

	cap.set(3, width)
	cap.set(4, height)
	return (width, height)

def resize_frame(frame, percent=100):
	width = int(frame.shape[1] * percent / 100)
	height = int(frame.shape[0] * percent / 100)
	widthAndHeight = (width, height)
	return cv2.resize(frame, widthAndHeight, interpolation=cv2.INTER_AREA)

#video camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# video reord setup
video_name = 'first_video.avi'
cur_dir = os.path.dirname(__file__)
video_dir = os.path.join(cur_dir, 'videos', video_name)
frames_per_second = 24
video_res = '720p'

# camera setup and video writer setup
video_dim = set_resolution(cap, video_res)
video_code_type = get_video_type(video_name)
video_writer = cv2.VideoWriter(video_dir, video_code_type, frames_per_second, video_dim)

face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognize.yml')
id_label_dict = dict()
with open('label_ids.pickle', 'rb') as f:
	label_id_dict = pickle.load(f)
	id_label_dict = {v:k for k, v in label_id_dict.items()}

print(id_label_dict.items())
while True:
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	# 偵測是否正確收到frame
	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# 用haar feature 去快速偵測臉部
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
	for face in faces:
		(xi, yi, w, h) = face
		face_region_gray = gray[yi:yi+h, xi:xi+w] # 取得偵測到的臉部範圍 矩陣從 [row, col] 開始, 所以y先, 才x

		predicted_id, confidence = recognizer.predict(face_region_gray)
		predicted_label = id_label_dict[predicted_id]

		face_img = "./face.png"
		cv2.imwrite(face_img, face_region_gray)

		# draw face region
		start_point = (xi, yi); end_point = (xi+w, yi+h)
		edge_color = (0, 0, 255) # BGR 0-255 
		stroke_width = 2
		cv2.rectangle(frame, start_point, end_point, edge_color, stroke_width)
		
		# draw face name
		font = cv2.FONT_HERSHEY_COMPLEX
		name = predicted_label
		if confidence<=30:
			name = "Unknown"
		
		# name = "handsome guy"
		color = (255, 255, 255)
		font_stroke_width = 2
		cv2.putText(frame, name, start_point, font, 1, color, font_stroke_width, cv2.LINE_8)

	video_writer.write(frame)
	cv2.imshow('video_window', frame)

	if (cv2.waitKey(20) & 0xFF) == ord('q'):
		break

cap.release()
video_writer.release()
cv2.destroyAllWindows()