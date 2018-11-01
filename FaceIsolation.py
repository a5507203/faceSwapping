import sys
import os
import numpy as np
import cv2
import dlib
import imutils

class FaceIsolation(object): 

	def __init__(self):

		self.DESIRE_FACE_WIDTH = 512
		self.DESIRE_FACE_HEIGHT = 512
		self.DESIRE_EYE_DIST_RATE = 0.5
		self.LEFT_EYE_IDX = (36, 42)
		self.RIGHT_EYE_IDX = (42, 48)
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor("./dat/shape_predictor_68_face_landmarks.dat")

	def isolate_faces(self, input_folder , output_folder):
		os.makedirs(output_folder)
		print('start faces isloation from'+input_folder +' to '+output_folder)

		for img_name in os.listdir(input_folder):

			img = cv2.imread(input_folder + '/' + img_name)
			if img is None:
				continue
			img = imutils.resize(img, width=1200)
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			rects = self.detector(img_gray, 1)

			for n, rect in enumerate(rects):
				x = rect.left()
				y = rect.top()
				w = rect.right() - x
				h = rect.bottom() - y
				faceOrig = imutils.resize(img_gray[y:y+h, x:x+w], width=256)

				shape_tmp = self.predictor(img_gray, rect)
				shape = np.zeros((shape_tmp.num_parts, 2), dtype='int')
				for i in range(shape_tmp.num_parts):
					shape[i] = (shape_tmp.part(i).x, shape_tmp.part(i).y)

				le = shape[self.LEFT_EYE_IDX[0]:self.LEFT_EYE_IDX[1]]
				re = shape[self.RIGHT_EYE_IDX[0]:self.RIGHT_EYE_IDX[1]]
				lec = le.mean(axis=0).astype('int')
				rec = re.mean(axis=0).astype('int')
				dx = rec[0] - lec[0]
				dy = rec[1] - lec[1]
				angle = np.degrees(np.arctan2(dy, dx))
				dist = np.sqrt((dx ** 2) + (dy ** 2))

				scale = self.DESIRE_FACE_WIDTH * self.DESIRE_EYE_DIST_RATE / dist
				fc = tuple(shape[17:].mean(axis=0).astype('int'))
				M = cv2.getRotationMatrix2D(fc, angle, scale)
				tx = self.DESIRE_FACE_WIDTH * 0.5
				ty = self.DESIRE_FACE_HEIGHT * 0.5
				M[0,2] += (tx - fc[0])
				M[1,2] += (ty - fc[1])

				faceAligned = cv2.warpAffine(img, M, (self.DESIRE_FACE_WIDTH, self.DESIRE_FACE_HEIGHT), flags=cv2.INTER_CUBIC)

				cv2.imwrite(output_folder + '/' + 'Aligned_' + img_name.split('.')[0] + '_' + str(n) + '.jpg', faceAligned)
				cv2.waitKey(0)

