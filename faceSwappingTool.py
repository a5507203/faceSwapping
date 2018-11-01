import cv2
import numpy
import tensorflow as tf
import argparse
import dlib
import face_recognition
import face_recognition_models
import os
import shutil


predict__model = face_recognition_models.pose_predictor_model_location()
pose_predictor = dlib.shape_predictor(predict__model)


def overWritePath(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

def detect_faces(frame, model="hog"):
    fp = face_recognition.face_locations(frame, model=model)
    if len(fp) > 0:
        fl = [fp[0]]
    else:
        fl = []

    lm = face_landmarks(frame, fl)
    for ((y, right, bottom, x), landmarks) in zip(fl, lm):
        yield DetectedFace(frame[y: bottom, x: right], x, right - x, y, bottom - y, landmarks)


# for ((y, right, bottom, x), landmarks) in zip(face_locations, landmarks):
#     yield DetectedFace(frame[y: bottom, x: right], x, right - x, y, bottom - y, landmarks)

def face_landmarks(face_image, face_locations):
    face_locations = [get_rect(face_location) for face_location in face_locations]
    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def get_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])



class DetectedFace(object):
    def __init__(self, image, x, w, y, h, landmarks):
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarks = landmarks

    def landmarksAsXY(self):
        return [(p.x, p.y) for p in self.landmarks.parts()]



class FaceSwappingTool(object):
    def __init__(self):

        self.sess = tf.Session()
        saver = tf.train.import_meta_graph('./save_model/model.ckpt.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./save_model'))
        graph = tf.get_default_graph()

        self.input_img = graph.get_tensor_by_name('input_1:0')
        self.encoder = graph.get_tensor_by_name('pixel_shuffler_1/Reshape_1:0')
        self.code = graph.get_tensor_by_name('input_2:0')
        self.decoder = graph.get_tensor_by_name('conv2d_9/Sigmoid:0')


    def patch_image( self, original, face_detected ):
        #assert image.shape == (256, 256, 3)
        image = cv2.resize(face_detected.image, (256, 256))
        crop = slice(48, 208)
        face = image[crop, crop]
        old_face = face.copy()
        face = cv2.resize(face, (64, 64))
        face = numpy.expand_dims(face, 0)
        code = self.sess.run(self.encoder,feed_dict={self.input_img:face / 255.0})
        new_face = self.sess.run(self.decoder,feed_dict={self.code:code})[0]

        new_face = numpy.clip(new_face * 255, 0, 255).astype(image.dtype)
        new_face = cv2.resize(new_face, (160, 160))
        self.adjust_avg_color(old_face,new_face)
        self.smooth_mask(old_face,new_face)

        new_face = self.superpose(image, new_face, crop)
        original[slice(face_detected.y, face_detected.y + face_detected.h), slice(face_detected.x, face_detected.x + face_detected.w)] = cv2.resize(new_face, (face_detected.w, face_detected.h))
        return original

    def adjust_avg_color(self,img_old,img_new):

        for i in range(img_new.shape[-1]):
            old_avg = img_old[:, :, i].mean()
            new_avg = img_new[:, :, i].mean()
            diff_int = (int)(old_avg - new_avg)
            for m in range(img_new.shape[0]):
                for n in range(img_new.shape[1]):
                    temp = (img_new[m,n,i] + diff_int)
                    if temp < 0:
                        img_new[m,n,i] = 0
                    elif temp > 255:
                        img_new[m,n,i] = 255
                    else:
                        img_new[m,n,i] = temp

    def smooth_mask(self,img_old,img_new):
        w,h,c = img_new.shape
        crop = slice(0,w)
        mask = numpy.zeros_like(img_new)
        mask[h//15:-h//15,w//15:-w//15,:] = 255
        mask = cv2.GaussianBlur(mask,(15,15),10)
        img_new[crop,crop] = mask/255*img_new + (1-mask/255)*img_old

    def superpose(self,image, new_face, crop):
        new_image = image.copy()
        new_image[crop, crop] = new_face
        return new_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="enter input images path",
        type=str
    )

    parser.add_argument(
        "output",
        help="enter output path",
        type=str
    )

    parser.add_argument(
        'type',
        help="select output type (image or video)",
        choices=['image', 'video']
    )
    args = parser.parse_args()

    frames_dirs = [[os.path.join(args.input, f), os.path.join(args.output, f)] for f in os.listdir(args.input)]
    overWritePath(args.output)
    FaceSwapTool = FaceSwappingTool()


    if(args.type == 'image'):
        for frame_dir in frames_dirs:
            img = cv2.imread(frame_dir[0])
            for preproceedImg in detect_faces(img):
            # preproceedImg = detect_faces(img)[0]
            # print(preproceedImg)
                newIm = FaceSwapTool.patch_image(img,preproceedImg)
                cv2.imwrite(frame_dir[1],newIm)
                print('converting image')

    else:

        video = None
        first_flag = 0
        for frame_dir in frames_dirs:
            img = cv2.imread(frame_dir[0])
            for preproceedImg in detect_faces(img):
                newIm = FaceSwapTool.patch_image(img,preproceedImg)
                if(first_flag == 0):

                    height, width, layers = newIm.shape
                    # video = cv2.VideoWriter(os.path.join(args.output,'video.avi'), -1, 1, (width, height))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
                    out = cv2.VideoWriter(os.path.join(args.output, 'video.avi'), fourcc, 20.0, (width, height))
                    first_flag = 1
                print('converting video')
                out.write(newIm)

        out.release()
        cv2.destroyAllWindows()
