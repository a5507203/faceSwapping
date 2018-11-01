from ConvertVideo import *
from FaceSwapTraining import *
import tensorflow as tf
from face_detection import *
import cv2

import generateInput









face = FaceSwapTraining(person_a='been.mp4', person_b='oliver.mp4')
#
# img2 = face.swap_face(['./asset/faces/been.mp4/Aligned_frame0_0.jpg'])
# cv2.imwrite(['./asset/converted.jpg'],img2)

# face.train('oliver.mp4','asdf')
# import imageio
# imageio.plugins.ffmpeg.download()
# convertvideo = ConvertVideo()

# convertvideo.add_video('oliver', 'https://www.youtube.com/watch?v=7XOBxNqNl3o')

# convertvideo.fetch_video()
# convertvideo._extract_faces()

# print("iter:%d loss_A:%1.2f loss_B:%1.2f" % (iter, loss_A, loss_B))

# sess = tf.Session()
# saver = tf.train.import_meta_graph('./save_model/model.ckpt.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./save_model'))
#
# graph = tf.get_default_graph()

# for op in tf.get_default_graph().get_operations():
#     print(str(op.name))###



# generator = TrainingDataGenerator(160)
# img_paths = ['./asset/faces/been.mp4/Aligned_frame0_0.jpg']
# batch_images = generator.minibatchAB(img_paths, len(img_paths))
# # for img_path in img_paths:
# # img = cv2.imread(img_path)
# # print(img)
# epoch, warped_A, target_A = next(batch_images)




# print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


# inputs = graph.get_tensor_by_name('input_1:0')
# #
# encoder = graph.get_tensor_by_name('pixel_shuffler_1/Reshape_1:0')
#
#
# inputs2 = graph.get_tensor_by_name('input_2:0')
#
# outputs = graph.get_tensor_by_name('conv2d_9/Sigmoid:0')
#


# code = sess.run(encoder,feed_dict={inputs:target_A})
#
# imgs = sess.run(outputs,feed_dict={inputs2:code})
#
#
# for img in imgs:
#     print(img)
#     img = img*255
#     cv2.imwrite('./asset/converted.jpg', img)
# self.saver.restore(self.sess, checkpoint.model_checkpoint_path)