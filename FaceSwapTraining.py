from personInfo import *

from generateInput import*
from model import *
import cv2



class FaceSwapTraining(object):

    def __init__(self, person_a, person_b, videos_frames_dir='./asset/videoframes',faces_path = './asset/faces', save_interval = 1, batch_size = 64):
        self.faces_path = faces_path
        self.videos_frames_dir = videos_frames_dir
        self.save_interval = save_interval
        self.source_person = PersonInfo(person_a)
        self.target_person = PersonInfo(person_b)
        self.batch_size = batch_size
        images_A_path = self.source_person.faces
        images_B_path = self.source_person.faces
        generator = TrainingDataGenerator(160)
        self.batch_images_A = generator.minibatchAB(images_A_path, self.batch_size)
        self.batch_images_B = generator.minibatchAB(images_B_path, self.batch_size)
        self.model = Model()
        self.model.load_model()



    def _train_one_step(self, interation):

        print('training start...')
        epoch, warped_A, target_A = next(self.batch_images_A)
        epoch, warped_B, target_B = next(self.batch_images_B)

        loss_A, loss_B = self.model.train_on_batch(warped_A, target_A, warped_B, target_B)
        print("interation:%d loss_A:%1.2f loss_B:%1.2f"%(interation,loss_A,loss_B))


    def start_train(self):


        try:
            print('Starting. Press "Enter" to stop training and save model')
            for epoch in range(0, 10000000):
                save_iteration = epoch % self.save_interval == 0
                self._train_one_step(epoch)
                if save_iteration:
                    self.model.save_model(epoch)
        except KeyboardInterrupt:
            try:
                pass
                self.model.save_model(epoch)
            except KeyboardInterrupt:
                print('Saving model weights has been cancelled!')
            exit(0)
        except Exception as e:
            print(e)
            exit(1)





