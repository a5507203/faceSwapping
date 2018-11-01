import os
class PersonInfo(object):

    def __init__(self, person, videos_frames_dir='./asset/videoframes', faces_path = './asset/faces' ):

        self.person = person
        self.faces_path = faces_path
        self.videos_frames_dir = videos_frames_dir

        self.faces = self.get_images(self.faces_path)
        self.frames = self.get_images(self.videos_frames_dir)

    def get_images(self, dir):
        person_image_dir = os.path.join(dir, self.person)
        return [os.path.join(person_image_dir,f) for f in os.listdir(person_image_dir)]




