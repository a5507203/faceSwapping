
import youtube_dl
import os
import tqdm
import time
import cv2
from VideoClip import*
from FaceIsolation import*



class ConvertVideo(object): 

    def __init__(self, download_video_path='./asset/video', videos_frames_dir='./asset/videoframes',faces_path = './asset/faces'):
        self.faces_path = faces_path
        self.download_video_path = download_video_path
        self.videos_frames_dir = videos_frames_dir
        self.fps = 1
        self.online_videos = {}
        self.face_isolation = FaceIsolation()


    def add_video(self, person, video_url):
        self.online_videos[person] = video_url


     
    def fetch_video(self, fetch_voice = False):
        if (fetch_voice):
            format = 'bestvideo+bestaudio/best'
            
        else:
            format = 'bestvideo/best'
        for person, video_url in self.online_videos.items():

            options = {
                'format': format,
                'outtmpl': os.path.join(self.download_video_path, person)+'.%(ext)s',
            }
            with youtube_dl.YoutubeDL(options) as ydl:
                ydl.download([video_url])


    def clip_videos(self, scource = 1):
        if (scource == 1):
            video_files = [f for f in os.listdir(self.download_video_path)]
            print('asset list'+str(video_files))

            for video_file in video_files:
                self.clip_video( os.path.join(self.download_video_path,video_file), os.path.join(self.videos_frames_dir, video_file))
        else:
            #may need to add webcam
            pass

    def _clip_video(self, video_path, video_frames_dir):
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        success = True
        os.makedirs(video_frames_dir)
        vedio_fps = vidcap.get(cv2.CAP_PROP_FPS)
        print(vedio_fps)
        delay = int(vedio_fps/self.fps)

        while success:
            if(count % delay == 0):
                cv2.imwrite(video_frames_dir+"/frame%d.jpg" % count, image)     # save frame as JPEG file 
            success,image = vidcap.read()
            count += 1
        print("clip_complete")



    def _extract_faces(self):
        frames_dirs = [f for f in os.listdir(self.videos_frames_dir)]
        for frames_dir in frames_dirs:
            self.face_isolation.isolate_faces( os.path.join(self.videos_frames_dir,frames_dir), os.path.join(self.faces_path, frames_dir))











