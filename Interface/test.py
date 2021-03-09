from tkinter import *
import threading
from PIL import ImageTk, Image 
from tkinter import filedialog as fd
import cv2
import ffmpeg_streaming
import numpy as np
import imageio

class App:
    def __init__(self, master):
        self.master = master
        self.video =''

        master.title('name')
        master.config(bg='white')
        master.minsize(1000,800)

        # Create Frames to organize widgets in the window
        self.left_frame = Frame(master, width=700, height=800, bg='white')
        self.left_frame.pack(side='left', fill='both', padx=10, pady=5, expand=True)
 
        self.right_frame = Frame(master, width=300, height=400, bg='white')
        self.right_frame.pack(side='right', fill='y', padx=10, pady=5, expand=False)

        self.right_top_frame = Frame(self.right_frame,width=300, height=300,bg='white')
        self.right_top_frame.pack(side='top', fill='both', padx=10, pady=5, expand=True)

        self.right_bottom_frame = Frame(self.right_frame,width=300, height=500,bg='white')
        self.right_bottom_frame.pack(side='bottom', fill='both', padx=10, pady=5, expand=True)

        self.left_top_frame = Frame(self.left_frame,width=700, height=400,bg='white')
        self.left_top_frame.pack(side='top', fill='both', padx=10, pady=5, expand=True)

        self.left_bottom_frame = Frame(self.left_frame,width=700, height=180,bg='white')
        self.left_bottom_frame.pack(side='bottom', fill='both', padx=10, pady=5, expand=True)

        self.left_mid_frame = Frame(self.left_frame,width=700, height=10,bg='white')
        self.left_mid_frame.pack(side='bottom', fill='x', padx=10, pady=5, expand=False)


        self.left_bottom_right_frame = Frame(self.left_bottom_frame,width=100, height=180,bg='white')
        self.left_bottom_right_frame.pack(side='right', fill='both', padx=10, pady=5, expand=True)

        self.left_bottom_left_frame = Frame(self.left_bottom_frame,width=100, height=180,bg='white')
        self.left_bottom_left_frame.pack(side='right', fill='both', padx=10, pady=5, expand=True)
        
        # Welcome message
        self.label = Label(self.left_mid_frame, text='Welcome to Shefah'+'\n'+'Please selelct a file or record a new video.', justify='center', bg='white').pack(side='top', padx=5, pady=5)
 
        # Shefah's Logo
        self.image = Image.open("C:\\Users\\MHK47\\Desktop\\University\\8\\graduation project\\Github\\Shefah\\Shefah\\Interface\\logo.png")
        self.image = self.image.resize((300, 278), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.image)
        self.panel = Label(self.right_top_frame, image = self.img, bg='white').pack(fill='both', padx=5, pady=5, side='top', expand = False)

        def stream(path, vid_label):
            ''' takes a video's path and play the video ''' 
            self.video = imageio.get_reader(path)

            for image in self.video.iter_data():
                frame = Image.fromarray(image)
                frame = frame.resize((700,400))
                frame_image = ImageTk.PhotoImage(frame)
                vid_label.config(image=frame_image)
                vid_label.image = frame_image
 

        def open_filedialog():
            ''' Open filedialog to let the user choose the file to process '''
            self.video = fd.askopenfilename()
            if(self.video != ""):
                vid_label = Label(self.left_top_frame, bg='black')
                vid_label.pack(expand=True)
                thread = threading.Thread(target=stream, args=(self.video,vid_label))
                thread.daemon = 1
                thread.start()
                self.start_processing['state'] = 'normal'

        def open_camera():
            ''' Open the user's camera to record a video to process '''
            self.video = cv2.MyVideoCapture(0)
            if(self.video != ""):
                vid_label = Label(self.left_top_frame, bg='black')
                vid_label.pack(expand=True)
                thread = threading.Thread(target=stream, args=(self.video,vid_label))
                thread.daemon = 1
                thread.start()
            stream(self.video, vid_label)
            
            start_processing['state'] = 'normal'

        def process_video():
            ''' Send the video to Shefah's model to process '''
            # pridict(video)

        # Buttons to make actions
        self.select_a_file = Button(self.right_bottom_frame, text="Select a File", command=open_filedialog, width = 20)
        self.select_a_file.pack(side='top',padx=5, pady=5, ipadx=10)

        self.record_a_video = Button(self.right_bottom_frame, text="Record a Video", width = 20, command= open_camera)
        self.record_a_video.pack(side='top',padx=5, pady=5, ipadx=10)

        self.start_processing = Button(self.right_bottom_frame, text="Start Processing", width = 23, command=process_video, state=DISABLED)
        self.start_processing.pack(side='top',padx=5, pady=5)

        self.exit_shefah = Button(self.right_bottom_frame, text="Exit", command=quit, width = 23).pack(side='top',padx=5, pady=5)


root = Tk()
my_gui = App(root)
root.mainloop()