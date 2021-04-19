from tkinter import *
import threading
import tkinter
from PIL import ImageTk, Image
from tkinter import filedialog as fd
import cv2
import imageio
from tkinter import ttk
from tkinter import messagebox
import tkinter.font as font
import time
from preprocess_videos import get_frames_mouth, face_detector, predictor
from Utilities import *
from Displaythread import Displaythread
from predict import *

# add labels to the squares
# add constrains notifacation


class App:
    def __init__(self, master):
        self.input_thread = None
        self.face_thread = None
        self.ROI_thread = None
        self.processing_thread = None
        self.recording = False
        self.master = master
        self.pac = None
        self.input_video = []
        self.face_video = []
        self.ROI_video = []
        self.status = ""  # To change the current function of the partial bar
        self.text_size = font.Font(size=14)
        self.shefah_model = load_model()

        master.title("Shefah")
        master.config(bg="white")
        master.minsize(900, 600)

        logo = PhotoImage(file=".\\icons\\logo.png")

        # Setting icon of master window
        master.iconphoto(False, logo)
        # Create Frames to organize widgets in the window
        # self.frame_r = Frame(master, width=300, bg="white")
        # self.frame_r.pack(side="right", fill=Y, padx=2, pady=2)

        self.frame = Frame(master, bg="white")
        self.frame.pack(side="left", expand=1, fill=BOTH)
        Grid.columnconfigure(self.frame, 0, weight=1)
        Grid.columnconfigure(self.frame, 1, weight=1)

        Grid.rowconfigure(self.frame, 0, weight=1)
        Grid.rowconfigure(self.frame, 2, weight=1)

        self.frame_t = Frame(self.frame, bg="#c2c3c4")
        self.frame_t.grid(row=0, column=0, padx=8, pady=8, columnspan=2, sticky=N + S + E + W)

        # self.text_frame_t = Frame(self.frame_t, bg="#c2c3c4")

        # self.text_frame_t.pack(pady=80)

        Grid.rowconfigure(self.frame_t, 0, weight=1)
        Grid.rowconfigure(self.frame_t, 1, weight=1)
        Grid.rowconfigure(self.frame_t, 2, weight=1)

        Grid.columnconfigure(self.frame_t, 0, weight=1)
        Grid.columnconfigure(self.frame_t, 1, weight=1)
        Grid.columnconfigure(self.frame_t, 2, weight=1)
        self.frame_t.pack_propagate(False)

        # self.inner_frame_t = Frame(self.frame_t, bg="#c2c3c4")
        # self.inner_frame_t.grid(row=0, column=0, padx=8, pady=8,
        #                   columnspan=3, rowspan = 4, sticky=N + S + E + W)
        # self.inner_frame_t.pack_propagate(False)

        # Where the output should be
        self.frame_m = Frame(self.frame, bg="white")
        self.frame_m.grid(row=1, column=0, padx=8, pady=8, columnspan=2, sticky=N + S + E + W)

        # Where the preprocessed videos should be
        self.frame_b_l = Frame(self.frame, bg="#c2c3c4")
        self.frame_b_r = Frame(self.frame, bg="#c2c3c4")
        self.frame_b_l.pack_propagate(False)

        self.frame_b_l.grid(row=2, column=0, padx=8, pady=8, columnspan=1, sticky=N + S + E + W)
        self.frame_b_r.grid(row=2, column=1, padx=8, pady=8, columnspan=1, sticky=N + S + E + W)
        self.frame_b_r.pack_propagate(False)

        Grid.rowconfigure(self.frame_b_l, 0, weight=1)
        Grid.rowconfigure(self.frame_b_l, 1, weight=1)
        Grid.rowconfigure(self.frame_b_l, 2, weight=1)

        Grid.columnconfigure(self.frame_b_l, 0, weight=1)
        Grid.columnconfigure(self.frame_b_l, 1, weight=1)
        Grid.columnconfigure(self.frame_b_l, 2, weight=1)

        Grid.rowconfigure(self.frame_b_r, 0, weight=1)
        Grid.rowconfigure(self.frame_b_r, 1, weight=1)
        Grid.rowconfigure(self.frame_b_r, 2, weight=1)

        Grid.columnconfigure(self.frame_b_r, 0, weight=1)
        Grid.columnconfigure(self.frame_b_r, 1, weight=1)
        Grid.columnconfigure(self.frame_b_r, 2, weight=1)

        self.frame_m_b = Frame(self.frame_m, bg="white")
        self.frame_m_b.pack(side=TOP)

        img_video_in = ImageTk.PhotoImage(Image.open(".\\icons\\video_in.png"))
        self.text_t = Label(self.frame_t, bg="#c2c3c4")
        self.text_t.image = img_video_in
        self.text_t.config(image=img_video_in)
        self.text_t.pack(expand=True, fill=BOTH)

        img_face = ImageTk.PhotoImage(Image.open(".\\icons\\face.png"))
        self.text_b_l = Label(self.frame_b_l, bg="#c2c3c4")
        self.text_b_l.image = img_face
        self.text_b_l.config(image=img_face)
        self.text_b_l.pack(expand=True, fill=BOTH)

        img_lips = ImageTk.PhotoImage(Image.open(".\\icons\\lips.png"))
        self.text_b_r = Label(self.frame_b_r, bg="#c2c3c4")
        self.text_b_r.image = img_lips
        self.text_b_r.config(image=img_lips)
        self.text_b_r.pack(expand=True, fill=BOTH)

        # Buttons to make actions
        img_select = ImageTk.PhotoImage(Image.open(".\\icons\\select.png"))
        self.btn_select = Button(
            self.frame_m_b, text="اختر فيديو ", command=self.open_filedialog, compound=RIGHT, width=120
        )
        self.btn_select.image = img_select
        self.btn_select.config(image=img_select)
        self.btn_select.grid(column=0, row=0, sticky=NSEW, padx=4, pady=2)

        img_camera = ImageTk.PhotoImage(Image.open(".\\icons\\camera.png"))
        self.btn_record = Button(
            self.frame_m_b, text="افتح الكاميرا ", command=self.open_camera, compound=RIGHT, width=120
        )
        self.btn_record.image = img_camera
        self.btn_record.config(image=img_camera)

        self.btn_record.grid(column=1, row=0, sticky=NSEW, padx=4, pady=2)

        self.btn_record["font"] = self.btn_select["font"] = self.text_size

        img_record = ImageTk.PhotoImage(Image.open(".\\icons\\record.png"))
        self.btn_record_start = Button(
            self.frame_m_b, text="سجل ", command=self.toggle_recording, compound=RIGHT, width=120
        )
        self.btn_record_start.image = img_record
        self.btn_record_start.config(image=img_record)
        self.btn_record_start.grid(column=1, row=0, sticky=NSEW, padx=4, pady=2)
        self.btn_record_start.grid_remove()

        img_stop = ImageTk.PhotoImage(Image.open(".\\icons\\stop.png"))
        self.btn_record_end = Button(
            self.frame_m_b, text="توقف ", command=self.toggle_recording, compound=RIGHT, width=120
        )
        self.btn_record_end.image = img_stop
        self.btn_record_end.config(image=img_stop)
        self.btn_record_end.grid(column=2, row=0, sticky=NSEW, padx=4, pady=2)
        self.btn_record_end.grid_remove()

        self.btn_record_start["font"] = self.btn_record_end["font"] = self.text_size

        # progress bars
        self.partial_progress_bar_label = Label(self.frame, text="", bg="white")
        self.partial_progress_bar_label.grid(row=100, columnspan=2, padx=4, sticky=E)
        self.partial_progress_bar = ttk.Progressbar(self.frame, orient=HORIZONTAL, mode="determinate")
        self.partial_progress_bar.grid(row=101, columnspan=2, padx=8, sticky=W + E)

        self.total_progress_bar_label = Label(self.frame, text="نسبة الإنجاز", bg="white")
        self.total_progress_bar_label.grid(row=102, columnspan=2, padx=4, sticky=E)
        self.total_progress_bar = ttk.Progressbar(self.frame, orient=HORIZONTAL, mode="determinate")
        self.total_progress_bar.grid(row=103, columnspan=2, padx=8, pady=4, sticky=W + E)

        self.total_progress_bar_label["font"] = self.partial_progress_bar_label["font"] = self.text_size

        self.total_progress_bar_label.grid_remove()
        self.total_progress_bar.grid_remove()
        self.partial_progress_bar_label.grid_remove()
        self.partial_progress_bar.grid_remove()

        # self.btn_prcs = Button(frame_helper, text="ابدأ المعالجة", command=self.process_video, state=DISABLED)
        # self.btn_prcs.grid(row=3, sticky=W + E, pady=2)
        # self.btn_exit = Button(frame_helper, text="خروج", command=quit)
        # self.btn_exit.grid(row=4, sticky=W + E, pady=2)

        # padding
        # Frame(frame_helper, height=16, bg="white").grid(row=10)

    def stream(self, video, vid_label, parent):
        """ takes a video and play the video """
        start_time = time.time()
        event = threading.Event()
        while True:
            for image in video:
                frame = Image.fromarray(image)
                h = vid_label.winfo_height()
                if h < 20:
                    h = 20
                w = int(h * (frame.width / frame.height))
                parent_width = vid_label.winfo_width()
                if w > parent_width:
                    ratio = parent_width / w
                    h = int(h * ratio)
                    w = int(w * ratio)
                frame = frame.resize((w, h))
                frame_image = ImageTk.PhotoImage(frame)
                vid_label.config(image=frame_image)
                vid_label.image = frame_image
                end_time = time.time()
                waiting_time = 1 / 30 - (end_time - start_time)
                start_time = time.time() + waiting_time
                if waiting_time > 0:
                    event.wait(waiting_time)
            event.wait(1 / 30)

    def open_filedialog(self):
        """ Open filedialog to let the user choose the file to process """
        messagebox.showwarning(
            "تنبيه",
            "عند اختيار الفيديو يشترط التالي:\n1. أن يحتوي الفيديو على شخص واحد فقط.\n2. أن يكون وجه وشفتين المتحدث واضحتين.\n3. أن ينطق المتحدث رقم واحد بين 0-9 بشكل واضح.\n4. أن لا تتعدا مدة الفيديو عن ثانيتين وإذا تعدا سيتم اخذ اخر ثانيتين.",
        )
        video_path = fd.askopenfilename()

        for child in self.frame_t.winfo_children():
            child.destroy()
        if self.input_thread != None:
            self.input_thread.raise_exception()

        for child in self.frame_b_l.winfo_children():
            child.destroy()
        if self.face_thread != None:
            self.face_thread.raise_exception()

        for child in self.frame_b_r.winfo_children():
            child.destroy()
        if self.ROI_thread != None:
            self.ROI_thread.raise_exception()

        self.input_video = [i for i in imageio.get_reader(video_path).iter_data()]
        if self.input_video:
            vid_label = Label(self.frame_t, bg="black")
            vid_label.pack(expand=True, fill=BOTH)
            self.input_thread = Displaythread(target=self.stream, args=(self.input_video, vid_label, self.frame_t))
            self.input_thread.daemon = 1
            self.input_thread.start()
            self.process_video()
            # self.btn_prcs["state"] = "normal"

    def open_camera(self):
        """ Open the user's camera to record a video to process """
        messagebox.showwarning(
            "تنبيه",
            "عند تسجيل الفيديو يشترط التالي:\n1. أن يحتوي الفيديو على شخص واحد فقط.\n2. أن يكون وجه وشفتين المتحدث واضحتين.\n3. أن ينطق المتحدث رقم واحد بين 0-9 بشكل واضح.\n4. أن لا تتعدا مدة الفيديو عن ثانيتين وإذا تعدا سيتم اخذ اخر ثانيتين.",
        )
        for child in self.frame_t.winfo_children():
            child.destroy()
        if self.input_thread != None:
            self.input_thread.raise_exception()

        for child in self.frame_b_l.winfo_children():
            child.destroy()
        if self.face_thread != None:
            self.face_thread.raise_exception()

        for child in self.frame_b_r.winfo_children():
            child.destroy()
        if self.ROI_thread != None:
            self.ROI_thread.raise_exception()

        self.cap = cv2.VideoCapture(0)
        if self.cap:
            vid_label = Label(self.frame_t, bg="black")
            vid_label.pack(expand=True, fill=BOTH)
            self.input_thread = Displaythread(
                target=self.stream_camera_and_capture, args=(self.cap, vid_label, self.frame_t)
            )
            self.input_thread.daemon = 1
            self.input_thread.start()

        self.btn_select.grid(column=0, row=0, sticky=W + E, pady=2)
        # self.btn_prcs["state"] = DISABLED
        self.btn_record.grid_remove()
        self.btn_record_start.grid()
        self.btn_record_end.grid()
        self.btn_record_end["state"] = DISABLED

    def toggle_recording(self):

        self.recording = not self.recording

        if self.recording:
            self.btn_select["state"] = DISABLED
            self.input_video = []
            self.btn_record_start["state"] = DISABLED
            self.btn_record_end["state"] = NORMAL

        else:
            self.btn_select["state"] = NORMAL
            self.btn_record.grid()
            self.btn_record_end.grid_remove()
            self.btn_record_start.grid_remove()
            self.cap.release()

            for child in self.frame_t.winfo_children():
                child.destroy()
            self.input_thread.raise_exception()

            if self.input_video:
                vid_label = Label(self.frame_t, bg="black")
                vid_label.pack(expand=True, fill=BOTH)
                self.input_thread = Displaythread(target=self.stream, args=(self.input_video, vid_label, self.frame_t))
                self.input_thread.daemon = 1
                self.input_thread.start()
                # self.btn_prcs["state"] = "normal"
                self.process_video()

    def stream_camera_and_capture(self, capture, vid_label, parent):
        """ takes a video and play the video """
        start_time = time.time()
        event = threading.Event()
        while True:
            _, image = capture.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            image = cv2.flip(image, 1)

            if self.recording:
                self.input_video.append(image)
                if len(self.input_video) > 75:
                    self.input_video.pop(0)

            frame = Image.fromarray(image)
            h = parent.winfo_height()
            w = int(h * (frame.width / frame.height))
            frame = frame.resize((w, h))

            frame_image = ImageTk.PhotoImage(frame)
            vid_label.config(image=frame_image)
            vid_label.image = frame_image
            end_time = time.time()
            waiting_time = 1 / 30 - (end_time - start_time)
            start_time = time.time() + waiting_time
            if waiting_time > 0:
                event.wait(waiting_time)

    def process_video(self):

        # self.text_frame_b_l.destroy()
        # self.text_frame_b_r.destroy()

        self.total_progress_bar_label.grid()
        self.total_progress_bar.grid()
        self.partial_progress_bar_label.grid()
        self.partial_progress_bar.grid()

        self.face_video = []
        vid_label = Label(self.frame_b_l, bg="black")
        vid_label.pack(expand=True, fill=BOTH)
        self.face_thread = Displaythread(target=self.stream, args=(self.face_video, vid_label, self.frame_t))
        self.face_thread.daemon = 1
        self.face_thread.start()

        self.ROI_video = []
        vid_label = Label(self.frame_b_r, bg="black")
        vid_label.pack(expand=True, fill=BOTH)
        self.ROI_thread = Displaythread(target=self.stream, args=(self.ROI_video, vid_label, self.frame_t))
        self.ROI_thread.daemon = 1
        self.ROI_thread.start()

        def inner_func():
            time.sleep(0.5)
            i = 0
            video_for_prediction = []
            self.partial_progress_bar_label["text"] = "معالجة"
            self.partial_progress_bar["value"] = 10
            for frame in self.input_video:
                frame = frame[:, :, :3]
                cropped_frame = get_frames_mouth(face_detector, predictor, frame, interface=self)

                self.partial_progress_bar["value"] += i

                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                # cv2.imwrite(".\\G\\frame%d.png" % i, cropped_frame)
                video_for_prediction.append(cropped_frame)
                i += 1

            while i < max_frame_count:
                video_for_prediction.append(np.zeros((frame_h, frame_w, 3)))
                i += 1
            self.partial_progress_bar["value"] = 100
            self.total_progress_bar["value"] = 25
            self.partial_progress_bar_label["text"] = "تحميل النموذج"
            self.partial_progress_bar["value"] = 10
            video_for_prediction = np.array([video_for_prediction]).astype(np.float32) / 255
            self.partial_progress_bar["value"] = 50
            # shefah_model = load_model()
            self.partial_progress_bar["value"] = 100

            self.total_progress_bar["value"] = 75

            self.partial_progress_bar_label["text"] = "يتوقع"
            self.partial_progress_bar["value"] = 30
            (predicted, predicted_as_number) = predict_lip(video_for_prediction, self.shefah_model)
            self.partial_progress_bar["value"] = 100

            self.partial_progress_bar_label["text"] = "انتهى"

            self.total_progress_bar["value"] = 100

            self.result = Label(
                self.frame, text="الرقم المنطوق: %s" % predicted_as_number, bg="white", font=("Arial", 20)
            )
            self.result.grid(row=3000, column=0, padx=8, pady=8, columnspan=2, sticky=N + S + E + W)

        if self.processing_thread != None:
            self.processing_thread.raise_exception()
        self.processing_thread = Displaythread(target=inner_func, args=())
        self.processing_thread.daemon = 1
        self.processing_thread.start()


root = Tk()
my_gui = App(root)
# root.call("wm", "iconphoto", root._w, PhotoImage(file=".\\logo.png"))
root.mainloop()
