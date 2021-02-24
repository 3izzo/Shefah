# import tkinter and all its functions
from tkinter import * 
from PIL import ImageTk, Image 
from tkinter import filedialog as fd
import cv2

# A variable to hold the video to process
vid = ''
result = ''
 
root = Tk() # create root window
root.title("Shefah")
root.maxsize(1000, 800) 
root.config(bg="white")

# Create Frames to organize widgets in the window
left_frame = Frame(root, width=650, height=400, bg='white')
left_frame.pack(side='left', fill='both', padx=10, pady=5, expand=True)
 
right_frame = Frame(root, width=320, height=400, bg='white')
right_frame.pack(side='right', fill='both', padx=10, pady=5, expand=True)

right_top_frame = Frame(right_frame,width=200, height=200,bg='white')
right_top_frame.pack(side='top', fill='both', padx=10, pady=5, expand=True)

right_bottom_frame = Frame(right_frame,width=200, height=200,bg='white')
right_bottom_frame.pack(side='bottom', fill='both', padx=10, pady=5, expand=True)

left_top_frame = Frame(left_frame,width=200, height=180,bg='white')
left_top_frame.pack(side='top', fill='both', padx=10, pady=5, expand=True)

left_mid_frame = Frame(left_frame,width=200, height=40,bg='white')
left_mid_frame.pack(side='top', fill='both', padx=10, pady=5, expand=True)

left_bottom_frame = Frame(left_frame,width=200, height=180,bg='white')
left_bottom_frame.pack(side='bottom', fill='both', padx=10, pady=5, expand=True)

# Welcome message
Label(left_mid_frame, text='Welcome to Shefah'+'\n'+'Please selelct a file or record a new video.', justify='center', bg='white').pack(side='top', padx=5, pady=5)
 
# Shefah's Logo
image = ImageTk.PhotoImage(Image.open("C:\\Users\\MHK47\\Desktop\\University\\8\\graduation project\\Github\\Shefah\\Shefah\\Interface\\TRQ-Grad_invert.png"))
panel = Label(right_top_frame, image = image, bg='white').pack(fill='both', padx=5, pady=5, side='top')
 

 
def open_filedialog():
    ''' Open filedialog to let the user choose the file to process '''
    vid = fd.askopenfilename()
    if(vid != ""):
        start_processing['state'] = 'normal'

def open_camera():
    ''' Open the user's camera to record a video to process '''
    vid = cv2.MyVideoCapture()
    start_processing['state'] = 'normal'

def process_video():
    ''' Send the video to Shefah's model to process '''
    save_output['state'] = 'normal'

def save_text():
    ''' Save the output in the desired location '''
    fd.asksaveasfilename()
 
# Buttons to make actions
select_a_file = Button(right_bottom_frame, text="Select a File", command=open_filedialog, width = 20)
select_a_file.pack(side='top',padx=5, pady=5, ipadx=10)

record_a_video = Button(right_bottom_frame, text="Record a Video", width = 20, command= open_camera)
record_a_video.pack(side='top',padx=5, pady=5, ipadx=10)

start_processing = Button(right_bottom_frame, text="Start Processing", width = 23, command=process_video, state=DISABLED)
start_processing.pack(side='top',padx=5, pady=5)

save_output = Button(right_bottom_frame, text="Save Output", width = 20, state=DISABLED)
save_output.pack(side='top',padx=5, pady=5, ipadx=10)

exit_shefah = Button(right_bottom_frame, text="Exit", command=quit, width = 23).pack(side='top',padx=5, pady=5)
 
root.mainloop()