import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import font

# gstreamer_pipeline(CSI Camera) 
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Application
class Application(tk.Frame):
    def __init__(self,master, video_source=0):
        super().__init__(master)
        
        self.master.geometry("1080x760")
        self.master.title("Tkinter with AI Rock-paper-scissors")

        # Font
        self.font_frame = font.Font( family="Meiryo UI", size=15, weight="normal" )
        self.font_btn_big = font.Font( family="Meiryo UI", size=20, weight="bold" )
        self.font_btn_small = font.Font( family="Meiryo UI", size=15, weight="bold" )

        self.font_lbl_bigger = font.Font( family="Meiryo UI", size=45, weight="bold" )
        self.font_lbl_big = font.Font( family="Meiryo UI", size=30, weight="bold" )
        self.font_lbl_middle = font.Font( family="Meiryo UI", size=15, weight="bold" )
        self.font_lbl_small = font.Font( family="Meiryo UI", size=12, weight="normal" )

        # Open the video source
        self.vcap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        self.width = self.vcap.get( cv2.CAP_PROP_FRAME_WIDTH )
        self.height = self.vcap.get( cv2.CAP_PROP_FRAME_HEIGHT )

        # Widget
        self.create_widgets()

        # Canvas Update
        self.delay = 15 #[mili seconds]
        self.mode = 0
        self.update()


    def create_widgets(self):

        #Frame_Camera1
        self.frame_cam = tk.LabelFrame(self.master, text = 'MY ONE HAND', font=self.font_frame, labelanchor="n")
        self.frame_cam.place(x = 100, y = 10)
        self.frame_cam.configure(width = 350, height = 350)
        self.frame_cam.grid_propagate(0)

        #Canvas1
        self.canvas1 = tk.Canvas(self.frame_cam)
        self.canvas1.configure( width= 300, height= 300)
        self.canvas1.grid(column= 0, row=0,padx = 20, pady=10)
        
        #Frame_img2
        self.frame_img2 = tk.LabelFrame(self.master, text = 'COMPUTER', font=self.font_frame, labelanchor="n")
        self.frame_img2.place(x = 600, y = 10)
        self.frame_img2.configure(width = 350, height = 350)
        self.frame_img2.grid_propagate(0)
        
         #Canvas2
        self.canvas2 = tk.Canvas(self.frame_img2)
        self.canvas2.configure( width= 300, height= 300)
        self.canvas2.grid(column= 0, row=0,padx = 20, pady=10)
        
        # Frame_Button(Control)
        self.frame_btn_control = tk.LabelFrame( self.master, text='Control', font=self.font_frame, labelanchor="n")
        self.frame_btn_control.place( x=35, y=500 )
        self.frame_btn_control.configure( width= 960, height= 120 )
        self.frame_btn_control.grid_propagate( 0 )

        # Start Button
        self.btn_start = tk.Button( self.frame_btn_control, text='Start', font=self.font_btn_big)
        self.btn_start.configure(width = 15, height = 1, command=self.press_start_button)
        self.btn_start.grid(column=0, row=0, padx=20, pady= 20)
        
        # Stop Button
        self.btn_stop = tk.Button( self.frame_btn_control, text='Stop', font=self.font_btn_big)
        self.btn_stop.configure(width = 15, height = 1, command=self.press_stop_button)
        self.btn_stop.grid(column=1, row=0, padx=20, pady= 20)
        
        # Close
        self.btn_close = tk.Button( self.frame_btn_control, text='Close', font=self.font_btn_big )
        self.btn_close.configure( width = 15, height = 1, command=self.press_close_button )
        self.btn_close.grid( column=2, row=0, padx=20, pady= 20 )
        
    def update(self):
        #Get a frame from the video source
        if self.mode == 1:
            
            _, frame = self.vcap.read()
    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_mini = cv2.resize(frame,(300,300))
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_mini))
    
            #self.photo -> Canvas
            self.canvas1.create_image(0,0, image= self.photo, anchor = tk.NW)
            
            self.img = PIL.Image.open(open('../data/png_data/0.png', 'rb'))
            self.img.thumbnail((300, 300), PIL.Image.ANTIALIAS)
            self.photo2 = PIL.ImageTk.PhotoImage(self.img)
    
            #self.photo2 -> Canvas
            self.canvas2.create_image(0,0, image= self.photo2, anchor = tk.NW)
            
    
        self.master.after(self.delay, self.update)

    def press_start_button(self):
        print("start!!")
        self.mode = 1
    
    def press_stop_button(self):
        print("stop!!")
        self.mode = 0
        

    def press_close_button(self):
        self.master.destroy()
        self.vcap.release()

    





def main():
    root = tk.Tk()
    app = Application(master=root)#Inherit
    app.mainloop()

if __name__ == "__main__":
    main()