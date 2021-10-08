import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import font

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import argparse

import numpy as np
import mediapipe as mp

from utils import CvFpsCalc

landmark_jan = []

#MLP##################################################↓

class MLP(nn.Module):
    # network
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(42, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x= self.fc5(x)

        return x

mlp=MLP()
model_path = './data/model_data/model1.pth'
mlp.load_state_dict(torch.load(model_path))

# model load #############################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
max_num_hands=1,
min_detection_confidence=0.7,
min_tracking_confidence=0.5,
)

        # FPS Measurement module ########################################################
#cvFpsCalc = CvFpsCalc(buffer_len=10)

def discrimination(input_x):
  outputs = mlp(input_x)
  print(outputs)
  tensor_label = outputs.argmax(dim=1, keepdim=True)
  out_label = tensor_label.item()
  return out_label

##################################################↑

#mediapipe#########################################↓


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
        self.cap = cv2.VideoCapture(0)
        self.width = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.height = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)

        # Widget
        self.create_widgets()

        #MLP######################################

        #############################################

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

        if self.mode == 1:
            
            ret, frame = self.cap.read()
            landmark_jan = []
    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_mini = cv2.resize(frame,(300,300))
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame_mini))
    
            #self.photo -> Canvas
            self.canvas1.create_image(0,0, image= self.photo, anchor = tk.NW)

            # Camera capture #####################################################
            #ret, image = cap.read()
            retflag=1
            if not ret:
                retflag = 0
            #image = cv2.flip(frame, 1)  # ミラー表示
            #debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if retflag == 1:
                results = hands.process(frame)

            # Drawing ################################################################
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                        for index, landmark in enumerate(hand_landmarks.landmark):
                            if landmark.visibility < 0 or landmark.presence < 0:
                                continue
                            landmark_x = min(int(landmark.x * 1080), 1080 - 1)
                            landmark_y = min(int(landmark.y * 760), 760 - 1)
                            landmark_jan.append(landmark_x)
                            landmark_jan.append(landmark_y)

            #cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                      #cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            if len(landmark_jan) == 42:
                landmark_jan2 = np.array(landmark_jan).reshape(1,-1).tolist()
                in_torch = torch.tensor(landmark_jan2,dtype=torch.float32)
                out=discrimination(in_torch)
                print("out_label:{}".format(out))
                if out == 0:
                    self.img = PIL.Image.open(open('./data/png_data/2.png', 'rb'))
                    self.img.thumbnail((300, 300), PIL.Image.ANTIALIAS)
                    self.photo2 = PIL.ImageTk.PhotoImage(self.img)
                    self.canvas2.create_image(0,0, image= self.photo2, anchor = tk.NW)
                elif out == 1:
                    self.img = PIL.Image.open(open('./data/png_data/0.png', 'rb'))
                    self.img.thumbnail((300, 300), PIL.Image.ANTIALIAS)
                    self.photo2 = PIL.ImageTk.PhotoImage(self.img)
                    self.canvas2.create_image(0,0, image= self.photo2, anchor = tk.NW)
                elif out == 2:
                    self.img = PIL.Image.open(open('./data/png_data/1.png', 'rb'))
                    self.img.thumbnail((300, 300), PIL.Image.ANTIALIAS)
                    self.photo2 = PIL.ImageTk.PhotoImage(self.img)
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
        self.cap.release()

    





def main():
    root = tk.Tk()
    app = Application(master=root)#Inherit
    app.mainloop()

if __name__ == "__main__":
    main()
