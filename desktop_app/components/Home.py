import tkinter as tk
from tkinter import Toplevel, PhotoImage, filedialog
from PIL import Image, ImageTk
import cv2
from services.Introductions import get_introductions
from services.Histories import send_push_notification
from models.plank.PlankModel import PlankModel
from models.squat.SquatModel import SquatModel
from models.lunge.LungeModel import LungeModel


class Home(tk.Frame):
    def __init__(self, parent, controller, camera):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.video_capture = camera
        self.camera_window = None
        self.video_capture_from_device = None
        self.camera_canvas = None
        self.current_camera_canvas = (1280, 960)
        self.camera_image = None
        self.camera_image_from_device = None
        self.is_running = True  # Flag to control webcam updating
        self.is_running_from_device = False  # Flag to control video updating

        self.plank_model = PlankModel()
        self.squat_model = SquatModel()
        self.lunge_model = LungeModel()

        # Initialize UI components here...
        # Title label
        self.label_title = tk.Label(self, text="Workout editor", font=("Arial", 46, "bold"), fg="#3C2937")
        self.label_title.pack(pady=(20, 10), padx=20, anchor="w")
        
        # Activities frame border radius
        self.activities_frame = tk.Frame(self, bg="#DFDFDF", height=100)
        self.activities_frame.pack(pady=(0, 20), padx=20, fill="both", expand=True)

        # Activities label
        label = tk.Label(self.activities_frame, text="Activities", font=("Arial", 24, "bold"), fg="#3C2937", background="#DFDFDF")
        label.pack(padx=20, anchor="w")

        self.introductions = get_introductions()

        # Activities buttons
        self.activities = ["Squat", "Lunge", "Plank", "Push up", "Bicep Curl"]
        self.activitie_selected = "Squat"
        for activity in self.activities:
            loadimage = PhotoImage(file=f'./desktop_app/assets/home/activity/buttons/{activity.lower()}.png')
            btn = tk.Button(self.activities_frame, image=loadimage, bg="#DFDFDF", activebackground='#DFDFDF', bd=0, border=0, 
                            command=lambda activity_name = activity: self.switch_activity(activity_name))
            btn.image = loadimage  # Prevent image from being garbage collected
            btn.pack(pady=10, padx=12, side="left")
            
        # Main content
        main_frame = tk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        # Introduction frame
        self.introduction_frame = tk.Frame(main_frame, height=100)
        self.introduction_frame.pack(pady=(0, 20), padx=20, fill="both", expand=True, side="left")

        # Introduction label
        label = tk.Label(self.introduction_frame, text="Introduction", font=("Arial", 24, "bold"), fg="#3C2937")
        label.pack(padx=20, anchor="w")

        # Introduction text
        text = "Select an activity to view instructions"
        self.label_introduction = tk.Label(self.introduction_frame, text=text, font=("Arial", 12), fg="#3C2937", wraplength=400, justify="left")
        self.label_introduction.pack(padx=20, anchor="w")

        # Canvas area for video from device
        self.canvas = tk.Canvas(main_frame, width=711, height=400, bg="white")
        self.canvas.pack(pady=(0, 20), padx=20, fill="both", expand=True, side="right")

        # Actions frame
        self.actions_frame = tk.Frame(self, height=100)
        self.actions_frame.pack(pady=(0, 20), padx=20, fill="both", expand=True, side="right")

        loadimage = PhotoImage(file=f'./desktop_app/assets/home/actions/StartBtn.png')
        start_btn = tk.Button(self.actions_frame, image=loadimage, bd=0, border=0, command=self.toggle_camera_window)
        start_btn.image = loadimage  # Prevent image from being garbage collected
        start_btn.pack(pady=10, padx=12, side="right")

        loadimage = PhotoImage(file=f'./desktop_app/assets/home/actions/UploadBtn.png')
        upload_btn = tk.Button(self.actions_frame, image=loadimage, bd=0, border=0, command=self.open_file_dialog)
        upload_btn.image = loadimage  # Prevent image from being garbage collected
        upload_btn.pack(pady=10, padx=12, side="right")

        # End of UI components

        self.pack()
        self.update_Webcam()

    def switch_activity(self, activity_name):
        index = 1
        for introduction in self.introductions:
            if introduction["Name"].lower() == "lunges":
                introduction["Name"] = "Lunge"
            if introduction["Name"].lower() == activity_name.lower():
                self.label_introduction.config(text=introduction["Instruction"])
                break
            index += 1

        activity_btns = self.activities_frame.winfo_children()
        arr_index = [1, 2, 3, 4, 5]
        for i in range(5):
            btn = activity_btns[arr_index[i]]
            loadimage = PhotoImage(file=f'./desktop_app/assets/home/activity/buttons/{self.activities[i].lower()}.png')
            btn.config(image=loadimage)
            btn.image = loadimage  # Prevent image from being garbage collected
        
        ind_final = 1
        if (index == 5):
            ind_final = 5
        else:
            ind_final = 5 - index
        btn = activity_btns[ind_final]
        loadimage = PhotoImage(file=f'./desktop_app/assets/home/activity/active_buttons/{activity_name.lower()}.png')
        btn.config(image=loadimage)
        btn.image = loadimage  # Prevent image from being garbage collected

        self.activitie_selected = "Other"

        if (activity_name == "Squat"):
            self.activitie_selected = "Squat"

        if (activity_name == "Plank"):
            self.activitie_selected = "Plank"

        if (activity_name == "Lunge"):
            self.activitie_selected = "Lunge"

    def update_Webcam(self):
        if not self.is_running or not self.camera_window or not self.camera_canvas:
            self.is_running_from_device = False
            return

        ret, frame = self.video_capture.read()
        if ret:
            # get camera_window frame and resize it to fit the canvas
            # frame = self.plank_model.plank_detection(frame, size_original=self.current_camera_canvas)
            switch = {
                "Squat": [self.squat_model.squat_detection_realtime, self.squat_model.init_history],
                "Plank": [self.plank_model.plank_detection_realtime, self.plank_model.init_history],
                "Lunge": [self.lunge_model.lunge_detection_realtime, self.lunge_model.init_history]
            }

            if self.activitie_selected == "Other":
                frame = cv2.resize(frame, self.current_camera_canvas)
            else:
                if self.is_running_from_device == False:
                    # switch[self.activitie_selected][1]()
                    self.is_running_from_device = True
                frame = switch[self.activitie_selected][0](frame, size_original=self.current_camera_canvas)
            
            self.camera_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.camera_canvas.create_image(0, 0, image=self.camera_image, anchor="nw")
        self.after(10, self.update_Webcam)

    def toggle_camera_window(self):
        if self.camera_window is None:
            self.camera_window = Toplevel(self)
            self.camera_window.title("Webcam")
            self.camera_canvas = tk.Canvas(self.camera_window, width=self.current_camera_canvas[0], height=self.current_camera_canvas[1], bg="white")
            self.camera_canvas.pack()
            self.camera_window.protocol("WM_DELETE_WINDOW", self.on_close_camera_window)
            self.start_canvas()  # Start updating when window is created
        else:
            self.on_close_camera_window()

    def on_close_camera_window(self):
        self.stop_canvas()
        if self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None
            self.camera_canvas = None

    def stop_canvas(self):
        self.is_running = False  # Pause updating the webcam view
        # Sử dụng hàm send_push_notification để gửi thông điệp đến thiết bị có token tương ứng
        # send_push_notification("Finish the exercise", "You have finished the exercise, please check the result")

    def start_canvas(self):
        if self.video_capture is None or not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(0)
        self.is_running = True
        self.update_Webcam()

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if file_path:
            if self.video_capture_from_device is None or not self.video_capture_from_device.isOpened():
                self.video_capture_from_device = cv2.VideoCapture(file_path)
            elif self.video_capture_from_device.isOpened():
                self.video_capture_from_device.release()
                self.video_capture_from_device = cv2.VideoCapture(file_path)
            self.play_video()
        else:
            print("No file selected")

    def play_video(self):
        if self.video_capture_from_device is None or not self.video_capture_from_device.isOpened():
            return
        ret, frame = self.video_capture_from_device.read()
        if ret:
            frame = scale_image(frame, 711, 400)
            self.camera_image_from_device = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.camera_image_from_device, anchor="nw")
        else:
            self.video_capture_from_device.release()
            self.video_capture_from_device = None
            
        self.after(10, self.play_video)

# End of Home.py

def scale_image(image, width, height):
    width_ratio = width / image.shape[1]
    height_ratio = height / image.shape[0]
    ratio = min(width_ratio, height_ratio)
    new_width = int(image.shape[1] * ratio)
    new_height = int(image.shape[0] * ratio)
    return cv2.resize(image, (new_width, new_height))