# Code is based on examples from https://realpython.com/pysimplegui-python/

# Additional info about PySimpleGUI
# https://pysimplegui.readthedocs.io/en/latest/cookbook/
import PySimpleGUI as sg
import cv2
import numpy as np

def main():
    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        [sg.Image(filename="", key="-IMAGE-")],
        [
            sg.Button("LOG", size=(10, 1)),
            sg.Slider(
                (-10, 10),
                0,
                0.1,
                orientation="h",
                size=(40, 10),
                key="-LOG SLIDER-",
            ),
            sg.Button("GAMMA", size=(10, 1)),
            sg.Slider(
                (0, 25),
                1,
                0.1,
                orientation="h",
                size=(40, 10),
                key="-GAMMA SLIDER-",
            ),
            
        ],  
        [
            sg.Button("AVERAGE", size=(10, 1)),
            sg.Slider(
                (1, 21),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-BLUR SLIDER-",
            ),
            sg.Button("MEDIAN", size=(10, 1)),
            sg.Slider(
                (1, 21),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-MEDIAN SLIDER-",
            ),
            
        ],
        [
            sg.Button("HSV_THS", size=(10, 1)),
            sg.Text('H mid'),
            sg.Slider(
                (0, 360),
                180,
                1,
                orientation="h",
                size=(15, 10),
                key="-HSV SLIDER Hth-",
            ),
            sg.Text('H range'),
            sg.Slider(
                (0, 255),
                50,
                1,
                orientation="h",
                size=(15, 10),
                key="-HSV SLIDER Hr-",
            ),
            sg.Text('S Low'),
            sg.Slider(
                (0, 100),
                50,
                1,
                orientation="h",
                size=(10, 10),
                key="-HSV SLIDER S LOW-",
            ),
            sg.Text('S High'),
            sg.Slider(
                (0, 100),
                55,
                1,
                orientation="h",
                size=(10, 10),
                key="-HSV SLIDER S HIGH-",
            ),
            sg.Text('V Low'),
            sg.Slider(
                (0, 100),
                50,
                1,
                orientation="h",
                size=(10, 10),
                key="-HSV SLIDER V LOW-",
            ),
            sg.Text('V High'),
            sg.Slider(
                (0, 100),
                55,
                1,
                orientation="h",
                size=(10, 10),
                key="-HSV SLIDER V HIGH-",
            ),
        ],
        [
            sg.Button("ERODE", size=(10, 1)),
            sg.Slider(
                (1, 15),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-ERODE SLIDER-",
            ),
            sg.Button("DILATE", size=(10, 1)),
            sg.Slider(
                (1, 15),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-DILATE SLIDER-",
            ),
            
        ],
        [sg.Button("Reset_RGB", size=(10, 1)),sg.Button("Reset_BW", size=(10, 1)),sg.Button("Histogram", size=(10, 1)),sg.Button("Exit", size=(10, 1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("GUI Example", layout, location=(800, 400))

    img = cv2.imread('test1.jpg')
    #M, N = img.shape
    bw_image = img.copy()
    img_tmp = img.copy()
    
    frame = np.concatenate((img_tmp, bw_image), axis=1)

    while True:
        event, values = window.read(timeout=200)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "AVERAGE":
            b_val = int(values["-BLUR SLIDER-"])
            if (b_val % 2) == 0:
                b_val = b_val+1
            img_tmp = cv2.blur(img_tmp, (b_val, b_val), )
            frame = np.concatenate((img_tmp, bw_image), axis=1)
            print('average')            
        elif event == "HSV_THS":
            # fallowing code just an example
            ret,thresh1 = cv2.threshold(img_tmp[:,:,2],int(values["-HSV SLIDER Hth-"]),255,cv2.THRESH_BINARY_INV)
            frame_rgb = cv2.cvtColor(thresh1,cv2.COLOR_GRAY2RGB)
            frame = np.concatenate((img_tmp, frame_rgb.copy()), axis=1)


        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

    window.close()

main()