# import everything from tkinter module
import tkinter
from tkinter import *
from tkinter import ttk
import os


def load_gmm():
    return tkinter.Button(os.system('python GMM-final_log.py'))

def load_supervised_model():
    return tkinter.Button(os.system('python RF_Supervised_model.py'))

def load_supervised_model_output():
    return tkinter.Button(os.system('python RF_predict-all_monsif-7amo.py'))

# create a tkinter window

window = Tk()


window.geometry('1300x300')
leftFrame = Frame(window)
leftFrame.pack(side=LEFT, anchor='center', fill='both')

rightFrame = Frame(window)
rightFrame.pack(side=RIGHT, anchor='center', fill='both')

# Initialize a Label to display the User Input

label = Label(leftFrame, text="Step 1. Create Masked-All folder inside directory", font="Courier 14 bold")

label.pack()

# Initialize a Label to display the User Input

label = Label(leftFrame, text="Step 2. Run GMM", font="Courier 14 bold")
label.pack()

# Create a Button to load file path

ttk.Button(leftFrame, text="Run GMM", width=20, command=load_gmm).pack(pady=20)

# Initialize a Label to display the User Input

label = Label(window, text="Run Supervised Model", font="Courier 14 bold")
label.pack(pady=5)

# Create a Button to load file path

ttk.Button(window, text="Run RF Supervised Model", width=20, command=load_supervised_model).pack(pady=20)

# Initialize a Label to display the User Input

label = Label(rightFrame, text="Predict all Monsif", font="Courier 14 bold")
label.pack()
# Create an Entry widget to accept User Input


# Create a Button to load file path

ttk.Button(rightFrame, text="Run RF Predict", width=20, command=load_supervised_model_output).pack(pady=20)


window.mainloop()
