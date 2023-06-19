# import everything from tkinter module
import tkinter
from tkinter import *
from tkinter import ttk
import os


def load_gmm():
    # Location of masked images from matlab code
    global fileEntry
    # Output
    global savePathEntry
    fileEntryString = fileEntry.get()
    savePathEntryString = savePathEntry.get()
    return tkinter.Button(os.system('python GMM-final_log.py ' + fileEntryString + ' ' + savePathEntryString))

def load_supervised_model():
    # File Output by GMM
    global gmmOutputEntry
    gmmOutputEntryString = gmmOutputEntry.get()
    return tkinter.Button(os.system('python RF_Supervised_model.py ' + gmmOutputEntryString))

def load_supervised_model_output():
    # Location of supervised model output
    global supervisedModelLocationEntry
    # Locatin of masked files
    global maskedFilesEntry
    # Location of segmented files
    global segmentedFilesEntry
    # Location of Core300um files (list)
    global core300UmFilesEntry
    supervisedModelLocationEntryString = supervisedModelLocationEntry.get()
    maskedFilesEntryString = maskedFilesEntry.get()
    segmentedFilesEntryString = segmentedFilesEntry.get()
    core300UmFilesEntryString = core300UmFilesEntry.get()
    return tkinter.Button(os.system('python RF_predict-all_monsif-7amo.py ' + supervisedModelLocationEntryString + ' ' + maskedFilesEntryString + ' ' + segmentedFilesEntryString + ' ' + core300UmFilesEntryString))

# create a tkinter window

window = Tk()


window.geometry('1300x300')
leftFrame = Frame(window)
leftFrame.pack(side=LEFT, anchor='center', fill='both')

rightFrame = Frame(window)
rightFrame.pack(side=RIGHT, anchor='center', fill='both')

# Initialize a Label to display the User Input

label = Label(leftFrame, text="Input the File Location of Masked Images for GMM", font="Courier 14 bold")

label.pack()


# Create an Entry widget to accept User Input

fileEntry = Entry(leftFrame, width=40)
fileEntry.focus_set()
fileEntry.pack()

# Initialize a Label to display the User Input

label = Label(leftFrame, text="Input Save Location for GMM Output", font="Courier 14 bold")
label.pack()
# Create an Entry widget to accept User Input

savePathEntry = Entry(leftFrame, width=40)
savePathEntry.focus_set()
savePathEntry.pack()

# Create a Button to load file path

ttk.Button(leftFrame, text="Run GMM", width=20, command=load_gmm).pack(pady=20)

# Initialize a Label to display the User Input

label = Label(window, text="Input GMM File Output", font="Courier 14 bold")
label.pack(pady=5)
# Create an Entry widget to accept User Input

gmmOutputEntry = Entry(window, width=40)
gmmOutputEntry.focus_set()
gmmOutputEntry.pack()

# Create a Button to load file path

ttk.Button(window, text="Run RF Supervised Model", width=20, command=load_supervised_model).pack(pady=20)

# Initialize a Label to display the User Input

label = Label(rightFrame, text="Input Location of Supervised Model Output", font="Courier 14 bold")
label.pack()
# Create an Entry widget to accept User Input

supervisedModelLocationEntry = Entry(rightFrame, width=40)
supervisedModelLocationEntry.focus_set()
supervisedModelLocationEntry.pack()

# Initialize a Label to display the User Input

label = Label(rightFrame, text="Input Location of Masked Files", font="Courier 14 bold")
label.pack()
# Create an Entry widget to accept User Input

maskedFilesEntry = Entry(rightFrame, width=40)
maskedFilesEntry.focus_set()
maskedFilesEntry.pack()

# Initialize a Label to display the User Input

label = Label(rightFrame, text="Input Location of Segmented Files", font="Courier 14 bold")
label.pack()
# Create an Entry widget to accept User Input

segmentedFilesEntry = Entry(rightFrame, width=40)
segmentedFilesEntry.focus_set()
segmentedFilesEntry.pack()

# Initialize a Label to display the User Input

label = Label(rightFrame, text="Input Location of Core 300um Files", font="Courier 14 bold")
label.pack()
# Create an Entry widget to accept User Input

core300UmFilesEntry = Entry(rightFrame, width=40)
core300UmFilesEntry.focus_set()
core300UmFilesEntry.pack()

# Create a Button to load file path

ttk.Button(rightFrame, text="Run RF Predict", width=20, command=load_supervised_model_output).pack(pady=20)


window.mainloop()
