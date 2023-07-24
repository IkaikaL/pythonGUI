# import everything from tkinter module
import tkinter
from tkinter import *
from tkinter import ttk
import os
import subprocess
import time
import threading


class Main_Frame(object):
    def __init__(self, window=None):
        # save root reference
        self.window = window


        # start button calls the "initialization" function bar_init, you can pass a variable in here if desired
        # self.start_button = ttk.Button(rightFrame, text='Start bar', command=lambda: self.bar_init(2500))
        # self.start_button.pack()

        create_label(frame=window,
                     text="Step 1. Move/Create a \"Masked-All\" folder \n inside the current directory", padding_y=10,
                     padding_x=20)

        create_label(frame=window, text='Input the image number you wish\n to use to train the model with', padding_y=(20, 0), padding_x=0)

        global scanNumberEntry
        scanNumberEntry = Entry(window, width=10)
        scanNumberEntry.focus_set()
        scanNumberEntry.pack()

        create_label(window, text="Step 2. Run GMM", padding_y=10, padding_x=0)


        create_label(window, text="Step 3. Run Supervised Model", padding_y=10, padding_x=0)


        create_label(window, text="Step 4. Predict all Monsif", padding_y=(10, 0), padding_x=0)


        create_label(window, text="Status Bar", padding_y=(20 ,0), padding_x=0)
        # the progress bar will be referenced in the "bar handling" and "work" threads
        self.load_bar = ttk.Progressbar(window)
        self.load_bar.pack(pady=(10), padx=10)
        self.load_gmm_button = ttk.Button(window, text="Run Program", width=40,
                                          command=lambda:  self.bar_init(var=2500))
        self.load_gmm_button.pack(pady=(10, 20),padx=10)
        # run mainloop
        self.window.mainloop()

    def bar_init(self, var):
        # first layer of isolation, note var being passed along to the self.start_bar function
        # target is the function being started on a new thread, so the "bar handler" thread
        self.start_bar_thread = threading.Thread(target=self.start_bar, args=(var, ))
        # start the bar handling thread
        self.start_bar_thread.start()

    def start_bar(self, var):
        # the load_bar needs to be configured for indeterminate amount of bouncing
        self.load_bar.config(mode='indeterminate', maximum=500, value=0)
        # 8 here is for speed of bounce
        self.load_bar.start(8)
        # start the work-intensive thread, again a var can be passed in here too if desired
        self.work_thread = threading.Thread(target=self.work_task, args=(var,))
        self.work_thread.start()
        # close the work thread
        self.work_thread.join()
        # stop the indeterminate bouncing
        self.load_bar.stop()
        # reconfigure the bar so it appears reset
        self.load_bar.config(value=0, maximum=0)

    def work_task(self, wait_time):
        scanNumber = scanNumberEntry.get()
        output = subprocess.run(['python', 'GMM-final_log.py', str(scanNumber)], stdout=subprocess.PIPE)
        print(output.stdout.decode('utf-8'))
        if output.stdout.decode('utf-8') != "":
            self.load_bar.stop()

        self.load_bar.start()
        output = subprocess.run('python RF_Supervised_model.py', stdout=subprocess.PIPE)
        print(output.stdout.decode('utf-8'))
        if output.stdout.decode('utf-8') != "":
            self.load_bar.stop()

        self.load_bar.start()
        output = subprocess.run('python RF_predict-all_monsif-7amo.py', stdout=subprocess.PIPE)
        print(output.stdout.decode('utf-8'))
        if output.stdout.decode('utf-8') != "":
            self.load_bar.stop()

        self.load_bar.stop()


def create_label(frame, text, padding_y, padding_x):
    ttk.Label(frame, text=text, font="Courier 14 bold").pack(pady=padding_y, padx=padding_x)


window = Tk()


window.geometry('500x450')
leftFrame = Frame(window)
leftFrame.pack(side=LEFT, anchor='center', fill='both')

scanNumber = 0
Main_Frame(window=window)

window.mainloop()
