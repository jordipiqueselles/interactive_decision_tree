from tkinter import *
import tkinter.ttk as ttk

import matplotlib
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler

from matplotlib.figure import Figure


class Language:
    english = 'english'
    spanish = 'spanish'
    catalan = 'catalan'
    def __init__(self, language=english):
        if language == Language.english:
            self.autosplit = 'Autosplit'
            self.prune = 'Prune'
            self.join = 'Join'
            self.varSplit = 'Variable split'
            self.naiveBayes = 'Naive Bayes'
        elif language == Language.spanish:
            pass
        elif language == Language.catalan:
            pass

lg = Language()
root = Tk()
# Left Frame #
leftFrame = Frame(root)
leftFrame.pack(side=LEFT)
# Buttons
b_autoSplit = Button(leftFrame, text=lg.autosplit)
b_autoSplit.pack(side=TOP)
b_prune = Button(leftFrame, text=lg.prune)
b_prune.pack(side=TOP)
b_join = Button(leftFrame, text=lg.join)
b_join.pack(side=TOP)

cb_NaiBay = Checkbutton(leftFrame, text=lg.naiveBayes)
cb_NaiBay.pack(side=BOTTOM)

# Central Frame #
centralFrame = Frame(root)
centralFrame.pack(side=LEFT)
# Tree
gui_tree = ttk.Treeview(centralFrame)
id1 = gui_tree.insert('', 'end', text='Tutorial1')
id2 = gui_tree.insert('', 'end', text='Tutorial2')
id3 = gui_tree.insert(id2, 'end', text='Tutorial3')
gui_tree.pack()

# Right Frame #
rightFrame = Frame(root)
rightFrame.pack(side=RIGHT)
# Info
infoFrame = Frame(rightFrame)
infoFrame.pack(side=BOTTOM)
infoNumber = Label(infoFrame, text='infoNumber')
infoNumber.grid(row=0, column=0)
infoAccuracy = Label(infoFrame, text='infoAccuracy')
infoAccuracy.grid(row=0, column=1)

# prova grafic
f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0, 3.0, 0.01)
s = sin(2*pi*t)

a.plot(t, s)

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=rightFrame)
canvas.show()
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

root.mainloop()


