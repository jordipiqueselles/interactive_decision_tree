from tkinter import *
import tkinter.ttk as ttk
import tkinter.filedialog as fDialog
import pandas as pd
import decisionTree

import matplotlib
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler

from matplotlib.figure import Figure

def deleteAllWidgets(master):
    master = Tk()


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

            self.file = 'File'
            self.newTree = 'New Decision Tree'
            self.editTree = 'Edit Decision Tree'
            self.newPrediction = 'New prediction'
            self.saveTree = 'Save Decision Tree'
            self.loadTree = 'Load Decision Tree'
            self.loadDataSet = 'Load DataSet'

        elif language == Language.spanish:
            pass
        elif language == Language.catalan:
            pass
        
class MyMenu:
    def __init__(self, master):
        # Menu #
        self.master = master
        self.menu = Menu(self.master)
        self.mFile = Menu(self.menu)
        self.mFile.add_command(label=lg.newTree, command=self.newTree)
        self.mFile.add_command(label=lg.editTree, command=self.editTree)
        self.mFile.add_command(label=lg.newPrediction, command=self.newPrediction)
        self.mFile.add_separator()
        self.mFile.add_command(label=lg.loadTree, command=self.loadTree)
        self.mFile.add_command(label=lg.saveTree, command=self.saveTree)
        self.mFile.add_command(label=lg.loadDataSet, command=self.loadDataSet)
        self.menu.add_cascade(label=lg.file, menu=self.mFile)
        self.master.config(menu=self.menu)

        self.mainFrame = Frame(self.master)

    def resetFrame(self):
        self.mainFrame.destroy()
        self.mainFrame = Frame(master=self.master)
        self.mainFrame.pack()

    def newTree(self):
        file = fDialog.askopenfile(mode='r')
        print(file)
        print(file.name)

        df = pd.read_csv('dadesSantPauProc.csv')
        df2 = df.get(['diesIngr', 'nIngr', 'nUrg', 'estacioAny', 'diagPrinc']) # 'diagPrinc'
        df3 = df.get(['reingres'])
        aux2 = df2.values.tolist()
        aux3 = df3.values.flatten().tolist()
        dcTree = decisionTree.DecisionTree(aux2, aux3, [True, False], f=decisionTree.gini)
        dcTree.splitNode(4)

        print(dcTree)
        # self.resetFrame()
        # EditTreeGUI(self.mainFrame)

    def editTree(self):
        pass

    def newPrediction(self):
        self.resetFrame()
        PredictGUI(self.mainFrame)

    def saveTree(self):
        pass

    def loadTree(self):
        pass

    def loadDataSet(self):
        pass

class EditTreeGUI:
    def __init__(self, master):
        self.master = master
        # Left Frame #
        leftFrame = Frame(master)
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
        centralFrame = Frame(self.master)
        centralFrame.pack(side=LEFT)
        # Tree
        self.gui_tree = ttk.Treeview(centralFrame)
        # id1 = gui_tree.insert('', 'end', text='Tutorial1')
        # id2 = gui_tree.insert('', 'end', text='Tutorial2')
        # id3 = gui_tree.insert(id2, 'end', text='Tutorial3')
        tree_root_id = self.gui_tree.insert('', 'end', text='Root')
        self.gui_tree.bind('<Button-1>', self.nodeClicked)
        self.gui_tree.pack()

        # Right Frame #
        rightFrame = Frame(self.master)
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

    def nodeClicked(self, event):
        print(self.gui_tree.focus())

class PredictGUI:
    def __init__(self, master):
        self.master = master
        label = Label(self.master, text='hola')
        label.pack()

# Important variables #
mapNode = dict() # diccionary that translates the id of a node in the GUI to a node from the class decisionTree

lg = Language()
root = Tk()
menu = MyMenu(root)

# editTree = EditTree(root)

root.mainloop()


