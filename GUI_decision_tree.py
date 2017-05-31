from tkinter import *
import tkinter.ttk as ttk
import tkinter.filedialog as fDialog
import pandas as pd
from collections import Counter
import decisionTree

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import arange, sin, pi
import numpy as np
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

            self.infoNumElems = 'Number of elements: '
            self.infoAccuray = 'Accuracy: '
            self.infoPrediction = 'Prediction: '
            self.infoAttrSplit = 'Variable split: '
            self.inforImpurity = 'Gini impurity: '

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

        df = pd.read_csv(file.name)
        df2 = df.get(['diesIngr', 'nIngr', 'nUrg', 'estacioAny', 'diagPrinc']) # 'diagPrinc'
        listAttr = list(df2.columns.values)
        df3 = df.get(['reingres'])
        X = df2.values.tolist()
        y = df3.values.flatten().tolist()
        dcTree = decisionTree.DecisionTree(X, y, [True, False], f=decisionTree.gini)

        self.resetFrame()
        EditTreeGUI(self.mainFrame, dcTree, listAttr)

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
    def __init__(self, master, dcTree, listAttr):
        self.mapNode = dict() # diccionary that translates the id of a node in the GUI to a node from the class decisionTree
        self.master = master
        self.dcTree = dcTree
        self.listAttr = listAttr
        # Left Frame #
        leftFrame = Frame(master)
        leftFrame.pack(side=LEFT)
        # Buttons
        b_autoSplit = Button(leftFrame, text=lg.autosplit)
        b_autoSplit.pack(side=TOP)
        b_autoSplit.bind('<Button-1>', self.autoSplit)
        b_prune = Button(leftFrame, text=lg.prune)
        b_prune.pack(side=TOP)
        b_prune.bind('<Button-1>', self.prune)
        b_join = Button(leftFrame, text=lg.join)
        b_join.pack(side=TOP)
        b_join.bind('<Button-1>', self.joinNodes)

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
        self.mapNode[tree_root_id] = dcTree
        self.gui_tree.bind('<Button-1>', self.nodeClicked)
        self.gui_tree.focus(tree_root_id)
        self.gui_tree.pack()

        # Right Frame #
        rightFrame = Frame(self.master)
        rightFrame.pack(side=RIGHT, expand=True)
        # Info
        infoFrame = Frame(rightFrame)
        infoFrame.pack(side=BOTTOM)
        self.infoNumber = Label(infoFrame, text=lg.infoNumElems)
        self.infoNumber.grid(row=0, column=0, sticky=W)
        self.infoAccuracy = Label(infoFrame, text=lg.infoAccuray)
        self.infoAccuracy.grid(row=0, column=1, sticky=W)
        self.infoGini = Label(infoFrame, text=lg.inforImpurity)
        self.infoGini.grid(row=1, column=0, sticky=W)
        self.infoAttrSplit = Label(infoFrame, text=lg.infoAttrSplit)
        self.infoAttrSplit.grid(row=1, column=1, sticky=W)
        self.infoPrediction = Label(infoFrame, text=lg.infoPrediction)

        # Attr split frame
        attrFrame = Frame(rightFrame)
        attrFrame.pack(side=TOP)
        self.tkvar = StringVar(root)
        self.tkvar.set(self.listAttr[0]) # set the default option
        self.tkvar.trace('w', self.optionMenuClicked)
        self.popupMenu = OptionMenu(attrFrame, self.tkvar, *self.listAttr)
        self.popupMenu.pack()

        # prova grafic
        self.figure = Figure(figsize=(5, 4), dpi=100)
        subPlot = self.figure.add_subplot(111)
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)

        subPlot.plot(t, s)

        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(self.figure, master=rightFrame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    def optionMenuClicked(self, *args):
        selectedAttr = self.tkvar.get()
        print(selectedAttr)
        node = self.mapNode[self.gui_tree.focus()]
        # node = decisionTree.DecisionTree()
        segData = node.getSegmentedData(self.listAttr.index(selectedAttr))
        self.figure.clear()
        subPlot = self.figure.add_subplot(111)
        # comprovar que les dades no sigui buides
        if type(segData[0][0]) == int or type(segData[0][0]) == float:
            auxHist = [0] * 30
            for data in segData:
                h = subPlot.hist(data, bins=30, bottom=auxHist)
                auxHist += h[0]
        else:
            s = set()
            for data in segData:
                s = s.union(set(data))
            s = sorted((list(s)))
            x = list(range(len(s)))
            acumY = np.array([0] * len(x))
            for data in segData:
                aux_y = Counter(data)
                y = [0] * len(x)
                for (i, elem) in enumerate(s):
                    if elem in aux_y:
                        y[i] = aux_y[elem]
                subPlot.bar(x, y, bottom=acumY)
                acumY += y

        self.canvas.show()

    def nodeClicked(self, event):
        # print(self.gui_tree.focus())
        dcTree = self.mapNode[self.gui_tree.focus()]
        # dcTree = decisionTree.DecisionTree()
        # print(dcTree)
        self.infoNumber['text'] = lg.infoNumElems + str(dcTree.getNumElems())
        self.infoPrediction['text'] = lg.infoPrediction + str(dcTree.getPrediction())
        self.infoAccuracy['text'] = lg.infoAccuray + str(dcTree.getAccuracy())
        self.infoAttrSplit['text'] = lg.infoAttrSplit + str(dcTree.attrSplit)
        self.infoGini['text'] = lg.inforImpurity + str(dcTree.getImpurity())

    def autoSplit(self, event):
        print('Autospliting')
        dcTree = self.mapNode[self.gui_tree.focus()]
        dcTree.autoSplit(minSetSize=1000, giniReduction=0.01)
        print('Autosplit completed')
        self.addNodes(self.gui_tree.focus(), dcTree)
        pass

    def joinNodes(self, event):
        print('joinning')
        setNodes = self.gui_tree.selection()
        if len(setNodes) >= 2:
            parent = self.gui_tree.parent(setNodes[0])
            if all((self.gui_tree.parent(node) == parent for node in setNodes)):
                dcTree = self.mapNode[parent]
                joinedNode = dcTree.joinNodes([self.gui_tree.index(son) for son in setNodes])
                for son in setNodes:
                    self.mapNode.pop(son)
                    self.gui_tree.delete(son)
                idJoinedNode = self.gui_tree.insert(parent, 'end', text='joined')
                self.mapNode[idJoinedNode] = joinedNode

    def prune(self, event):
        print('prunning')
        nodeGUI = self.gui_tree.focus()
        dcTree = self.mapNode[nodeGUI]
        dcTree.prune()
        for node in self.gui_tree.get_children(nodeGUI):
            self.mapNode.pop(node)
            self.gui_tree.delete(node)
        pass

    def addNodes(self, rootGUI, rootDT):
        # rootDT = decisionTree.DecisionTree()
        for (i, son) in enumerate(rootDT.getSons()):
            idSon = self.gui_tree.insert(rootGUI, 'end', text=str(i))
            self.mapNode[idSon] = son
            self.addNodes(idSon, son)

class PredictGUI:
    def __init__(self, master):
        self.master = master
        label = Label(self.master, text='hola')
        label.pack()

# Important variables #

lg = Language()
root = Tk()
menu = MyMenu(root)

# editTree = EditTree(root)

root.mainloop()


