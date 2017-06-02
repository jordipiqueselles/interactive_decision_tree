from tkinter import *
import tkinter.ttk as ttk
import tkinter.filedialog as fDialog
import tkinter.messagebox as tkMessageBox
import pandas as pd
from collections import Counter
import decisionTree
from abc import ABC
import functools
import math

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from numpy import arange, sin, pi
import numpy as np
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
            self.setEnglish()
        elif language == Language.spanish:
            self.setSpanish()
        elif language == Language.catalan:
            self.setCatalan()

    def setEnglish(self):
        self.autosplit = 'Autosplit'
        self.prune = 'Prune'
        self.join = 'Join'
        self.split = 'Split'
        self.bestSplit = 'Best split'
        self.adOptions = 'Advanced options'
        self.validate = 'Validate'
        self.varSplit = 'Variable split'
        self.naiveBayes = 'Naive Bayes'

        self.infoNumElems = 'Number of elements: '
        self.infoAccuray = 'Accuracy: '
        self.infoPrediction = 'Prediction: '
        self.infoAttrSplit = 'Variable split: '
        self.inforImpurity = 'Gini impurity: '
        self.accuracy = 'Accuracy: '

        self.file = 'File'
        self.newTree = 'New Decision Tree'
        self.editTree = 'Edit Decision Tree'
        self.newPrediction = 'New prediction'
        self.saveTree = 'Save Decision Tree'

    def setSpanish(self):
        pass

    def setCatalan(self):
        pass
        
class MyMenu:
    def __init__(self, master):
        """
        :param master: The root Tk window where the menu will be inserted
        Fills the master frame with a menu. This menu can give access to the main functionalities of the application
        as creating a decision tree or using one to predict
        """
        self.master = master
        self.menu = Menu(self.master)
        self.mFile = Menu(self.menu)
        self.mFile.add_command(label=lg.newTree, command=self.newTree)
        self.mFile.add_command(label=lg.newPrediction, command=self.newPrediction)
        self.mFile.add_separator()
        self.mFile.add_command(label=lg.editTree, command=self.editTree)
        self.mFile.add_command(label=lg.saveTree, command=self.saveTree)
        self.menu.add_cascade(label=lg.file, menu=self.mFile)
        self.master.config(menu=self.menu)

        self.mainFrame = Frame(self.master)
        self.currentView = None

    def resetFrame(self):
        """
        Destroy the main frame and all the widgets that contains. It is used to before showing a different view in
        the same window
        """
        self.mainFrame.destroy()
        self.mainFrame = Frame(master=self.master)
        self.mainFrame.pack()

    def newTree(self):
        """
        Shows a dialog window to choose a file and it creates a view for building a decision tree from the data of the
        selected file
        """
        file = fDialog.askopenfile(mode='r')
        df = pd.read_csv(file.name).sample(frac=1)

        # TODO Aquesta part s'ha de fer general per a qualsevol tipus de DataSet
        df2 = df.get(['diesIngr', 'nIngr', 'nUrg', 'estacioAny', 'diagPrinc']) # 'diagPrinc'
        df3 = df.get(['reingres'])
        X = df2.values.tolist()
        y = df3.values.flatten().tolist()
        nTrain = round(0.7 * len(y))

        dcTree = decisionTree.DecisionTree(X[:nTrain], y[:nTrain], [True, False], f=decisionTree.gini, attrNames=list(df2.columns))
        self.resetFrame()
        self.currentView = EditTreeGUI(self.mainFrame, dcTree, X[nTrain:], y[nTrain:])

    def newPrediction(self):
        file = fDialog.askopenfile(mode='r')
        dcTree = decisionTree.DecisionTree.load(file.name)
        self.resetFrame()
        self.currentView = PredictGUI(self.mainFrame, dcTree)

    def saveTree(self):
        if type(self.currentView) == EditTreeGUI:
            file = fDialog.asksaveasfile(mode='w')
            self.currentView.saveDcTree(file.name)


    def editTree(self):
        file = fDialog.askopenfile(mode='r')
        dcTree = decisionTree.DecisionTree.load(file.name)
        self.resetFrame()
        self.currentView = EditTreeGUI(self.mainFrame, dcTree)


class TreeFrame(ABC):
    # tree.tag_configure('ttk', background='yellow')
    # tree.tag_bind('ttk', '<1>', itemClicked)
    keyImpurity = "impurity"
    keyPrediction = "prediction"
    keyAttrSplit = "attrSplit"
    def __init__(self, master, dcTree, parent, packSide=BOTTOM):
        self.dcTree = dcTree
        self.master = master
        self.parent = parent
        self.mapNode = dict() # diccionary that translates the id of a node in the GUI to a node from the class decisionTree

        self.gui_tree = ttk.Treeview(master)
        self.gui_tree["columns"] = (TreeFrame.keyImpurity, TreeFrame.keyPrediction, TreeFrame.keyAttrSplit)
        self.gui_tree.column(TreeFrame.keyImpurity, width=100)
        self.gui_tree.column(TreeFrame.keyPrediction, width=100)
        self.gui_tree.column(TreeFrame.keyAttrSplit, width=100)
        self.gui_tree.heading(TreeFrame.keyImpurity, text=lg.inforImpurity)
        self.gui_tree.heading(TreeFrame.keyPrediction, text=lg.infoPrediction)
        self.gui_tree.heading(TreeFrame.keyAttrSplit, text=lg.infoAttrSplit)

        tree_root_id = self.gui_tree.insert('', 'end', text=str(self.dcTree.getNumElems()),
                                         values=(str(self.dcTree.getImpurity()), str(self.dcTree.getPrediction()),
                                                 str(self.dcTree.getAttrSplit())))
        self.mapNode[tree_root_id] = dcTree
        self.addNodes(tree_root_id, dcTree)

        self.gui_tree.bind('<Button-1>', self.nodeClicked)
        self.gui_tree.focus(tree_root_id)
        self.gui_tree.pack(side=packSide)

    def addNodes(self, rootGUI, rootDT):
        self.gui_tree.set(rootGUI, TreeFrame.keyAttrSplit, str(rootDT.getAttrSplit()))
        for (i, son) in enumerate(rootDT.getSons()):
            idSon = self.gui_tree.insert(rootGUI, 'end', text=str(son.getNumElems()),
                                         values=(str(son.getImpurity()), str(son.getPrediction()), str(son.getAttrSplit())))
            self.mapNode[idSon] = son
            self.addNodes(idSon, son)

    def nodeClicked(self, event):
        pass

    def predict_cv(self, X, naiveBayes):
        return self.dcTree.predict(X, naiveBayes)


class TreeFrameEdit(TreeFrame):
    def getSegData(self, selectedAttr):
        node = self.mapNode[self.gui_tree.focus()]
        segData = node.getSegmentedData(self.dcTree.attrNames.index(selectedAttr))
        return segData

    def nodeClicked(self, event):
        dcTree = self.mapNode[self.gui_tree.focus()]

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
                idJoinedNode = self.gui_tree.insert(parent, 'end', text=str(joinedNode.getNumElems()),
                                         values=(str(joinedNode.getImpurity()), str(joinedNode.getPrediction()),
                                                 str(joinedNode.getAttrSplit())))
                self.mapNode[idJoinedNode] = joinedNode

    def prune(self, event):
        print('prunning')
        nodeGUI = self.gui_tree.focus()
        dcTree = self.mapNode[nodeGUI]
        self.gui_tree.set(nodeGUI, TreeFrame.keyAttrSplit, str(dcTree.getAttrSplit()))
        dcTree.prune()
        for node in self.gui_tree.get_children(nodeGUI):
            self.mapNode.pop(node)
            self.gui_tree.delete(node)

    def split(self, idxAttr):
        nodeGUI = self.gui_tree.focus()
        dcTree = self.mapNode[nodeGUI]
        dcTree.splitNode(idxAttr)
        self.addNodes(nodeGUI, dcTree)

    def bestSplit(self):
        nodeGUI = self.gui_tree.focus()
        dcTree = self.mapNode[nodeGUI]
        return dcTree.bestSplit()

    def nodeClicked(self, event):
        self.parent.changePlot()


class TreeFramePredict(TreeFrame):
    pass

class EditTreeGUI:
    def __init__(self, master, dcTree, X_cv, y_cv):
        """
        :param master:
        :param dcTree:
        :return:
        """
        self.master = master
        self.dcTree = dcTree
        self.X_cv = X_cv
        self.y_cv = y_cv

        # Central Frame #
        centralFrame = Frame(self.master)
        centralFrame.pack(side=LEFT)
        # Tree
        self.treeFrame = TreeFrameEdit(centralFrame, self.dcTree, self)

        # Buttons Frame #
        buttonsFrame = Frame(centralFrame)
        buttonsFrame.pack(side=TOP)
        # Buttons
        # Button validate
        b_validate = Button(buttonsFrame, text=lg.validate)
        b_validate.grid(row=0, column=0)
        b_validate.bind('<Button-1>', self.predict_cv)
        # Button advanced options
        b_adOptions = Button(buttonsFrame, text=lg.adOptions)
        b_adOptions.grid(row=1, column=0)
        b_adOptions.bind('<Button-1>', None)
        # Button prune
        b_prune = Button(buttonsFrame, text=lg.prune)
        b_prune.grid(row=0, column=1)
        b_prune.bind('<Button-1>', self.treeFrame.prune)
        # Button join
        b_join = Button(buttonsFrame, text=lg.join)
        b_join.grid(row=1, column=1)
        b_join.bind('<Button-1>', self.treeFrame.joinNodes)
        # Button autosplit
        b_autoSplit = Button(buttonsFrame, text=lg.autosplit)
        b_autoSplit.grid(row=0, column=2)
        b_autoSplit.bind('<Button-1>', self.treeFrame.autoSplit)
        # Button best split
        b_bestSplit = Button(buttonsFrame, text=lg.bestSplit)
        b_bestSplit.grid(row=1, column=2)
        b_bestSplit.bind('<Button-1>', self.bestSplit)
        # Option Menu
        self.tkvar = StringVar(root)
        self.tkvar.set(self.dcTree.attrNames[0]) # set the default option
        self.tkvar.trace('w', self.optionMenuClicked)
        self.popupMenu = OptionMenu(buttonsFrame, self.tkvar, *self.dcTree.attrNames)
        self.popupMenu.grid(row=0, column=3)
        # Button best split
        b_split = Button(buttonsFrame, text=lg.split)
        b_split.grid(row=1, column=3)
        b_split.bind('<Button-1>', self.split)


        # cb_NaiBay = Checkbutton(buttonsFrame, text=lg.naiveBayes)
        # cb_NaiBay.pack(side=BOTTOM)

        # Right Frame #
        rightFrame = Frame(self.master)
        rightFrame.pack(side=RIGHT, expand=True)
        # prova grafic
        self.figure = Figure(figsize=(7, 6), dpi=100)
        # self.figure.add_axes()
        subPlot = self.figure.add_subplot(111)
        t = arange(0.0, 3.0, 0.01)
        s = sin(2*pi*t)

        subPlot.plot(t, s)

        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(self.figure, master=rightFrame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    def split(self, event):
        selectedAttr = self.tkvar.get()
        idxAttr = self.dcTree.attrNames.index(selectedAttr)
        self.treeFrame.split(idxAttr)

    def bestSplit(self, event):
        listSplits = self.treeFrame.bestSplit()
        listSplits = [(gImp, self.dcTree.attrNames[i]) for (gImp, i) in listSplits]
        InfoBestSplits(listSplits)

    def predict_cv(self, event):
        pred_y = self.treeFrame.predict_cv(self.X_cv, False)
        accuracy = sum([elem[0] == elem[1] for elem in zip(self.y_cv, pred_y)]) / len(self.y_cv)
        tkMessageBox.showinfo('', lg.accuracy + str(round(accuracy, 4)))

    def optionMenuClicked(self, *args):
        self.changePlot()

    def changePlot(self):
        selectedAttr = self.tkvar.get()
        segData = self.treeFrame.getSegData(selectedAttr)
        self.figure.clear()
        subPlot = self.figure.add_subplot(111)
        axes = self.figure.add_axes()
        # TODO Una mica lleig, intentar compactar el codi i fer-lo mes clar
        if type(segData[0][0]) == int or type(segData[0][0]) == float:
            auxHist = [0] * 30
            for (i, data) in enumerate(segData):
                h = subPlot.hist(data, bins=30, bottom=auxHist, label=str(self.dcTree.classes[i]))
                auxHist += h[0]
        else:
            s = set()
            for data in segData:
                s = s.union(set(data))
            s = sorted((list(s)))
            x = list(range(len(s)))
            acumY = np.array([0] * len(x))
            for (i, data) in enumerate(segData):
                aux_y = Counter(data)
                y = [0] * len(x)
                for (j, elem) in enumerate(s):
                    if elem in aux_y:
                        y[j] = aux_y[elem]
                subPlot.bar(x, y, bottom=acumY, label=str(self.dcTree.classes[i]))
                acumY += y
            self.figure.axes[0].set_xticks(x)
            self.figure.axes[0].set_xticklabels(s)

        subPlot.legend()
        self.canvas.show()

    def saveDcTree(self, file):
        self.dcTree.save(file)

class InfoBestSplits:
    def __init__(self, listSplits):
        root = Tk()
        root.title(lg.bestSplit)
        for (i, (gImp, attr)) in enumerate(listSplits):
            labelAttr = Label(root, text=attr, borderwidth=1)
            labelAttr.grid(row=i, column=0)
            labelGImp = Label(root, text=str(round(gImp, 4)), borderwidth=1)
            labelGImp.grid(row=i, column=1)
        root.mainloop()

class PredictGUI:
    def __init__(self, master, dcTree):
        self.master = master
        # Central Frame #
        centralFrame = Frame(self.master)
        centralFrame.pack(side=LEFT)
        # Tree
        self.dcTree = dcTree
        self.treeFrame = TreeFramePredict(centralFrame, dcTree, self)

# Important variables #

lg = Language()
root = Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
menu = MyMenu(root)

# editTree = EditTree(root)

root.mainloop()


