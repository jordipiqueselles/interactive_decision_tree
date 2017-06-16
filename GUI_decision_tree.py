import pickle
import tkinter.filedialog as fDialog
import tkinter.messagebox as tkMessageBox
import tkinter.ttk as ttk
from abc import ABC
from collections import Counter
from tkinter import *

import matplotlib
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, accuracy_score

import decisionTree

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class Language:
    english = 'english'
    spanish = 'spanish'
    catalan = 'catalan'
    def __init__(self, language=english):
        """
        :param language: The language the user wants to select
        Initialize all the strings in this class in the specified language
        """
        if language == Language.english:
            self.setEnglish()
        elif language == Language.spanish:
            self.setSpanish()
        elif language == Language.catalan:
            self.setCatalan()

    def setEnglish(self):
        """
        Initialize all the string to English
        """
        self.title = 'Decision Tree Classifier'

        self.autosplit = 'Autosplit'
        self.prune = 'Prune'
        self.join = 'Join'
        self.split = 'Split'
        self.bestSplit = 'Best split'
        self.adOptions = 'Advanced options'
        self.validate = 'Validate'
        self.varSplit = 'Variable split'
        self.naiveBayes = 'Naive Bayes'
        self.bestTree = 'Best Tree'

        self.infoNumElems = 'Number of elements'
        self.infoAccuray = 'Accuracy'
        self.infoPrediction = 'Prediction'
        self.infoAttrSplit = 'Variable split'
        self.inforImpurity = 'Impurity'
        self.accuracy = 'Accuracy: '

        self.file = 'File'
        self.newTree = 'New Decision Tree'
        self.editTree = 'Edit Decision Tree'
        self.newPrediction = 'New prediction'
        self.saveTree = 'Save Decision Tree'
        self.help = 'Help'
        self.about = 'About'

        self.predict = 'Predict'
        self.predictFile = 'Predict from file'
        self.advOptions = 'Advanced options'
        self.minSetSize = 'Min dataset size'
        self.minImpReduction = 'Min impurity reduction'
        self.fImp = 'Function impurity'
        self.fPerfKmeans = 'Performance Kmeans'
        self.variable = 'Variable'
        self.howToSplit = 'How to split'
        self.accept = 'Accept'
        self.cancel = 'Cancel'
        self.yesNo = 'Yes/No'

        self.gini = 'Gini'
        self.entropy = 'Entropy'
        self.silhouette = 'Silhouette'
        self.varRed = 'Variance reduction'
        self.error = 'Error'

        self.fpr = 'False Positive Rate'
        self.tpr = 'True Positive Rate'
        self.rocCurve = 'ROC curve'
        self.predictionDone = 'Prediction done!'
        self.precision = ' | Precision: '
        self.recall = ' | Recall: '

    def setSpanish(self):
        """
        Initialize all the string to Spanish
        """
        pass

    def setCatalan(self):
        """
        Initialize all the string to Catalan
        """
        pass
        
class MyMenu:
    def __init__(self, master):
        """
        :param master: The root Tk window where the menu will be inserted
        Fills the master frame with a menu. This menu can give access to the main functionalities of the application,
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

        self.mHelp = Menu(self.menu)
        self.mHelp.add_command(label=lg.help, command=None)
        self.mHelp.add_command(label=lg.about, command=None)
        self.menu.add_cascade(label=lg.help, menu=self.mHelp)

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
        Shows a dialog window to choose a csv file and it creates a view for building a decision tree from the data of the
        selected file
        """
        FILEOPENOPTIONS = dict(defaultextension='.csv', filetypes=[('cvs file','*.csv')])
        file = fDialog.askopenfile(mode='r', **FILEOPENOPTIONS)
        df = pd.read_csv(file.name).sample(frac=1)

        df2 = df.iloc[:,:len(df.columns)-1]
        df3 = df.iloc[:,len(df.columns)-1]
        X = df2.values.tolist()
        y = df3.values.flatten().tolist()
        nTrain = round(0.7 * len(y))

        dcTree = decisionTree.DecisionTree(X[:nTrain], y[:nTrain], sorted(list(set(y)), reverse=True),
                                           f=decisionTree.gini, attrNames=list(df2.columns))
        self.resetFrame()
        self.currentView = EditTreeGUI(self.mainFrame, dcTree, X[nTrain:], y[nTrain:])

    def newPrediction(self):
        """
        Shows a dialog window to choose a pkl file (an existing DecisionTree) and creates a view to make predictions
        based on the loaded DecisionTree
        """
        FILEOPENOPTIONS = dict(defaultextension='.pkl', filetypes=[('pkl file','*.pkl')])
        file = fDialog.askopenfile(mode='r', **FILEOPENOPTIONS)
        with open(file.name, 'rb') as input_:
            auxDcTree = pickle.load(input_)
            dcTree = decisionTree.DecisionTree.copyVarTree(auxDcTree)
        self.resetFrame()
        self.currentView = PredictGUI(self.mainFrame, dcTree)

    def saveTree(self):
        """
        Opens a dialog window to save the current DecisionTree
        """
        if type(self.currentView) == EditTreeGUI:
            file = fDialog.asksaveasfile(mode='w', defaultextension=".pkl")
            self.currentView.saveDcTree(file.name)


    def editTree(self):
        """
        Opens a dialog window to open an existing DesicionTree and continue editing it
        """
        FILEOPENOPTIONS = dict(defaultextension='.pkl', filetypes=[('pkl file','*.pkl')])
        file = fDialog.askopenfile(mode='r', **FILEOPENOPTIONS)
        with open(file.name, 'rb') as input_:
            auxDcTree = pickle.load(input_)
            X_cv = auxDcTree.X_cv
            y_cv = auxDcTree.y_cv
            dcTree = decisionTree.DecisionTree.copyVarTree(auxDcTree)
        self.resetFrame()
        self.currentView = EditTreeGUI(self.mainFrame, dcTree, X_cv, y_cv)


class TreeFrame(ABC):
    keyImpurity = "impurity"
    keyPrediction = "prediction"
    keyAttrSplit = "attrSplit"
    def __init__(self, master, dcTree, parent, packSide=BOTTOM):
        """
        :param master: The root frame where this view will be inserted
        :param dcTree: A DecisionTree
        :param parent: The parent frame
        :param packSide: How to pack this view
        Creates a frame that displays a DecisionTree. Abstract class
        """
        self.dcTree = dcTree
        self.master = master
        self.parent = parent
        self.mapNode = dict() # diccionary that translates the id of a node in the GUI to a node from the class decisionTree

        # Set up the GUI Tree
        self.gui_tree = ttk.Treeview(master, height=25)
        self.gui_tree["columns"] = (TreeFrame.keyImpurity, TreeFrame.keyPrediction, TreeFrame.keyAttrSplit)
        self.gui_tree.column(TreeFrame.keyImpurity, width=100) # information about the impurity
        self.gui_tree.column(TreeFrame.keyPrediction, width=100) # information about the prediction
        self.gui_tree.column(TreeFrame.keyAttrSplit, width=100) # information about the attribute used to split the node
        self.gui_tree.heading(TreeFrame.keyImpurity, text=lg.inforImpurity)
        self.gui_tree.heading(TreeFrame.keyPrediction, text=lg.infoPrediction)
        self.gui_tree.heading(TreeFrame.keyAttrSplit, text=lg.infoAttrSplit)

        self.tree_root_id = self.gui_tree.insert('', 'end', text=str(self.dcTree.getNumElems()),
                                         values=(str(self.dcTree.getImpurity()), str(self.dcTree.getPrediction()),
                                                 str(self.dcTree.getAttrSplit())))
        # update mapNode with the root DecisionTree
        self.mapNode[self.tree_root_id] = dcTree
        self.addNodes(self.tree_root_id, dcTree)

        self.gui_tree.bind('<Button-1>', self.nodeClicked)
        self.gui_tree.focus(self.tree_root_id)
        self.gui_tree.pack(side=packSide)

    def addNodes(self, rootGUI, rootDT):
        """
        :param rootGUI: Identifier of a node in the GUI
        :param rootDT: A node of the DecisionTree
        Add recursivelly all the sons that rootDT has to the rootGUI. rootGUI and rootDT must represent the same node
        """
        self.gui_tree.set(rootGUI, TreeFrame.keyAttrSplit, str(rootDT.getAttrSplit()))
        for (i, son) in enumerate(rootDT.getSons()):
            idSon = self.gui_tree.insert(rootGUI, 'end', text=str(son.getNumElems()),
                                         values=(str(son.getImpurity()), str(son.getPrediction()), str(son.getAttrSplit())))
            self.mapNode[idSon] = son
            self.addNodes(idSon, son)

    def refreshInfoNodes(self, rootGUI, rootDT):
        """
        :param rootGUI: Identifier of a node in the GUI
        :param rootDT: A node of the DecisionTree
        Updates all the information displayed in rootGUI and all its sons recursivelly using the data from rootDT.
        rootGUI and rootDT must represent the same node
        """
        self.gui_tree.set(rootGUI, TreeFrame.keyImpurity, str(rootDT.getImpurity()))
        for (i, son) in enumerate(rootDT.getSons()):
            self.refreshInfoNodes(self.gui_tree.get_children(rootGUI)[i], son)

    def nodeClicked(self, event):
        """
        Function used to catch a left clic event in the GUI Tree
        """
        pass

    def predict_cv(self, X, naiveBayes):
        """
        :param X: A matrix containing the data that will be predicted
        :param naiveBayes: If true, the Naive Bayes predictor will be used
        :return: Calls the predict function from the DecisionTree and returns its value
        """
        return self.dcTree.predict(X, naiveBayes)


class TreeFrameEdit(TreeFrame):
    """
    Class that extends the TreeFrame abstract class
    """
    def getSegData(self, selectedAttr):
        """
        :param selectedAttr: Name of an attribute
        :return: The data of the selected node segmented using selectedAttr
        """
        node = self.mapNode[self.gui_tree.focus()]
        segData = node.getSegmentedData(self.dcTree.attrNames.index(selectedAttr))
        return segData

    def nodeClicked(self, event):
        dcTree = self.mapNode[self.gui_tree.focus()]

    def autoSplit(self, minSetSize, giniReduction):
        """
        Splits automatically and recursivelly the selected node
        """
        self.prune()
        dcTree = self.mapNode[self.gui_tree.focus()]
        dcTree.autoSplit(minSetSize=minSetSize, giniReduction=giniReduction)
        self.addNodes(self.gui_tree.focus(), dcTree)

    def updateTreeView(self):
        for node in self.gui_tree.get_children(self.tree_root_id):
            self.mapNode.pop(node)
            self.gui_tree.delete(node)
        self.addNodes(self.tree_root_id, self.dcTree)

    def joinNodes(self):
        """
        Joins the selected nodes into one
        """
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

    def prune(self):
        """
        Eliminates all the sons of the selected node
        """
        nodeGUI = self.gui_tree.focus()
        dcTree = self.mapNode[nodeGUI]
        self.gui_tree.set(nodeGUI, TreeFrame.keyAttrSplit, str(dcTree.getAttrSplit()))
        dcTree.prune()
        for node in self.gui_tree.get_children(nodeGUI):
            self.mapNode.pop(node)
            self.gui_tree.delete(node)

    def split(self, idxAttr):
        """
        :param idxAttr: Index of the attribute used to split the node
        Splits the selected node according to the attribute specified in idxAttr
        """
        self.prune()
        nodeGUI = self.gui_tree.focus()
        dcTree = self.mapNode[nodeGUI]
        dcTree.splitNode(idxAttr)
        self.addNodes(nodeGUI, dcTree)

    def bestSplit(self):
        """
        :return: A sorted list in increasing order of the possible splits
        """
        nodeGUI = self.gui_tree.focus()
        dcTree = self.mapNode[nodeGUI]
        return dcTree.bestSplit()

    def nodeClicked(self, event):
        """
        Shows the plot corresponding to the selected node
        """
        self.parent.changePlot()


class TreeFramePredict(TreeFrame):
    pass

class EditTreeGUI:
    def __init__(self, master, dcTree, X_cv, y_cv):
        """
        :param master: Root frame where this view will be displayed
        :param dcTree: A DecisionTree
        :param X_cv: X test data
        :param y_cv: y test data
        """
        self.master = master
        self.dcTree = dcTree
        self.X_cv = X_cv
        self.y_cv = y_cv
        self.minSetSize = 1000
        self.minImpRed = 0.01
        self.naiveBayes = False

        # Left Frame #
        leftFrame = Frame(self.master)
        leftFrame.pack(side=LEFT, padx=10, pady=10, anchor=S+W)
        # Tree
        self.treeFrame = TreeFrameEdit(leftFrame, self.dcTree, self)

        # Buttons Frame #
        buttonsFrame = Frame(leftFrame)
        buttonsFrame.pack(side=TOP, padx=10, pady=10)
        # Buttons
        # Button validate
        b_validate = Button(buttonsFrame, text=lg.validate)
        b_validate.grid(row=0, column=0, sticky=N+S+E+W)
        b_validate.bind('<Button-1>', self.predict_cv)
        # Button advanced options
        b_adOptions = Button(buttonsFrame, text=lg.adOptions)
        b_adOptions.grid(row=1, column=0, sticky=N+S+E+W)
        b_adOptions.bind('<Button-1>', self.advancedOptions)
        # Button prune
        b_prune = Button(buttonsFrame, text=lg.prune)
        b_prune.grid(row=0, column=1, sticky=N+S+E+W)
        b_prune.bind('<Button-1>', self.prune)
        # Button join
        b_join = Button(buttonsFrame, text=lg.join)
        b_join.grid(row=1, column=1, sticky=N+S+E+W)
        b_join.bind('<Button-1>', self.joinNodes)
        # Button autosplit
        b_autoSplit = Button(buttonsFrame, text=lg.autosplit)
        b_autoSplit.grid(row=0, column=2, sticky=N+S+E+W)
        b_autoSplit.bind('<Button-1>', self.autoSplit)
        # Button best split
        b_bestSplit = Button(buttonsFrame, text=lg.bestSplit)
        b_bestSplit.grid(row=1, column=2, sticky=N+S+E+W)
        b_bestSplit.bind('<Button-1>', self.bestSplit)
        # Button best split
        b_split = Button(buttonsFrame, text=lg.split)
        b_split.grid(row=0, column=3, sticky=N+S+E+W)
        b_split.bind('<Button-1>', self.split)
        # Button best tree
        b_bestTree = Button(buttonsFrame, text=lg.bestTree)
        b_bestTree.grid(row=1, column=3, sticky=N+S+E+W)
        b_bestTree.bind('<Button-1>', self.bestTree)
        # Option Menu
        self.tkvar = StringVar(root)
        self.tkvar.set(self.dcTree.attrNames[0]) # set the default option
        self.tkvar.trace('w', self.optionMenuClicked)
        self.popupMenu = OptionMenu(buttonsFrame, self.tkvar, *self.dcTree.attrNames)
        self.popupMenu.grid(row=0, column=4, sticky=N+S+E+W)

        # Right Frame #
        rightFrame = Frame(self.master, padx=10, pady=10)
        rightFrame.pack(side=RIGHT, anchor=S+E)
        # The plot
        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=rightFrame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.changePlot()

    def bestTree(self, event):
        """
        Calls the TreeFrame.bestTree() method
        """
        self.dcTree.bestTree(self.X_cv, self.y_cv, self.minImpRed)
        self.treeFrame.updateTreeView()

    def joinNodes(self, event):
        """
        Calls the TreeFrame.joinNodes() method
        """
        self.treeFrame.joinNodes()

    def prune(self, event):
        """
        Calls the TreeFrame.prune() method
        """
        self.treeFrame.prune()

    def autoSplit(self, event):
        """
        Calls the TreeFrame.autoSplit() method
        """
        self.treeFrame.autoSplit(self.minSetSize, self.minImpRed)

    def split(self, event):
        """
        Calls the TreeFrame.split() method
        """
        selectedAttr = self.tkvar.get()
        idxAttr = self.dcTree.attrNames.index(selectedAttr)
        self.treeFrame.split(idxAttr)

    def bestSplit(self, event):
        """
        Calls the TreeFrame.bestSlpit() method and show the information computed
        """
        listSplits = self.treeFrame.bestSplit()
        listSplits = [(gImp, self.dcTree.attrNames[i]) for (gImp, i) in listSplits]
        InfoBestSplits(listSplits)

    def predict_cv(self, event):
        """
        Calculates the accuracy of the DecisionTree using the test data and plots the ROC curve
        """
        pred_y = self.treeFrame.predict_cv(self.X_cv, self.naiveBayes) # [(prob, cls), (prob, cls), ...]
        tags = [self.dcTree.classes[max(enumerate(elem), key=lambda x: x[1])[0]] for elem in pred_y]
        # accuracy = sum([elem[0] == elem[1] for elem in zip(self.y_cv, tags)]) / len(self.y_cv)
        accuracy = accuracy_score(self.y_cv, tags)
        precision = precision_score(self.y_cv, tags, average='macro')
        recall = recall_score(self.y_cv, tags, average='macro')

        fpr = dict() # false positives
        tpr = dict() # true positives
        prob = np.array([list(zip(*elem))[0] for elem in pred_y])
        y_cv = np.array(self.y_cv)
        plt.figure().canvas.set_window_title(lg.rocCurve)
        for i in range(len(self.dcTree.classes)):
            fpr[i], tpr[i], _ = roc_curve(y_cv, prob[:, i], pos_label=self.dcTree.classes[i])
            area = round(auc(fpr[i], tpr[i]), 2)
            plt.plot(fpr[i], tpr[i], label=str(self.dcTree.classes[i]) + ' (area: ' + str(area) + ')')
        plt.title(lg.accuracy + str(round(accuracy, 4)) + lg.precision + str(round(precision, 4)) +
                  lg.recall + str(round(recall, 4)))
        plt.xlabel(lg.fpr)
        plt.ylabel(lg.tpr)
        plt.legend(loc="lower right")
        plt.show()

    def optionMenuClicked(self, *args):
        self.changePlot()

    def changePlot(self):
        """
        Change the plot according to the selected node and the selected attribute
        """
        selectedAttr = self.tkvar.get()
        segData = self.treeFrame.getSegData(selectedAttr)
        self.figure.clear()
        subPlot = self.figure.add_subplot(111)
        axes = self.figure.add_axes()
        # TODO Una mica lleig, intentar compactar el codi i fer-lo mes clar
        if type(segData[0][0]) == int or type(segData[0][0]) == float:
            rang = (min(min(elem) for elem in segData), max(max(elem) for elem in segData))
            nBins = min(300, max([len(set(d)) for d in segData]))
            auxHist = [0] * nBins
            for (i, data) in enumerate(segData):
                h = subPlot.hist(data, bins=nBins, range=rang, bottom=auxHist, label=str(self.dcTree.classes[i]))
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
        """
        :param file: File where the DecisionTree will be saved
        Saves the DecisionTree in the specified file
        """
        self.dcTree.X_cv = self.X_cv
        self.dcTree.y_cv = self.y_cv
        with open(file, 'wb') as output:
            pickler = pickle.Pickler(output, -1)
            pickler.dump(self.dcTree)
            
    def advancedOptions(self, event):
        """
        Opens the advanced options window
        """
        AdvancedOptionsGUI(self, self.dcTree)

    def receiveChanges(self, minSetSize, minImpRed, naiveBayes):
        """
        Updates some parameters and the TreeView when there have been changes in the advanced options window
        """
        self.minSetSize = minSetSize
        self.minImpRed = minImpRed
        self.naiveBayes = naiveBayes
        self.treeFrame.refreshInfoNodes(self.treeFrame.tree_root_id, self.treeFrame.dcTree)

class AdvancedOptionsGUI:
    def __init__(self, frameParent, dcTree):
        """
        :param frameParent: Frame that has called this constructor
        :param dcTree: A DecisionTree
        """
        self.frameParent = frameParent
        self.dcTree = dcTree
        self.lAttr = dcTree.attrNames
        self.root = Tk()
        self.root.title(lg.advOptions)

        # Top Frame
        topFrame = Frame(self.root)
        topFrame.pack(side=TOP, padx=10, pady=10)
        # MinSetSize
        Label(topFrame, text=lg.minSetSize).grid(row=0, column=0)
        self.eMinSetSize = Entry(topFrame)
        self.eMinSetSize.grid(row=0, column=1)
        self.eMinSetSize.insert(END, str(self.frameParent.minSetSize))
        # MinGiniReduction
        Label(topFrame, text=lg.minImpReduction).grid(row=1, column=0)
        self.eMinImpReduction = Entry(topFrame)
        self.eMinImpReduction.grid(row=1, column=1)
        self.eMinImpReduction.insert(END, str(self.frameParent.minImpRed))
        # f_imp
        Label(topFrame, text=lg.fImp).grid(row=2, column=0)
        self.tkvarFImp = StringVar(topFrame)
        if self.dcTree.f == decisionTree.gini:
            self.tkvarFImp.set(lg.gini) # set the default option
        elif self.dcTree.f == decisionTree.entropy:
            self.tkvarFImp.set(lg.entropy) # set the default option
        # self.tkvarFImp.trace('w', self.fImpSelected)
        self.menuFImp = OptionMenu(topFrame, self.tkvarFImp, *[lg.gini, lg.entropy])
        self.menuFImp.grid(row=2, column=1)
        # f_Kmeans
        Label(topFrame, text=lg.fPerfKmeans).grid(row=3, column=0)
        self.tkvarFKmeans = StringVar(topFrame)
        if self.dcTree.perfKmeans == decisionTree.perfKmeansSilhouette:
            self.tkvarFKmeans.set(lg.silhouette) # set the default option
        elif self.dcTree.perfKmeans == decisionTree.perfKmeanVar:
            self.tkvarFKmeans.set(lg.varRed) # set the default option
        # self.tkvarFKmeans.trace('w', self.fImpSelected)
        self.menuFKmeans = OptionMenu(topFrame, self.tkvarFKmeans, *[lg.silhouette, lg.varRed])
        self.menuFKmeans.grid(row=3, column=1)
        # Naive Bayes
        Label(topFrame, text=lg.naiveBayes).grid(row=4, column=0)
        self.varNB = IntVar(topFrame)
        self.naiveBayes = Checkbutton(topFrame, text=lg.yesNo, variable=self.varNB, command=self.cb)
        self.naiveBayes.grid(row=4, column=1)
        self.varNB.set(self.frameParent.naiveBayes)

        # Middle Frame
        middleFrame = Frame(self.root)
        middleFrame.pack(side=TOP, padx=10, pady=10)
        # Titles
        Label(middleFrame, text=lg.variable, font="Verdana 10 bold").grid(row=0, column=0)
        Label(middleFrame, text=lg.howToSplit, font="Verdana 10 bold").grid(row=0, column=1)
        self.lHowToSplit = []
        for (i, attrName) in enumerate(self.lAttr):
            # How to split labels and entries
            Label(middleFrame, text=attrName).grid(row=i+1, column=0)
            entryHowToSplit = Entry(middleFrame)
            entryHowToSplit.grid(row=i+1, column=1)
            if i in self.dcTree.staticSplits:
                entryHowToSplit.insert(END, str(self.dcTree.staticSplits[i]))
            self.lHowToSplit.append(entryHowToSplit)

        # Bottom Frame
        bottomFrame = Frame(self.root)
        bottomFrame.pack(side=BOTTOM)
        self.b_accept = Button(bottomFrame, text=lg.accept)
        self.b_accept.bind('<Button-1>', self.accept)
        self.b_accept.grid(row=0, column=0)
        self.b_cancel = Button(bottomFrame, text=lg.cancel)
        self.b_cancel.bind('<Button-1>', self.cancel)
        self.b_cancel.grid(row=0, column=1)

        self.root.mainloop()
        
    # def fImpSelected(self, *args):
    #     pass

    def cb(self):
        # TODO Only for testing uses
        print(self.varNB.get())

    def accept(self, event):
        """
        Passes the changes to the parent frame and closes the window
        """
        # parse minSetSize
        try:
            minSetSize = int(self.eMinSetSize.get())
            if minSetSize <= 0:
                self.throwError()
        except ValueError:
            self.throwError()
        # parse minGiniReduction
        try:
            minGiniRed = float(self.eMinImpReduction.get())
            if minGiniRed < 0:
                self.throwError()
        except ValueError:
            self.throwError()
        # parse attrSplit
        dictHowToSplit = dict()
        for (i, auxEntry) in enumerate(self.lHowToSplit):
            if auxEntry.get() != '':
                dictHowToSplit[i] = eval(auxEntry.get())

        # set the new values to the dcTree
        if self.tkvarFImp.get() == lg.gini:
            self.dcTree.f = decisionTree.gini
        elif self.tkvarFImp.get() == lg.entropy:
            self.dcTree.f = decisionTree.entropy
        if self.tkvarFKmeans.get() == lg.silhouette:
            self.dcTree.perfKmeans = decisionTree.perfKmeansSilhouette
        elif self.tkvarFKmeans.get() == lg.varRed:
            self.dcTree.perfKmeans = decisionTree.perfKmeanVar
        self.dcTree.staticSplits = dictHowToSplit
        self.dcTree.propagateChanges()
        self.frameParent.receiveChanges(minSetSize, minGiniRed, self.varNB.get())

        self.root.destroy()

    def cancel(self, event):
        """
        Closes the window without saving any changes
        """
        self.root.destroy()

    def throwError(self):
        """
        Show a message box error to the user
        """
        tkMessageBox.showerror(lg.error, lg.error)

class InfoBestSplits:
    def __init__(self, listSplits):
        """
        :param listSplits: [(impurity1, attribute1), (impurity2, attribute2), ...]
        Creates a window that shows the impurity for each attribute given with the parameter listSplits
        """
        self.root = Tk()
        self.root.title(lg.bestSplit)
        for (i, (gImp, attr)) in enumerate(listSplits):
            labelAttr = Label(self.root, text=attr, borderwidth=1)
            labelAttr.grid(row=i, column=0)
            labelGImp = Label(self.root, text=str(round(gImp, 4)), borderwidth=1)
            labelGImp.grid(row=i, column=1)
        self.root.mainloop()

class PredictGUI:
    def __init__(self, master, dcTree):
        """
        :param master: Root frame where this view will be displayed
        :param dcTree: A DecisionTree
        Creates a view to show a DecisionTree and make predictions
        """
        self.master = master
        # Left Frame #
        leftFrame = Frame(self.master)
        leftFrame.pack(side=LEFT, anchor=W, padx=20, pady=20)
        # Tree
        self.dcTree = dcTree
        self.treeFrame = TreeFramePredict(leftFrame, dcTree, self)

        # Right frame
        rightFrame = Frame(self.master, width=100)
        rightFrame.pack(side=TOP, anchor=E, padx=20, pady=20)
        # All the entries
        self.listEntries = []
        for (i, attr) in enumerate(self.dcTree.attrNames):
            Label(rightFrame, text=str(attr)).grid(row=i, column=0, sticky=N)
            entry = Entry(rightFrame, width=13)
            entry.grid(row=i, column=1, sticky=N)
            self.listEntries.append(entry)
        Label(rightFrame, text=lg.infoPrediction, font="Verdana 10 bold").grid(row=len(self.dcTree.attrNames), column=0, sticky=N)
        self.labelPred = Label(rightFrame, text='-', relief=RIDGE)
        self.labelPred.grid(row=len(self.dcTree.attrNames), column=1, sticky=N)

        # The naive bayes checkbox
        Label(rightFrame, text=lg.naiveBayes).grid(row=len(self.dcTree.attrNames)+1, column=0)
        self.varNB = IntVar(rightFrame)
        self.naiveBayes = Checkbutton(rightFrame, text=lg.yesNo, variable=self.varNB)
        self.naiveBayes.grid(row=len(self.dcTree.attrNames)+1, column=1, padx=10, pady=20)
        self.varNB.set(False)
        # The single prediction button
        self.b_predict = Button(rightFrame, text=lg.predict)
        self.b_predict.bind('<Button-1>', self.predict)
        self.b_predict.grid(row=len(self.dcTree.attrNames)+2, column=0, padx=10)
        # The file prediction button
        self.b_predict_file = Button(rightFrame, text=lg.predictFile)
        self.b_predict_file.bind('<Button-1>', self.predictFile)
        self.b_predict_file.grid(row=len(self.dcTree.attrNames)+2, column=1, padx=10)

    def predict(self, event):
        """
        Shows the prediction of the element specified in the entries
        """
        ll = []
        for l_attr in self.listEntries:
            str_attr = l_attr.get()
            try:
                ll.append(float(str_attr))
            except:
                ll.append(str_attr)
        result = self.dcTree.predict(ll, self.varNB.get())
        # print(result[0])
        self.labelPred['text'] = str(result[0])

    def predictFile(self, event):
        """
        Shows a dialog to open a csv file and make a prediction. The result will be stored in a file called
        like the original file but with the '_prediction.csv' string appended
        """
        FILEOPENOPTIONS = dict(defaultextension='.csv', filetypes=[('cvs file','*.csv')])
        file = fDialog.askopenfile(mode='r', **FILEOPENOPTIONS)
        df = pd.read_csv(file.name).sample(frac=1)
        X = df.values.tolist()
        y_pred = self.dcTree.predict(X, False)
        prob = np.array([list(zip(*elem))[0] for elem in y_pred])
        for (i, cls) in enumerate(self.dcTree.classes):
            df[str(cls)] = prob[:, i]
        df.to_csv(path_or_buf=file.name + '_prediction.csv')
        tkMessageBox.showinfo('', lg.predictionDone)

if __name__ == '__main__':
    lg = Language()
    root = Tk()
    root.title = lg.title
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w, h))
    menu = MyMenu(root)
    root.mainloop()
