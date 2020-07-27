# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 00:23:30 2020

@author: mahda

# =============================================================================
# 
# Copyright (c) 2020 Mahdad Jafarzadeh
# 
# Zzzscoring: A GUI-based package for sleep scoring!
# =============================================================================
"""

from tkinter import LabelFrame, Label, Button, filedialog, messagebox,OptionMenu, StringVar,DoubleVar
from tkinter import *
import mne
import numpy as np
from   numpy import loadtxt
import time
from ssccoorriinngg import ssccoorriinngg
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


class Zzzscoring():
    
    def __init__(self, master):
        
        self.master = master
        
        master.title("Zzzscoring: Automatic sleep scoring package")
        
        #### !!~~~~~~~~~~~~~~~~~ DEFINE INPUT DATAFRAME ~~~~~~~~~~~~~~~~~!!####
        
        self.frame_import = LabelFrame(self.master, text = "Import files section", padx = 150, pady = 100,
                                  font = 'Calibri 18 bold')
        self.frame_import.grid(row = 0 , column = 0, padx = 200, pady = 50, columnspan = 8)
        
        
        #### ==================== Help pop-up button ======================####
        
        self.popup_button = Button(self.master, text = "Help", command = self.help_pop_up_func,
                              font = 'Calibri 13 bold', fg = 'white', bg = 'black')
        self.popup_button.grid(row = 1, column = 8)
        
        #### ==================== Import data EDFs ========================####
        # Label: Import EDF
        self.label_import = Label(self.frame_import, text = "Import EDF files here:",
                                  font = 'Calibri 12 bold')
        self.label_import.grid(row = 0 , column = 0, padx = 15, pady = 10)
        
        # Button: Import EDF (Browse)
        self.button_import_browse = Button(self.frame_import, text = "Browse data",
                                           padx = 100, pady = 20,font = 'Calibri 10 bold',
                                           command = self.load_data_file_dialog, fg = 'blue',
                                           relief = RIDGE)
        self.button_import_browse.grid(row = 1, column = 0, padx = 15, pady = 10)
        
        #### ================== Import hypnogram files ====================####
        # Show a message about hypnograms
        self.label_hypnos = Label(self.frame_import, text = "Import hypnogram file (.txt) here:",
                                  font = 'Calibri 12 bold')
        self.label_hypnos.grid(row = 0 , column = 1, padx = 15, pady = 10)
        
        # Define browse button to import hypnos
        self.button_hypnos_browse = Button(self.frame_import, text = "Browse labels", 
                                           padx = 100, pady = 20, font = 'Calibri 10 bold',
                                           command = self.load_hypno_file_dialog,fg = 'blue',
                                           relief = RIDGE)
        self.button_hypnos_browse.grid(row = 1, column = 1, padx = 15, pady = 10)
        
        #### ===================== Define train size ======================####
        # Define train size section
        self.label_train_size = Label(self.frame_import, text = "Train size portion (between 0 - 1):",
                                      font = 'Calibri 12 bold')
        self.label_train_size.grid(row = 0 , column = 3, padx = 15, pady = 10)
        
        # Bar to ask for user's entry
        self.train_size = DoubleVar()
        self.train_size.set(0.7)
        self.entry_train_size = OptionMenu(self.frame_import, self.train_size, 0.6, 0.7, 0.8, 0.9)
        self.entry_train_size.grid(row = 1, column = 3, padx = 15, pady = 10)
        self.entry_train_size.config(font= 'Calibri 10 bold', fg='black') 
        
        #### =================== Push apply to load data ==================####
        #Label to read data and extract features
        self.label_apply = Label(self.frame_import, text = "Press to Load, pre-process, and extract features!",
                                      font = 'Calibri 12 bold')
        self.label_apply.grid(row = 0 , column = 4)
        # Apply button
        self.button_apply = Button(self.frame_import, text = "Apply", padx = 100, pady=20,
                              font = 'Calibri 10 bold', relief = RIDGE, fg = 'blue',
                              command = self.Apply_button)
        self.button_apply.grid(row = 1 , column =4, padx = 15, pady = 10)

        #### !!~~~~~~~~~~~~~~ DEFINE ML SECTION FRAME ~~~~~~~~~~~~~~~~~~~!!####
        
        self.frame_ML = LabelFrame(self.master, text = "Machine Learning Section",
                                   padx = 150, pady = 100, font = 'Calibri 18 bold')
        self.frame_ML.grid(row = 1 , column = 0, padx = 200, pady = 50, columnspan = 8)
        
        #### ================ Pick ML Algorithm of interest ===============####
        # Label
        self.label_ML_algorithm = Label(self.frame_ML, text = "Choose the machine learning algorithm:",
                                        font = 'Calibri 12 bold')
        self.label_ML_algorithm.grid(row = 0, column = 0, padx = 15, pady = 10)
        
        # Dropdown menu
        self.selected_ML = StringVar()
        self.selected_ML.set("Random forest")
        self.drop = OptionMenu(self.frame_ML, self.selected_ML, "SVM", "Random forest","XGBoost","Logistic regression", "Naive bayes", "Randomized trees","GradientBoosting", "ADABoost")
        self.drop.grid(row = 1, column = 0)
        self.drop.config(font= 'Calibri 10 bold', fg='blue') 
        
        # label_selec
        self.label_select = Label(self.frame_ML, text = "Press after choosing ML algorithm:",
                                        font = 'Calibri 12 bold')
        self.label_select.grid(row = 0 , column =1)
        
        # select button
        self.button_select = Button(self.frame_ML, text = "Select!", padx = 100, pady=20,
                              font = 'Calibri 12 bold', relief = RIDGE, fg = 'blue',
                              command = self.Select_ML_button)
        self.button_select.grid(row = 1 , column =1, padx = 15, pady = 10) 
        
        # Chekbox for time-dependency
        
        self.td_var = IntVar()
        self.checkbox_td = Checkbutton(self.frame_ML, text = "Multi-to-one classifcation",
                                  font = 'Calibri 12 bold', variable = self.td_var)
        
        self.checkbox_td.grid(row = 2, column = 0)
        
        # Chekbox for feature selection
        
        self.feat_select_var = IntVar()
        self.checkbox_feat_select = Checkbutton(self.frame_ML, text = "Feature Selection",
                                  font = 'Calibri 12 bold', variable = self.feat_select_var)
        
        self.checkbox_feat_select.grid(row = 3, column = 0)
        

    #%% ################### DEFINE FUNCTIONS OF BUTTONS #######################
    #%% Function: Import EDF (Browse)
    def load_data_file_dialog(self):
    
        global data_files_list
        
        self.filenames        = filedialog.askopenfilenames(initialdir= "C:/",title = 'select data files', 
                                                       filetype = (("edf", "*.edf"), ("All Files", "*.*")))
        
        # Make a list of imported file names (full path)
        data_files_list       = self.frame_import.tk.splitlist(self.filenames)
        self.n_data_files     = len(data_files_list)
        
        # check if the user chose somthing
        if not data_files_list:
            
            self.label_data       = Label(self.frame_import, text = "No file has been selected!",
                                          fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 0)
    
        else:
            self.label_data       = Label(self.frame_import, text = str(self.n_data_files) + " EDF files has been loaded!",
                                          fg = 'green', font = 'Helvetica 9 bold').grid(row = 2, column = 0)
            
    #%% Function: Import Hypnogram (Browse)
    def load_hypno_file_dialog(self):
    
        global hypno_files_list
        
        self.filenames    = filedialog.askopenfilenames(initialdir= "C:/",title = 'select label files', 
                                                       filetype = (("txt", "*.txt"),("csv", "*.csv"), ("All Files", "*.*")))
        hypno_files_list  = self.frame_import.tk.splitlist(self.filenames)
        self.n_label_files     = len(hypno_files_list)
        
        # check if the user chose somthing
        if not hypno_files_list:
            
            self.label_labels  = Label(self.frame_import, text = "No hypnogram has been selected!",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 1)
            
        else:
            
            self.label_labels  = Label(self.frame_import, text = str(self.n_label_files) + " hypnogram files has been loaded!",
                                  fg = 'green', font = 'Helvetica 9 bold').grid(row = 2, column = 1)

        
    #%% Read EDF and hypnograms and apply feature extraction
    def Read_Preproc_FeatExtract(self):
        global subjects_dic, hyp_dic, dic_pciked_chans
        subjects_dic     = {}
        hyp_dic          = {}
        dic_pciked_chans = {}
        
        #### ======================= Create log window ====================####
        self.log_win = Toplevel()
        self.log_win.title("Log file of current processes")
        
        # Label
        self.label = Label(self.log_win, text= "Process log file:",font = 'Helvetica 12 bold')
        
        self.label.pack()

        self.close_log_win = Button(self.log_win, text="Dismiss", command=self.log_win.destroy)
        self.close_log_win.pack()
        
        #### ======================= Read data files ======================####

        for idx, c_subj in enumerate(data_files_list):
            self.log1_ = Label(self.log_win, text = "Analyzing data: " + str(c_subj[-11:-4]) + "\tPlease wait ...").pack()
            print (f'Analyzing data: {c_subj[-11:-4]}')
            ## Read in data
            file     = data_files_list[idx]
            tic      = time.time()
            data     = mne.io.read_raw_edf(file)
            # Data raw EEG --> Deactive
            # data.plot(duration = 30, highpass = .3 , lowpass = 25 )
            raw_data = data.get_data()
            print('Time to read EDF: {}'.format(time.time()-tic))
            self.log2_ = Label(self.log_win, text = "Time to read EDF data (s): " +str(np.round(time.time()-tic))).pack()

    #####=================Retrieving information from data====================#####
            
# =============================================================================
#             DataInfo          = data.info
#             AvailableChannels = DataInfo['ch_names']
#             self.fs                = int(DataInfo['sfreq'])
# =============================================================================
            

    #####================= Find index of required channels ===================#####
            
# =============================================================================
#             for indx, c in enumerate(AvailableChannels):
#                 if c in RequiredChannels:
#                     Idx.append(indx)
#                 elif c in Mastoids:
#                     Idx_Mastoids.append(indx)
# =============================================================================
        
    #####===== Sampling rate is 200hz; thus 1 epoch(30s) is 6000 samples =====#####
            self.fs = 256
            T = 30 #secs
            len_epoch   = self.fs * T
            start_epoch = 0
            n_channels  =  1
               
    #####============ Cut tail; use modulo to find full epochs ===============#####
        
            raw_data = raw_data[:, 0:raw_data.shape[1] - raw_data.shape[1]%len_epoch]
            
    #####========== Reshape data [n_channel, len_epoch, n_epochs] ============#####
            data_epoched = np.reshape(raw_data,
                                      (n_channels, len_epoch,
                                       int(raw_data.shape[1]/len_epoch)), order='F' )
            
    #####===================== Reading hypnogram data ========================#####
        
            hyp = loadtxt(hypno_files_list[idx])

    ### Create sepereate data subfiles based on hypnogram (N1, N2, N3, NREM, REM) 
            tic      = time.time()
            
    #####================= Concatenation of selected channels ================#####   
        
    # Calculate referenced channels: 
    
            #data_epoched_selected = data_epoched[Idx] - data_epoched[Idx_Mastoids]
            
    #####================= Find order of the selected channels ===============#####   
# =============================================================================
#             #Init
#             picked_channels = []
#             picked_refs     = []
#             List_Channels   = []
#             
#             # Find main channels
#             for jj,kk in enumerate(Idx):
#                 picked_channels = np.append(picked_channels, AvailableChannels[kk])
#             # Find references
#             for jj,kk in enumerate(Idx_Mastoids):
#                 picked_refs     = np.append(picked_refs, AvailableChannels[kk])
#             print(f'subject LK {c_subj} --> detected channels: {str(picked_channels)} -  {str(picked_refs)}')
#             self.log3_ = Label(self.log_win, text = "Dectected channels:"+str(picked_channels) +"-" +str(picked_refs)).pack()
# 
#             # Create lis of channels
#             for kk in np.arange(0, len(Idx)):
#                 List_Channels = np.append(List_Channels, picked_channels[kk] + '-' + picked_refs[kk])
# =============================================================================
            
        #%% Analysis section
        #####================= remove chanbnels without scroing ==================#####   
            
            # assign the proper data and labels
            #x_tmp_init = data_epoched_selected
            x_tmp_init = data_epoched

            y_tmp_init = hyp
            
            #Define ssccoorriinngg object:
            self.Object = ssccoorriinngg(filename='', channel='', fs = self.fs, T = 30)
            # Ensure equalituy of length for arrays:
            self.Object.Ensure_data_label_length(x_tmp_init, y_tmp_init)
            
            # Remove non-scored epochs
            x_tmp, y_tmp =  self.Object.remove_channels_without_scoring(hypno_labels = y_tmp_init,
                                                      input_feats = x_tmp_init)
            
            # Remove disconnections
            x_tmp, y_tmp =  self.Object.remove_disconnection(hypno_labels= y_tmp, 
                                                        input_feats=x_tmp)
            
        #####============= Create a one hot encoding form of labels ==============##### 
        
            # Create binary labels array
            self.yy = self.Object.One_hot_encoding(y_tmp)     
            
            # Ensure all the input labels have a class
            self.Object.Unlabaled_rows_detector(self.yy)
        
            #%% Function: Feature_Extraction
            # Initialize feature array:
            self.Feat_all_channels = np.empty((np.shape(x_tmp)[-1],0))
              
        #####================== Extract the relevant features ====================#####    
            
            for k in np.arange(np.shape(data_epoched)[0]):
                
                feat_temp         = self.Object.FeatureExtraction_per_subject(Input_data = x_tmp[k,:,:])
                self.Feat_all_channels = np.column_stack((self.Feat_all_channels,feat_temp))
                
            toc = time.time()
            print(f'Features of subject { c_subj[-11:-4]} were successfully extracted in: {toc-tic} secs')
            self.log4_ = Label(self.log_win, text = "Features of subject"+ str(c_subj[-11:-4])+" were successfully extracted in (secs):"+str(np.round(toc-tic))).pack()

            # Double check the equality of size of arrays
            self.Object.Ensure_feature_label_length(self.Feat_all_channels, self.yy)
            
            # Defining dictionary to save features PER SUBJECT
            subjects_dic["subject{}".format( c_subj)] = self.Feat_all_channels
            
            # Defining dictionary to save hypnogram PER SUBJECT
            hyp_dic["hyp{}".format( c_subj)] = self.yy
            
# =============================================================================
#             # Show picked channels per subject
#             dic_pciked_chans["subj{}".format(c_subj[-11:-4])] = List_Channels
# 
# =============================================================================
            
        #####=============== Removing variables for next iteration ===============#####      
            del x_tmp, y_tmp
            
            toc = time.time()
            
            print(f'Feature extraction of subject { c_subj[-11:-4]} has been finished.')   
            self.log5_ = Label(self.log_win, text = "Feature extraction of subject "+str(c_subj[-11:-4])+" has been finished.").pack()

        #print(f'Total feature extraction of subjects took {tic_tot - time.time()} secs.')
    

    #%% Function: Import Hypnogram (Browse)
    def Apply_button(self):
        
        print(f'Train size --> {str(self.train_size.get() * 100)}%')
        #### ======================= Get the train size ===================####
        self.train_size = self.train_size.get()
        
        # Has the user loaded hypnos?!
        if not hypno_files_list:            
            self.label_apply1  = Label(self.frame_import, text = "You haven't added any hypnogram!",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 4)
        # Has the user loaded EDF files?!    
        elif not data_files_list:            
            self.label_apply2  = Label(self.frame_import, text = "You haven't added any EDF files!",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 4)    
            
        # Check if train size is a valid value
        elif not self.train_size :
            self.label_apply1  = Label(self.frame_import, text = "No train size is entered!",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 4)
            
        elif float(self.train_size) > 0.99 or float(self.train_size) < 0.01 :
            self.label_apply1  = Label(self.frame_import, text = "Invalid train size! (acceptable range:0-1)",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 4)
        
        # Do the imported data an hypnos have the same amount of inputs?    
        elif len(data_files_list) != len(hypno_files_list):
            self.label_apply3  = Label(self.frame_import, text = "Size of the loaded hypons and EDF files do not match! Please recheck ...",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 4)
        # Else --> Go next
        if len(data_files_list) == len(hypno_files_list) and float(self.train_size)<.99 and float(self.train_size)>0.01:
            self.label_apply4  = Label(self.frame_import, text = "Train size: "+str(self.train_size)+"\nData and hypnogram have received in a good order!\n Go to next section to proceed ...",
                                  fg = 'green', font = 'Helvetica 9 bold').grid(row = 2, column = 4)
            
            self.Read_Preproc_FeatExtract()
            
    #%% Function: Import Hypnogram (Browse)
    def Select_ML_button(self):     
        # Check the current ML flag
        self.selected_ML = self.selected_ML.get()
        # Define label
        self.label_select1 = Label(self.frame_ML, text = str(self.selected_ML) + " has been selected.\n Please adjust the folloiwng hypermarameters.",
                                  fg = 'green', font = 'Helvetica 9 bold').grid(row = 2, column = 1)  
        
        # Report current td val
        print(f'Time dependence (0: deactive, 1: active) = {self.td_var.get()}')
        
        #### ============== Train the model and predict test ==============####
        self.label_train = Label(self.frame_ML, text = "Start Taining!",
                                        font = 'Calibri 12 bold')
        self.label_train.grid(row = 0 , column =3)
        # Activate train Button
        self.button_train = Button(self.frame_ML, text = "Train model!", padx = 100, pady=20,
                              font = 'Calibri 12 bold', relief = RIDGE, fg = 'blue',
                              command = self.Training_function)
        self.button_train.grid(row = 1 , column =3)
        
        #### ==============  multi-to-one classification flag ============ ####
        
        if int(self.td_var.get()) == 1:
             
            # Label
            self.label_checkbox = Label(self.frame_ML, text = "Time-dependence (#epochs):",
                                            font = 'Calibri 12 bold')
            self.label_checkbox.grid(row = 2 , column = 2)
            
            # Dropdown menu for td
            
            self.entry_td = IntVar()
            self.entry_td.set(5)
            self.drop_td = OptionMenu(self.frame_ML, self.entry_td, 1,2,3,4,5,6)
            self.drop_td.grid(row = 3, column = 2)
            self.drop_td.config(font= 'Calibri 10 bold', fg='blue') 
        # SVM Hyperparameters
            
        if self.selected_ML == "SVM":
            self.kernel_ = StringVar()
            #init
            self.kernel_.set('rbf')
            # Differnt Kernel functions of SVM:
            Kernel_rbf  = Radiobutton(self.frame_ML, text = 'rbf', variable = self.kernel_, value = 'rbf',
                                      font = 'Helvetica 12 bold')
            Kernel_sig  = Radiobutton(self.frame_ML, text = 'sigmoid', variable = self.kernel_, value = 'sigmoid',
                                      font = 'Helvetica 12 bold')
            Kernel_pol  = Radiobutton(self.frame_ML, text = 'poly', variable = self.kernel_, value = 'poly',
                                      font = 'Helvetica 12 bold')
            # Get the chosen Kernel 
            #self.kernel = self.kernel.get()
            
            # Shoving the radio buttons to the frame
            Kernel_rbf.grid(row = 0, column = 2)
            Kernel_sig.grid(row = 1, column = 2)
            Kernel_pol.grid(row = 2, column = 2)
        
        # Random forest hyperparameters
        elif self.selected_ML == "Random forest":
            self.n_estimator_RF = IntVar()
            self.n_estimator_RF.set(10)
            # Create n_estimators label
            self.label_n_estimators   = Label(self.frame_ML,text = "Number of trees:",
                                              font = 'Calibri 12 bold')
            self.label_n_estimators.grid(row = 0, column = 2, padx = 15, pady = 10)
            # Create entry for user
            self.entry_n_estimator_RF = Entry(self.frame_ML,text = " Enter the value here ", borderwidth = 8, width = 10)
            self.entry_n_estimator_RF.grid(row = 1, column = 2, padx = 15, pady = 10)
            # Assign the value to send to classifier
            #self.n_estimator_RF = self.entry_n_estimator_RF.get()
            
        # XGBoost
        elif self.selected_ML == "XGBoost":
            self.n_estimator_xgb = IntVar()
            # Create n_estimators label
            self.label_n_estimators   = Label(self.frame_ML,text = "Number of trees:",
                                              font = 'Calibri 12 bold')
            self.label_n_estimators.grid(row = 0, column = 2, padx = 15, pady = 10)
            # Create entry for user
            self.entry_n_estimator_xgb = Entry(self.frame_ML,text = " Enter the value here ", borderwidth = 8, width = 10)
            self.entry_n_estimator_xgb.grid(row = 1, column = 2, padx = 15, pady = 10)
            # Assign the value to send to classifier
            #self.n_estimator_xgb = self.entry_n_estimator_xgb.get()
            
        # ADABoost
        elif self.selected_ML == "ADABoost":
            self.n_estimator_ada = IntVar()
            # Create n_estimators label
            self.label_n_estimators   = Label(self.frame_ML,text = "Number of trees:",
                                              font = 'Calibri 12 bold')
            self.label_n_estimators.grid(row = 0, column = 2, padx = 15, pady = 10)
            # Create entry for user
            self.entry_n_estimator_ada = Entry(self.frame_ML,text = " Enter the value here ", borderwidth = 8, width = 10)
            self.entry_n_estimator_ada.grid(row = 1, column = 2, padx = 15, pady = 10)
            # Assign the value to send to classifier
            #self.n_estimator_ada = self.entry_n_estimator_ada.get()
            
        # GradientBoosting
        elif self.selected_ML == "GradientBoosting":
            self.n_estimator_gb = IntVar()
            # Create n_estimators label
            self.label_n_estimators   = Label(self.frame_ML,text = "Number of trees:",
                                              font = 'Calibri 12 bold')
            self.label_n_estimators.grid(row = 0, column = 2, padx = 15, pady = 10)
            # Create entry for user
            self.entry_n_estimator_gb = Entry(self.frame_ML,text = " Enter the value here ", borderwidth = 8, width = 10)
            self.entry_n_estimator_gb.grid(row = 1, column = 2, padx = 15, pady = 10)
            # Assign the value to send to classifier
            #self.n_estimator_gb = self.entry_n_estimator_gb.get()
            
        # Randomized trees
    
        elif self.selected_ML == "Randomized trees":
            self.n_estimator_rt = IntVar()
            # Create n_estimators label
            self.label_n_estimators   = Label(self.frame_ML,text = "Number of trees:",
                                              font = 'Calibri 12 bold')
            self.label_n_estimators.grid(row = 0, column = 2, padx = 15, pady = 10)
            # Create entry for user
            self.entry_n_estimator_rt = Entry(self.frame_ML,text = " Enter the value here ", borderwidth = 8, width = 10)
            self.entry_n_estimator_rt.grid(row = 1, column = 2, padx = 15, pady = 10)
            # Assign the value to send to classifier
            #self.n_estimator_rt = self.entry_n_estimator_rt.get()
        
        # Naive Bayes    
        elif self.selected_ML == "Naive Bayes":
            pass
        
        # Logistic regression
        elif self.selected_ML == "Logistic regression":
            pass
  
        print(f'Time dependence : {self.entry_td.get()} epochs')
    #%% Function: Help pop-up
    def help_pop_up_func(self):
        
        line1 = "Welcome to Zzzscoring!\n"
        line2 = "Pick the EDF files of interest and their corresponding hypnograms!\n"
        line3 = "** Notes:\n- hypnograms should be in .txt or .csv format.\n"
        line4 = "- The first and second column of hypnos are labels and artefact annotations, respecively.\n"
        line5 = "- Default labels should be as below:\n"
        line6 = "- Wake:0, N1: 1, N2: 2, SWS: 3, REM: 4.\n"
        line7 = "- Once pushing a button e,.g. 'Apply' or 'Train' the process is time-taking. One can follow the status of process in the console (e.g. command prompt).\n"
        line8 = "- After choosing a ML algorithm you should press 'Select' and then specify hyperparameters. After this, one is allowed to press 'Train' button.\n" 
        line9 = "- Activating feature selection extend the process time. Don't forget to follow the status from console. "
        lines = line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9
        
        messagebox.showinfo(title = "Help", message = lines)
        
    #%% Training function
    def Training_function(self):
        # Training perentage
        
        self.n_train = round(float(self.train_size) * len(data_files_list))
        
        # ========================== Show reuslt ============================ #
        # Label
        self.label_results = Label(self.frame_ML, text = "Train and prediction finished!",
                                        font = 'Calibri 12 bold')
        self.label_results.grid(row = 0 , column =4)
        # Activate results Button
        self.button_show_results   = Button(self.frame_ML, text = "Show results", padx = 100, pady=20,
                              font = 'Calibri 12 bold', relief = RIDGE, fg = 'blue',
                              command = self.show_results_function)
        self.button_show_results.grid(row = 1 , column =4)
        
        # ========================== Plot conf mat ========================== #
        # Activate plot confusion Button
        self.button_plot_conf = Button(self.frame_ML, text = "Plot confusion mat", padx = 100, pady=20,
                              font = 'Calibri 12 bold', relief = RIDGE, fg = 'blue',
                              command = self.plot_conf_mat)
        self.button_plot_conf.grid(row = 1 , column =5)
        
        # ========================== Activate plot hypnogram ================ #
        self.button_plot_hyp = Button(self.frame_ML, text = "Plot hypnograms", padx = 100, pady=20,
                              font = 'Calibri 12 bold', relief = RIDGE, fg = 'blue',
                              command = self.plot_hyp_function)
        self.button_plot_hyp.grid(row = 2 , column =4)
                      
        #######=== Randomly shuffle subjects to choose train and test splits ===######
    
# =============================================================================
#         subj_c = np.random.RandomState(seed=42).permutation(subj_c)
#         
# =============================================================================
        #######=============== Initialize train and test arrays ================#######
        global X_train, X_test, y_train, y_test
        X_train = np.empty((0, np.shape(self.Feat_all_channels)[1]))
        X_test  = np.empty((0, np.shape(self.Feat_all_channels)[1]))
        y_train = np.empty((0, np.shape(self.yy)[1]))
        y_test  = np.empty((0, np.shape(self.yy)[1]))
        
        ########======= Picking the train subjetcs and concatenate them =======########
        
        for c_subj in data_files_list[0:self.n_train]:
            
            self.tmp_name = c_subj
            
            # train hypnogram
            self.str_train_hyp  = 'hyp' + str(self.tmp_name)
            
            # train featureset
            self.str_train_feat = 'subject' + str(self.tmp_name)
            
            # create template arrays for featurs and label
            self.tmp_x          =  subjects_dic[self.str_train_feat]
            self.tmp_y          =  hyp_dic[self.str_train_hyp]
            
            # Concatenate features and labels
            X_train = np.row_stack((X_train, self.tmp_x))
            y_train = np.row_stack((y_train, self.tmp_y))
            
            #del self.tmp_x, self.tmp_y, self.tmp_name, self.str_train_hyp, self.str_train_feat
            
        ########======== Picking the test subjetcs and concatenate them =======########
        
        self.test_subjects_list = []
        
        for c_subj in data_files_list[self.n_train:]:
            
            self.tmp_name = c_subj
            # test hypnogram
            str_test_hyp  = 'hyp' + str(self.tmp_name)
            
            # test featureset
            str_test_feat = 'subject' + str(self.tmp_name)
            
            # create template arrays for featurs and  label
            self.tmp_x         =  subjects_dic[str_test_feat]
            self.tmp_y         =  hyp_dic[str_test_hyp]
            
            # Concatenate features and labels
            X_test = np.row_stack((X_test, self.tmp_x))
            y_test = np.row_stack((y_test, self.tmp_y))
            
            # keep the subject id
            self.test_subjects_list.append(str_test_feat)
            
            # remove for next iteration
            #del self.tmp_x, self.tmp_y,self.tmp_name, self.str_test_feat, self.str_test_hyp
        ################# FOR NOW WE IGNOR MOVEMENT AROUSAL ###################
        y_train = y_train[:,:5]
        y_test  = y_test[:,:5]
        
        # ========================= Time-dependency ========================= #
        global X_train_td, X_test_td
        if int(self.td_var.get()) == 1:
            print(f'it comes to many to one section, n_td : {self.entry_td.get()} ')
            X_train_td = self.Object.add_time_dependence_backward(X_train, n_time_dependence=int(self.entry_td.get()),padding_type = 'sequential')
    
            X_test_td  = self.Object.add_time_dependence_backward(X_test,  n_time_dependence=int(self.entry_td.get()),padding_type = 'sequential')
            
        # ======================== Feature selection ======================== #

        self.y_train_td = self.Object.binary_to_single_column_label(y_train)
        
        # Check activation of flag
        if int(self.feat_select_var.get()) == 1:
            
# =============================================================================
#             tmp1,tmp2,self.selected_feats_ind = self.Object.FeatSelect_Boruta(self.X_train_td, y_train, max_iter = 50)
#             X_train = self.X_train_td[:, self.selected_feats_ind]
#             X_test  = self.X_test_td[:, self.selected_feats_ind]
# =============================================================================
            
            tmp1,tmp2,self.selected_feats_ind = self.Object.FeatSelect_Boruta(X_train, self.y_train_td[:,0], max_iter = 50)
            X_train = X_train[:, self.selected_feats_ind]
            X_test  = X_test[:, self.selected_feats_ind]  
                
        ######## ================= Apply chosen ML ================= ##########
        global y_pred    
        # SVM
        if self.selected_ML == "SVM":
            
            y_pred = self.Object.KernelSVM_Modelling(X_train, y_train, X_test, y_test, kernel=self.kernel_.get())
            y_pred = np.expand_dims(y_pred, axis=1)
            # One hot encoding
            
            
        # Random forest    
        elif self.selected_ML == "Random forest":
            
            y_pred = self.Object.RandomForest_Modelling(X_train, y_train, X_test, y_test, n_estimators = int(self.entry_n_estimator_RF.get()))
            #y_pred = self.Object.RandomForest_Modelling(X_train, y_train, X_test, y_test, n_estimators = 10)
            self.Object.multi_label_confusion_matrix(y_test, y_pred, print_results = 'on')
        # XGB
        elif self.selected_ML == "XGBoost":
            y_pred = self.Object.XGB_Modelling(X_train, y_train,X_test, y_test, n_estimators = int(self.entry_n_estimator_xgb.get()), 
                      max_depth=3, learning_rate=.1)
        # ADABoost
        elif self.selected_ML == "ADABoost":
            y_pred = self.Object.ADAboost_Modelling(X_train, y_train,X_test, y_test, n_estimators = int(self.entry_n_estimator_ada.get()))
            
        # GRadient Boosting 
        elif self.selected_ML == "GradientBoosting":
            y_pred = self.Object.gradient_boosting_classifier(X_train, y_train,X_test, y_test, 
                                     n_estimators = int(self.entry_n_estimator_gb.get()), learning_rate= 1.0, max_depth=1)
        # Randomized trees
        elif self.selected_ML == "Randomized trees":
            y_pred = self.Object.Extra_randomized_trees(X_train, y_train, X_test,y_test, 
                                                   n_estimators= int(self.entry_n_estimator_rt.get()), 
                                                   max_depth = None, min_samples_split =2,
                                                   max_features="sqrt")

        
    #%% Def show_results function
    def show_results_function(self):
            
        from sklearn.metrics import multilabel_confusion_matrix, cohen_kappa_score

        #### =================== Create results window ================####
        self.results_win = Toplevel()
        self.results_win.title("Results of classification")
        
        # Label
        self.label_results_win = Label(self.results_win, text= "Results were found as below:\n", font = 'Calibri 16 bold')
        
        self.label_results_win.pack()
        
        self.close_res_win = Button(self.results_win, text="Dismiss", command=self.results_win.destroy)
        self.close_res_win.pack()
        
# =============================================================================
#         try: 
#             if np.shape(y_test)[1] != np.shape(y_pred)[1]:
#                 self.y_true = self.Object.binary_to_single_column_label(y_test)
#         except IndexError:
#             y_test = self.Object.binary_to_single_column_label(y_test)
# =============================================================================
            
        self.mcm = multilabel_confusion_matrix(y_test, y_pred)
        self.tn     = self.mcm[:, 0, 0]
        self.tp     = self.mcm[:, 1, 1]
        self.fn     = self.mcm[:, 1, 0]
        self.fp     = self.mcm[:, 0, 1]
        self.Recall = self.tp / (self.tp + self.fn)
        self.prec   = self.tp / (self.tp + self.fp)
        self.f1_sc  = 2 * self.Recall * self.prec / (self.Recall + self.prec)
        self.Acc = (self.tp + self.tn) / (self.tp + self.fp + self.fn+ self.tn)
        #kappa = cohen_kappa_score(y_true, y_pred)
        self.label_res1= Label(self.results_win, text = "Accuracy for Wake,N1,N2,N3,REM were respectively:" + str(self.Acc), font = 'Calibri 12 bold').pack()
        self.label_res2= Label(self.results_win, text = "Precision for Wake,N1,N2,N3,REM were respectively:" + str(self.prec), font = 'Calibri 12 bold').pack()
        self.label_res3= Label(self.results_win, text = "Recall for Wake,N1,N2,N3,REM were respectively:" + str(self.Recall),font = 'Calibri 12 bold').pack()
        self.label_res4= Label(self.results_win, text = "F1-score for Wake,N1,N2,N3,REM were respectively:" + str(self.f1_sc),font = 'Calibri 12 bold').pack()    
            
    #%% Plot confusion matrix
    def plot_conf_mat(self):
            
        self.Object.plot_confusion_matrix(y_test,y_pred, target_names = ['Wake','N1','N2','SWS','REM'],
                      title='Confusion matrix of ssccoorriinngg algorithm',
                      cmap=None,
                      normalize=True)
          
     #%% Plot hypnograms
    def plot_hyp_function(self):
        self.hyp_true = self.Object.binary_to_single_column_label(y_test)
        self.hyp_pred = self.Object.binary_to_single_column_label(y_pred)
        self.Object.plot_comparative_hyp(self.hyp_true, self.hyp_pred, mark_REM = 'active',
                             Title = 'True Hypnogram')
#%% Test section
root = Tk()
my_gui = Zzzscoring(root)
root.mainloop()
