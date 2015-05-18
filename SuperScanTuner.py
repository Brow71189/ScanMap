# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:51:41 2015

@author: mittelberger
"""
import gettext
import logging
import os
import threading
import time

from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import DataItem
from nion.ui import Binding

import maptools.autotune as autotune

_ = gettext.gettext

focus_step=2
astig2f_step=2
astig3f_step=75
coma_step=300
average_frames=3
integration_radius=1
only_focus=False
save_images=False
savepath=None

class SuperScanTuner(Panel.Panel):
    def __init__(self, document_controller, panel_id, properties):
        super(SuperScanTuner, self).__init__(document_controller, panel_id, "SSTuner")

        ui = document_controller.ui

        # user interface

        column = ui.create_column_widget()
        
        def Focus_finished(text):
            global focus_step
            if len(text)>0:
                try:
                    focus_step = float(text)
                except:
                    logging.warn(text+' is not a valid stepsize. Please input a floating point number.')
            else:
                focus_step = 2
                
        def C12_finished(text):
            global astig2f_step
            if len(text)>0:
                try:
                    astig2f_step = float(text)
                except:
                    logging.warn(text+' is not a valid stepsize. Please input a floating point number.')
            else:
                astig2f_step = 2
                
        def C21_finished(text):
            global coma_step
            if len(text)>0:
                try:
                    coma_step = float(text)
                except:
                    logging.warn(text+' is not a valid stepsize. Please input a floating point number.')
            else:
                coma_step = 300
                
        def C23_finished(text):
            global astig3f_step
            if len(text)>0:
                try:
                    astig3f_step = float(text)
                except:
                    logging.warn(text+' is not a valid stepsize. Please input a floating point number.')
            else:
                astig3f_step = 75
                
        def average_finished(text):
            global average_frames
            if len(text)>0:
                try:
                    average_frames = int(text)
                except:
                    logging.warn(text+' is not a valid number. Please input an integer.')
            else:
                average_frames = 3
        
        def integration_finished(text):
            global integration_radius
            if len(text)>0:
                try:
                    integration_radius = int(text)
                except:
                    logging.warn(text+' is not a valid number. Please input an integer.')
            else:
                integration_radius = 1
                
        def saving_finished(text):
            global savepath
            if len(text)>0:
                if os.path.isabs(text):
                    savepath = text
                else:
                    logging.warn(text+' is not an absolute path. Please enter a complete pathname starting from root.')
            else:
                savepath = None
                
        def start_button_clicked():
            global focus_step, astig2f_step, astig3f_step, coma_step, average_frames, integration_radius, only_focus, save_images, savepath
            
            reload(autotune)

            if focus_only.check_state == 'checked':
                only_focus = True
            else:
                only_focus = False

            if saving.check_state == 'checked':
                if savepath is not None:
                    save_images = True
                else:
                    logging.warn('You have to enter a valid path in \"savepath\" if you want to save the images acquired during tuning.')
                    return
            else:
                save_images = False

            logging.info('Started tuning.')
            self.event = threading.Event()
            self.thread = threading.Thread(target=do_something, args=(self.event, document_controller))
#            self.thread = threading.Thread(target=autotune.kill_aberrations, kwargs={'focus_step': focus_step, 'astig2f_step': astig2f_step, 'astig3f_step': astig3f_step,\
#                                'coma_step': coma_step, 'average_frames': average_frames, 'integration_radius': integration_radius, 'only_focus': only_focus, \
#                                'save_images': save_images, 'savepath': savepath, 'document_controller': document_controller, 'event': self.event})
            self.thread.start()
        
        def abort_button_clicked():
            #self.stop_tuning()
            logging.info('Aborting tuning after current aberration. (May take a short while until actual abort)')
            self.event.set()
            self.thread.join()
        
        descriptor_row1 = ui.create_row_widget()
        
        descriptor_row1.add(ui.create_label_widget(_("Define the stepsizes for the tuning here. Leave fields empty to use default values.")))
        
        parameters_row1 = ui.create_row_widget()
        
        parameters_row1.add(ui.create_label_widget(_("EHTFocus: ")))
        EHTFocus = ui.create_line_edit_widget()
        EHTFocus.placeholder_text = "Defaults to 2"
        parameters_row1.add(EHTFocus)
        parameters_row1.add(ui.create_label_widget(_("nm")))
        EHTFocus.on_editing_finished = Focus_finished
        
        parameters_row1.add_spacing(15)
        
        parameters_row1.add(ui.create_label_widget(_("C12 (Astig 2f): ")))
        C12 = ui.create_line_edit_widget()
        C12.placeholder_text = "Defaults to 2"
        parameters_row1.add(C12)
        parameters_row1.add(ui.create_label_widget(_("nm")))
        C12.on_editing_finfished = C12_finished
        
        parameters_row2 = ui.create_row_widget()
        
        parameters_row2.add(ui.create_label_widget(_("C21 (Coma): ")))
        C21 = ui.create_line_edit_widget()
        C21.placeholder_text = "Defaults to 300"
        parameters_row2.add(C21)
        parameters_row2.add(ui.create_label_widget(_("nm")))
        C21.on_editing_finished = C21_finished
        
        parameters_row2.add_spacing(15)
        
        parameters_row2.add(ui.create_label_widget(_("C23 (Astig 3f): ")))
        C23 = ui.create_line_edit_widget()
        C23.placeholder_text = "Defaults to 75"
        parameters_row2.add(C23)
        parameters_row2.add(ui.create_label_widget(_("nm")))
        C23.on_editing_finished = C23_finished
        
        descriptor_row2 = ui.create_row_widget()
        
        descriptor_row2.add(ui.create_label_widget(_("Additional parameters for tuning procedure")))
        
        parameters_row3 = ui.create_row_widget()
        parameters_row3.add(ui.create_label_widget(_("Average images: ")))
        number_average = ui.create_line_edit_widget()
        number_average.placeholder_text = "Defaults to 3"
        parameters_row3.add(number_average)
        parameters_row3.add_spacing(15)
        
        parameters_row3.add(ui.create_label_widget(_("Integration radius: ")))
        integration = ui.create_line_edit_widget()
        integration.placeholder_text = "Defaults to 1"
        parameters_row3.add(integration)
        parameters_row3.add(ui.create_label_widget(_("px")))
        
        checkbox_row = ui.create_row_widget()
        focus_only = ui.create_check_box_widget(_("Adjust focus only"))
        checkbox_row.add(focus_only)
        checkbox_row.add_spacing(15)
        saving = ui.create_check_box_widget(_("Save all images during tuning"))
        checkbox_row.add(saving)
        
        parameters_row4 = ui.create_row_widget()
        
        parameters_row4.add(ui.create_label_widget(_("Savepath: ")))
        saving_path = ui.create_line_edit_widget()        
        saving_path.placeholder_text = "If save images is checked you have to enter a path here."
        parameters_row4.add(saving_path)
        saving_path.on_editing_finished = saving_finished
        
        button_row = ui.create_row_widget()
        start_button = ui.create_push_button_widget(_("Start tuning"))
        start_button.on_clicked = start_button_clicked
        button_row.add(start_button)
        button_row.add_spacing(15)
        abort_button = ui.create_push_button_widget(_("Abort tuning"))
        abort_button.on_clicked = abort_button_clicked
        button_row.add(abort_button)
        
        column.add_spacing(15)
        column.add(descriptor_row1)
        column.add_spacing(5)
        column.add(parameters_row1)
        column.add(parameters_row2)
        column.add_spacing(15)
        column.add(descriptor_row2)
        column.add_spacing(5)
        column.add(parameters_row3)
        column.add_spacing(15)
        column.add(checkbox_row)
        column.add(parameters_row4)
        column.add_spacing(25)
        column.add(button_row)

        self.widget = column
    
def do_something(event, document_controller):
    counter = 0
    while True:
        if not event.is_set() and counter < 10:
            #logging.info('Still working')
            document_controller.queue_main_thread_task(lambda: logging.info('Still working'))
            #panel.queue_task(lambda: logging.info('Still working'))
            time.sleep(2)
        else:
            #logging.info('Finished')
            document_controller.queue_main_thread_task(lambda: logging.info('Still working'))
            break
        counter +=1
    
    return 'Finished'
  
workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(SuperScanTuner, "tuner-panel", _("SuperScan Tuning"), ["left", "right"], "right" )
