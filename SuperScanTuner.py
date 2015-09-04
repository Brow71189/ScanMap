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

try:
    from importlib import reload
except:
    pass

from .maptools import autotune

_ = gettext.gettext

focus_step=2
astig2f_step=2
astig3f_step=150
coma_step=300
average_frames=3
integration_radius=1
dirt_threshold = 0.015
save_images=False
savepath=None

class SuperScanTunerPanelDelegate(object):
    def __init__(self, api):
        self.__api = api
        self.panel_id = 'SuperScanTuner-Panel'
        self.panel_name = _('SuperScanTuner')
        self.panel_positions = ['left', 'right']
        self.panel_position = 'right'
        
    def create_panel_widget(self, ui, document_controller):
        
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
                astig3f_step = 150
                
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
        
        def dirt_finished(text):
            global dirt_threshold
            if len(text)>0:
                try:
                    dirt_threshold = float(text)
                except:
                    logging.warn(text+' is not a valid number. Please input floating point number.')
            else:
                dirt_threshold = 0.015
                
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
            global focus_step, astig2f_step, astig3f_step, coma_step, average_frames, integration_radius
            global dirt_threshold, save_images, savepath
            
            
            superscan = self.__api.get_hardware_source_by_id('scan_controller', '1')
            as2 = self.__api.get_instrument_by_id('autostem_controller', '1')
            
            
            keys = []
            
            reload(autotune)

            if EHTFocus.check_state == 'checked':
                keys.append('EHTFocus')
            if Twofold.check_state == 'checked':
                keys.append('C12_a')
                keys.append('C12_b')
            if Coma.check_state == 'checked':
                keys.append('C21_a')
            if Threefold.check_state == 'checked':
                keys.append('C23_a')
            if Coma.check_state == 'checked':
                keys.append('C21_b')
            if Threefold.check_state == 'checked':
                keys.append('C23_b')
            
            if len(keys) < 1:
                logging.warn('Tuning not started because no aberrations are selected for the tuning.')
                return

            if saving.check_state == 'checked':
                if savepath is not None:
                    save_images = True
                else:
                    logging.warn('You have to enter a valid path in \"savepath\" if you want to save ' +
                                 'the images acquired during tuning.')
                    return
            else:
                save_images = False
                
            
            steps = {'EHTFocus': focus_step, 'C12_a': astig2f_step, 'C12_b': astig2f_step, 'C21_a': coma_step,
                     'C21_b': coma_step, 'C23_a': astig3f_step, 'C23_b': astig3f_step}

            logging.info('Started tuning.')
            
            self.event = threading.Event()
            #self.thread = threading.Thread(target=do_something, args=(self.event, document_controller))
            self.thread = threading.Thread(target=autotune.kill_aberrations,
                                           kwargs={'steps': steps,
                                                   'average_frames': average_frames,
                                                   'integration_radius': integration_radius,
                                                   'save_images': save_images,
                                                   'savepath': savepath, 
                                                   'document_controller': document_controller,
                                                   'event': self.event,
                                                   'keys': keys,
                                                   'dirt_threshold': dirt_threshold,
                                                   'superscan': superscan,
                                                   'as2': as2})
                                
            self.thread.start()
            
            start_button.visible = False
            abort_button.visible = True
        
        def abort_button_clicked():
            logging.info('Aborting tuning after current aberration. (May take a short while until actual abort)')
            self.event.set()
            self.thread.join()
            logging.info('Finished')
            abort_button.visible = False
            start_button.visible = True
        
        descriptor_row1 = ui.create_row_widget()
        
        descriptor_row1.add(ui.create_label_widget(_("Define tuning stepsizes here (default values are used for empty fields):")))
        
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
        C12.on_editing_finished = C12_finished
        
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
        C23.placeholder_text = "Defaults to 150"
        parameters_row2.add(C23)
        parameters_row2.add(ui.create_label_widget(_("nm")))
        C23.on_editing_finished = C23_finished
        
        descriptor_row2 = ui.create_row_widget()
        
        descriptor_row2.add(ui.create_label_widget(_("Additional parameters for tuning procedure:")))
        
        parameters_row3 = ui.create_row_widget()
        parameters_row3.add(ui.create_label_widget(_("Average images: ")))
        number_average = ui.create_line_edit_widget()
        number_average.placeholder_text = "Defaults to 3"
        parameters_row3.add(number_average)
        number_average.on_editing_finished = average_finished
        parameters_row3.add_spacing(15)
        
        parameters_row3.add(ui.create_label_widget(_("Integration radius: ")))
        integration = ui.create_line_edit_widget()
        integration.placeholder_text = "Defaults to 1"
        parameters_row3.add(integration)
        integration.on_editing_finished = integration_finished
        parameters_row3.add(ui.create_label_widget(_("px")))
        
        parameters_row4 = ui.create_row_widget()
        parameters_row4.add(ui.create_label_widget(_("Dirt threshold: ")))
        dirt = ui.create_line_edit_widget()
        dirt.placeholder_text = "Defaults to 0.015"
        parameters_row4.add(dirt)
        dirt.on_editing_finished = dirt_finished
        parameters_row4.add_spacing(250)
        
        descriptor_row3 = ui.create_row_widget()
        descriptor_row3.add(ui.create_label_widget(_("Check all aberrations you want to include in the auto-tuning:")))
        
        checkbox_row1 = ui.create_row_widget()
        EHTFocus = ui.create_check_box_widget(_("EHTFocus"))
        checkbox_row1.add(EHTFocus)
        checkbox_row1.add_spacing(4)
        Twofold = ui.create_check_box_widget(_("Twofold Astig"))
        checkbox_row1.add(Twofold)
        checkbox_row1.add_spacing(4)
        Coma = ui.create_check_box_widget(_("Coma"))
        checkbox_row1.add(Coma)
        checkbox_row1.add_spacing(4)
        Threefold = ui.create_check_box_widget(_("Threefold Astig"))
        checkbox_row1.add(Threefold)
        EHTFocus.check_state = 'checked'
        Twofold.check_state = 'checked'
        Coma.check_state = 'checked'
        Threefold.check_state = 'checked'
        
        checkbox_row2 = ui.create_row_widget()
        saving = ui.create_check_box_widget(_("Save all images during tuning"))
        checkbox_row2.add(saving)
        
        parameters_row5 = ui.create_row_widget()

        parameters_row5.add(ui.create_label_widget(_("Savepath: ")))
        saving_path = ui.create_line_edit_widget()        
        saving_path.placeholder_text = "If save images is checked you have to enter a path here."
        parameters_row5.add(saving_path)
        saving_path.on_editing_finished = saving_finished
        
        button_row = ui.create_row_widget()
        start_button = ui.create_push_button_widget(_("Start tuning"))
        start_button.on_clicked = start_button_clicked
        button_row.add(start_button)
        abort_button = ui.create_push_button_widget(_("Abort tuning"))
        abort_button.on_clicked = abort_button_clicked
        button_row.add(abort_button)
        
        abort_button.visible = False
        
        column.add_spacing(15)
        column.add(descriptor_row1)
        column.add_spacing(3)
        column.add(parameters_row1)
        column.add(parameters_row2)
        column.add_spacing(15)
        column.add(descriptor_row2)
        column.add_spacing(3)
        column.add(parameters_row3)
        column.add(parameters_row4)
        column.add_spacing(15)
        column.add(descriptor_row3)
        column.add_spacing(3)
        column.add(checkbox_row1)
        column.add_spacing(15)
        column.add(checkbox_row2)
        column.add(parameters_row5)
        column.add_spacing(25)
        column.add(button_row)
        column.add_stretch()
        
        return column
        
class SuperScanTunerExtension(object):
    extension_id = 'univie.superscantuner'
    
    def __init__(self, api_broker):
        api = api_broker.get_api(version='1', ui_version='1')
        self.__panel_ref = api.create_panel(SuperScanTunerPanelDelegate(api))
    
    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None
    
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