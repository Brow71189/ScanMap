# standard libraries
import gettext
import logging
import numpy as np
from .maptools import autotune as at
from importlib import reload
import threading
import time
import os

_ = gettext.gettext
# _HardwareSource_hardware_source.acquire_data_elements()
class AnalyzeFFTPanelDelegate(object):


    def __init__(self, api):
        self.__api = api
        self.panel_id = 'AnalyzeFFT-Panel'
        self.panel_name = _('AnalyzeFFT')
        self.panel_positions = ['left', 'right']
        self.panel_position = 'right'
        self.T = None
        self.superscan = None
        self.as2 = None
        self.C12 = None

    def create_panel_widget(self, ui, document_controller):
        def find_focus_button_clicked():
            reload(at)
            self.T = at.Tuning()
            self.T.document_controller = document_controller
            if self.superscan is None:
                self.superscan = self.__api.get_hardware_source_by_id('superscan', '1')
            if self.as2 is None:
                self.as2 = self.__api.get_instrument_by_id('autostem_controller', '1')
            self.T.superscan = self.superscan
            self.T.as2 = self.as2
            self.T.imsize = self.superscan.get_record_frame_parameters()['fov_nm']
            def run_find_focus():
                self.T.focus = self.T.find_focus(method='general')[0][1]
                self.change_button_state(self.measure_astig_button, True)
                focus_string = 'Measurement {:s}:\n C10\t{:.2f} nm\n'.format(time.strftime('%m-%d %H:%M:%S'),
                                                                             self.T.focus)
                self.api.queue_task(lambda: self.result_widget.insert_text(focus_string))
                if self.save_tuning_data_checkbox.checked:
                    path = os.path.dirname(__file__)
                    savedict = {}
                    if os.path.isfile(os.path.join(path, 'tuning_results.npz')):
                        with np.load(os.path.join(path, 'tuning_results.npz')) as npzfile:
                            for key, value in npzfile.items():
                                savedict[key] = value
                    savedict[time.strftime('%Y%m%d-%Hh%M')] = np.array(self.T.analysis_results)
                    np.savez(os.path.join(path, 'tuning_results.npz'), **savedict)
                        
                    
            threading.Thread(target=run_find_focus).start()

        def measure_astig_button_clicked():
            def run_measure_astig():
                self.C12 = self.T.measure_astig(method='general')
                self.change_button_state(self.correct_button, True)
                astig_string = ' C12.a\t{:.2f} nm\n C12.b\t{:.2f} nm\n\n'.format(self.C12[1], self.C12[0])
                self.api.queue_task(lambda: self.result_widget.insert_text(astig_string))
            threading.Thread(target=run_measure_astig).start()
            
        def correct_button_clicked():
            if self.C12 is not None:
                aberrations = {'EHTFocus': self.T.focus, 'C12_a': -self.C12[1], 'C12_b': -self.C12[0]}
                self.T.image_grabber(aberrations=aberrations, acquire_image=False)
                self.change_button_state(self.correct_button, False)
                

#        def key_pressed(key):
#            print(key)
#
#        self.ui._UserInterface__ui.on_key_pressed = key_pressed
        column = ui.create_column_widget()
        #self.input_field.on_editing_finished = send_button_clicked
        self.find_focus_button = ui.create_push_button_widget('Find Focus')
        self.find_focus_button.on_clicked = find_focus_button_clicked
        self.measure_astig_button = ui.create_push_button_widget('Measure Astig')
        self.measure_astig_button.on_clicked = measure_astig_button_clicked
        self.measure_astig_button._PushButtonWidget__push_button_widget.enabled = False
        self.result_widget = ui.create_text_edit_widget()
        self.result_widget._TextEditWidget__text_edit_widget.editable = False
        self.save_tuning_data_checkbox = ui.create_check_box_widget()
        self.save_tuning_data_label = ui.create_label_widget('Save tuning data')
        self.save_tuning_data_checkbox.checked = True
        self.correct_button = ui.create_push_button_widget('Correct')
        self.correct_button._PushButtonWidget__push_button_widget.enabled = False
        
        focus_row = ui.create_row_widget()
        astig_row = ui.create_row_widget()
        correct_row = ui.create_row_widget()
        result_row = ui.create_row_widget()
        focus_row.add(self.find_focus_button)
        focus_row.add_spacing(10)
        focus_row.add(self.save_tuning_data_checkbox)
        focus_row.add_spacing(2)
        focus_row.add(self.save_tuning_data_label)
        focus_row.add_stretch()
        column.add_spacing(10)
        column.add(focus_row)
        column.add_spacing(10)
        astig_row.add(self.measure_astig_button)
        astig_row.add_stretch()
        column.add(astig_row)
        column.add_spacing(10)
        correct_row.add(self.correct_button)
        correct_row.add_stretch()
        column.add(correct_row)
        column.add_spacing(15)     
        column.add(result_row)
        result_row.add(self.result_widget)
        result_row.add_spacing(5)
        column.add_stretch()

        return column
    
    def change_button_state(self, button, state):
        def do_change():
            button._PushButtonWidget__push_button_widget.enabled = state
        self.__api.queue_task(do_change)

class AnalyzeFFTExtension(object):
    extension_id = 'univie.analyzefft'

    def __init__(self, api_broker):
        api = api_broker.get_api(version='1', ui_version='1')
        self.__panel_ref = api.create_panel(AnalyzeFFTPanelDelegate(api))

    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None