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
        def measure_button_clicked():
            reload(at)
            self.T = at.Tuning()
            self.T.method = self.method_combo._widget.current_item
            self.T.document_controller = document_controller
            if self.superscan is None:
                self.superscan = self.__api.get_hardware_source_by_id('superscan', '1')
            if self.as2 is None:
                self.as2 = self.__api.get_instrument_by_id('autostem_controller', '1')
            self.T.superscan = self.superscan
            self.T.as2 = self.as2
            self.T.imsize = self.superscan.get_record_frame_parameters()['fov_nm']
            save_tuning = False
            if self.save_tuning_data_checkbox.checked:
                save_tuning = True
            def run_measure():
                try:
                    self.change_label_text(self.state_label, 'Measuring...')
                    self.change_button_state(self.find_focus_button, False)
                    self.change_button_state(self.correct_button, False)
                    stepsize = int(self.T.imsize/10)
                    if stepsize < 2:
                        stepsize = 2
                    elif stepsize > 10:
                        stepsize = 10
                    self.T.focus = self.T.find_focus(stepsize=stepsize, range=3*stepsize)[0][1]
                    self.as2.set_control_output('EHTFocus', -self.T.focus*1e-9, options={'inform': True, 'confirm': True})
                    self.C12 = self.T.measure_astig()
                    if self.C12 is not None:
                        self.as2.set_control_output('C12.u', self.C12[1]*1e-9, options={'inform': True, 'confirm': True})
                        self.as2.set_control_output('C12.v', self.C12[0]*1e-9, options={'inform': True, 'confirm': True})

                    focus_string = 'Measurement {:s}:\n C10\t{:.2f} nm\n'.format(time.strftime('%d-%m-%Y %H:%M'),
                                                                                 self.T.focus)
                    if self.C12 is not None:
                        C12a = self.as2.get_control_output('C12.a')*1e9
                        C12b = self.as2.get_control_output('C12.b')*1e9
                        self.C12 = (C12b, C12a)
                        astig_string = ' C12.a\t{:.2f} nm\n C12.b\t{:.2f} nm\n\n'.format(self.C12[1], self.C12[0])
                    else:
                        astig_string = ' No detectable astigmatism\n\n'
                    def insert_text():
                        self.result_widget.text = focus_string + astig_string + self.result_widget.text
                    self.__api.queue_task(insert_text)
                    if save_tuning:
                        path = os.path.dirname(__file__)
                        savedict = {}
                        if os.path.isfile(os.path.join(path, 'tuning_results.npz')):
                            with np.load(os.path.join(path, 'tuning_results.npz')) as npzfile:
                                for key, value in npzfile.items():
                                    savedict[key] = value
                        savedict[time.strftime('%Y%m%d-%Hh%M')] = np.array(self.T.analysis_results)
                        np.savez(os.path.join(path, 'tuning_results.npz'), **savedict)
                except:
                    self.change_label_text(self.state_label, 'Error')
                    raise
                else:
                    self.change_button_state(self.correct_button, True)
                    self.change_label_text(self.state_label, 'Done')
                finally:
                    self.change_button_state(self.find_focus_button, True)


            threading.Thread(target=run_measure).start()

        def correct_button_clicked():
            def run_correct():
                try:
                    self.change_label_text(self.state_label, 'Correcting...')
                    self.change_button_state(self.find_focus_button, False)
                    self.change_button_state(self.correct_button, False)
                    if self.C12 is not None:
#                        aberrations1 = {'EHTFocus': self.T.focus, 'C12_a': -self.C12[1], 'C12_b': -self.C12[0]}
#                        aberrations2 = {'EHTFocus': self.T.focus, 'C12_a': self.C12[1], 'C12_b': self.C12[0]}
#                        self.T.image = self.T.image_grabber(aberrations=aberrations1, reset_aberrations=True)[0]
#                        self.T.analyze_fft()
#                        tuning1 = np.sum(self.T.peaks)
#                        self.T.image = self.T.image_grabber(aberrations=aberrations2, reset_aberrations=True)[0]
#                        self.T.analyze_fft()
#                        tuning2 = np.sum(self.T.peaks)
#                        aberrations = aberrations1 if tuning1 > tuning2 else aberrations2
                        #aberrations = {'EHTFocus': self.T.focus, 'C12_a': self.C12[1], 'C12_b': self.C12[0]}
                        C12a_target = self.as2.get_control_output('^C12.a')
                        C12b_target = self.as2.get_control_output('^C12.b')
                        self.as2.set_control_output('C12.a', C12a_target)
                        self.as2.set_control_output('C12.b', C12b_target)
                    self.as2.set_control_output('EHTFocus', 0)
                    #else:
                        #aberrations = {'EHTFocus': self.T.focus}
                    #self.T.image_grabber(aberrations=aberrations, acquire_image=False)

                except:
                    self.change_label_text(self.state_label, 'Error')
                    raise
                else:
                    self.change_label_text(self.state_label, 'Done')
                finally:
                    self.change_button_state(self.find_focus_button, True)

            if self.T is not None and self.T.focus is not None:
                threading.Thread(target=run_correct).start()

#        def key_pressed(key):
#            print(key)
#
#        self.ui._UserInterface__ui.on_key_pressed = key_pressed
        column = ui.create_column_widget()
        #self.input_field.on_editing_finished = send_button_clicked
        self.find_focus_button = ui.create_push_button_widget('Measure')
        self.find_focus_button.on_clicked = measure_button_clicked
        self.method_label = ui.create_label_widget('Method: ')
        self.method_combo = ui.create_combo_box_widget()
        self.method_combo.items = ['general', 'graphene']
        self.result_widget = ui.create_text_edit_widget()
        self.result_widget._TextEditWidget__text_edit_widget.editable = False
        self.save_tuning_data_checkbox = ui.create_check_box_widget()
        self.save_tuning_data_label = ui.create_label_widget('Save tuning data')
        self.save_tuning_data_checkbox.checked = True
        self.correct_button = ui.create_push_button_widget('Correct')
        self.correct_button.on_clicked = correct_button_clicked
        self.correct_button._PushButtonWidget__push_button_widget.enabled = False
        self.state_label = ui.create_label_widget('')

        focus_row = ui.create_row_widget()
        correct_row = ui.create_row_widget()
        method_row = ui.create_row_widget()
        result_row = ui.create_row_widget()
        focus_row.add_spacing(5)
        focus_row.add(self.find_focus_button)
        focus_row.add_spacing(10)
        focus_row.add(self.save_tuning_data_checkbox)
        focus_row.add_spacing(2)
        focus_row.add(self.save_tuning_data_label)
        focus_row.add_stretch()
        column.add_spacing(10)
        column.add(focus_row)
        column.add_spacing(10)
        method_row.add_spacing(5)
        method_row.add_stretch()
        method_row.add(self.method_label)
        method_row.add(self.method_combo)
        method_row.add_spacing(10)
        column.add_spacing(10)
        column.add(method_row)
        column.add_spacing(10)
        correct_row.add_spacing(5)
        correct_row.add(self.correct_button)
        correct_row.add_spacing(15)
        correct_row.add(self.state_label)
        correct_row.add_stretch()
        column.add(correct_row)
        column.add_spacing(15)
        column.add(result_row)
        result_row.add_spacing(5)
        result_row.add(self.result_widget)
        result_row.add_spacing(10)
        column.add_spacing(10)
        #column.add_stretch()

        return column

    def change_button_state(self, button, state):
        def do_change():
            button._PushButtonWidget__push_button_widget.enabled = state
        self.__api.queue_task(do_change)

    def change_label_text(self, label, text):
        def do_change():
            label.text = text
        self.__api.queue_task(do_change)

class AnalyzeFFTExtension(object):
    extension_id = 'univie.analyzefft'

    def __init__(self, api_broker):
        api = api_broker.get_api(version='1', ui_version='1')
        self.__panel_ref = api.create_panel(AnalyzeFFTPanelDelegate(api))

    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None