# standard libraries
import gettext
import logging
import numpy as np
from .maptools import autotune as at
from importlib import reload
import threading

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

    def create_panel_widget(self, ui, document_controller):
        def find_focus_button_clicked():
            reload(at)
            self.T = at.Tuning()
            self.T.document_controller = document_controller
            superscan = self.__api.get_hardware_source_by_id('superscan', '1')
            as2 = self.__api.get_instrument_by_id('autostem_controller', '1')
            self.T.superscan = superscan
            self.T.as2 = as2
            self.T.imsize = superscan.get_record_frame_parameters()['fov_nm']
            def run_find_focus():
                self.T.focus = self.T.find_focus(method='general')[0][1]
            threading.Thread(target=run_find_focus).start()

        def measure_astig_button_clicked():
            if self.T is None:
                reload(at)
                self.T = at.Tuning()
                self.T.document_controller = document_controller
                superscan = self.__api.get_hardware_source_by_id('superscan', '1')
                as2 = self.__api.get_instrument_by_id('autostem_controller', '1')
                self.T.superscan = superscan
                self.T.as2 = as2
                self.T.imsize = superscan.get_record_frame_parametesr()['fov_nm']
            threading.Thread(target=self.T.measure_astig, kwargs={'method': 'general'}).start()

#        def key_pressed(key):
#            print(key)
#
#        self.ui._UserInterface__ui.on_key_pressed = key_pressed
        column = ui.create_column_widget()
        #self.input_field.on_editing_finished = send_button_clicked
        find_focus_button = ui.create_push_button_widget('Find Focus')
        find_focus_button.on_clicked = find_focus_button_clicked
        measure_astig_button = ui.create_push_button_widget('Measure Astig')
        measure_astig_button.on_clicked = measure_astig_button_clicked

        column.add(find_focus_button)
        column.add_spacing(10)
        column.add(measure_astig_button)
        column.add_stretch()

        return column

class AnalyzeFFTExtension(object):
    extension_id = 'univie.analyzefft'

    def __init__(self, api_broker):
        api = api_broker.get_api(version='1', ui_version='1')
        self.__panel_ref = api.create_panel(AnalyzeFFTPanelDelegate(api))

    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None