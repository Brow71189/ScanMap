#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:27:57 2017

@author: mittelberger2
"""

# standard libraries
import gettext
import logging
import numpy as np
from .maptools import autotune as at
from importlib import reload
import threading
import time
import os

class SubSamplingPanelDelegate(object):


    def __init__(self, api):
        self.__api = api
        self.panel_id = 'SubSampling-Panel'
        self.panel_name = 'Sub-Sampling'
        self.panel_positions = ['left', 'right']
        self.panel_position = 'right'
        self.sub_samples = 2
        self.superscan = None

    def create_panel_widget(self, ui, document_controller):
        
        def sampling_finished(text):
            if len(text) > 0:
                try:
                    sub_samples = int(text)
                except ValueError:
                    pass
                else:
                    self.sub_samples = sub_samples
            sampling_field.text = '{:.0f}'.format(self.sub_samples)
        
        def record_button_clicked():
            if self.superscan is None:
                self.superscan = self.__api.get_hardware_source_by_id('superscan', '1')
            
            frame_parameters = self.superscan.get_record_frame_parameters()
            pixelsize = frame_parameters['fov_nm']/frame_parameters['size']
            sub_size = int(frame_parameters['size']/self.sub_samples)
            self.sub_samples = frame_parameters['size']/sub_size
            sampling_field.text = '{:.0f}'.format(self.sub_samples)
            result = np.zeros(frame_parameters['size'])
            result_data_item = document_controller.create_data_item_from_data(result, title='Sub-sampled (MAADF)')
            def record_image():
                for k in range(self.sub_samples):
                    for i in range(self.sub_samples):
                        sub_frame_parameters = frame_parameters.copy()
                        sub_frame_parameters['size'] = sub_size
                        sub_frame_parameters['center_nm'] = (k*pixelsize/self.sub_samples, i*pixelsize/self.sub_samples)
                        image = self.superscan.record(frame_parameters=sub_frame_parameters, channels_enabled=[False, True, False, False])[0]
                        result[k::self.sub_samples, i::self.sub_samples] = image.data
                        self.__api.queue_task(lambda: result_data_item.set_data(result))
                self.superscan.set_record_frame_parameters(frame_parameters)
            threading.Thread(target=record_image).start()
        
        sampling_label = ui.create_label_widget('Number sub-samples: ')
        sampling_field = ui.create_line_edit_widget()
        sampling_field.on_editing_finished = sampling_finished
        record_button = ui.create_push_button_widget('Record')
        record_button.on_clicked = record_button_clicked
        
        column = ui.create_column_widget()
        row1 = ui.create_row_widget()
        row2 = ui.create_row_widget()
        row1.add_spacing(10)
        row1.add(sampling_label)
        row1.add(sampling_field)
        row1.add_spacing(10)
        row1.add_stretch()
        row2.add_spacing(10)
        row2.add(record_button)
        row2.add_spacing(10)
        row2.add_stretch()
        
        column.add_spacing(10)
        column.add(row1)
        column.add_spacing(5)
        column.add(row2)
        column.add_spacing(10)
        column.add_stretch()

        return column

    def change_button_state(self, button, state):
        def do_change():
            button._PushButtonWidget__push_button_widget.enabled = state
        self.__api.queue_task(do_change)

    def change_label_text(self, label, text):
        def do_change():
            label.text = text
        self.__api.queue_task(do_change)

class SubSamplingExtension(object):
    extension_id = 'univie.subsampling'

    def __init__(self, api_broker):
        api = api_broker.get_api(version='1', ui_version='1')
        self.__panel_ref = api.create_panel(SubSamplingPanelDelegate(api))

    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None