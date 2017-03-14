# standard libraries
import gettext
import logging
import numpy as np
import os
import time
import copy

try:
    from importlib import reload
except:
    pass

from .maptools import mapper, autotune

_ = gettext.gettext

class ScanMapPanelDelegate(object):

    def __init__(self, api):
        self.__api = api
        self.panel_id = 'ScanMap-Panel'
        self.panel_name = _('ScanMap')
        self.panel_positions = ['left', 'right']
        self.panel_position = 'right'
        self.superscan = None
        self.as2 = None
        self.ccd = None
        self.coord_dict = {'top-left': None, 'top-right': None, 'bottom-right': None, 'bottom-left': None}
#        self.switches = {'do_retuning': False, 'use_z_drive': False, 'abort_series_on_dirt': False,
#                         'compensate_stage_error': False, 'acquire_overview': True, 'blank_beam': False,
#                         'isotope_mapping': False, 'show_last_frames_average': False, 'aligned_average': False}
#        self.frame_parameters = {'size_pixels': (2048, 2048), 'pixeltime': 0.2, 'fov': 20, 'rotation': 114.5}
#        self.isotope_frame_parameters = {'size_pixels': (512, 512), 'pixeltime': 1, 'fov': 3}
#        self.isotope_mapping_settings = {'overlap': 0.1, 'max_number_frames': 100, 'intensity_threshold': 0.8}
#        self.offset = 1
#        self.number_of_images = 1
#        self.dirt_area = 0.5
#        self.average_number = 1
#        self.max_align_dist = 0.01
#        self.sleeptime = 2
#        self.peak_intensity_reference = None
#        self.savepath = 'Z:/ScanMap/'
#        self.event = None
#        self.tune_event = None
#        self.abort_series_event = None
#        self.tune_now_event = None
#        self.thread = None
        self.thread_communication = None
#        self.retuning_mode = ['edges','manual']
        # Is filled later with the actual checkboxes. For now just the default values are stored
#        self._checkboxes = {'do_retuning': False, 'use_z_drive': False, 'abort_series_on_dirt': False,
#                            'compensate_stage_error': False, 'acquire_overview': True, 'blank_beam': False,
#                            'isotope_mapping': False, 'show_last_frames_average': False, 'aligned_average': False}
        self._checkboxes = {}
        self._buttons = {}
        self._dropdowns = {}
        self._text_fields = {}
        self.sync_gui_working = False
        self.Mapper = None
        self.waiting_for_focus = False


    def create_panel_widget(self, ui, document_controller):

        self.superscan = self.__api.get_hardware_source_by_id('scan_controller', '1')
        self.as2 = self.__api.get_instrument_by_id('autostem_controller', '1')
        self.ccd = self.__api.get_hardware_source_by_id('nionccd1010', '1')
        self.document_controller = document_controller

        self.Mapper = mapper.SuperScanMapper(superscan=self.superscan, as2=self.as2,
                                             document_controller=document_controller)
        self.Mapper.on_low_level_event_occured = self.low_level_event_occured

        column = ui.create_column_widget()

        def fov_finished(text):
            if len(text) > 0:
                try:
                    fov = float(text)
                except ValueError:
                    pass
                else:
                    self.Mapper.frame_parameters['fov'] = fov

                self.total_number_frames()

            fov_line_edit.text = '{:.1f}'.format(self.Mapper.frame_parameters['fov'])

        def size_finished(text):
            if len(text) > 0:
                try:
                    size = int(text)
                except ValueError:
                    pass
                else:
                    self.Mapper.frame_parameters['size_pixels'] = (size, size)

                self.total_number_frames()

            size_line_edit.text = '{:.0f}'.format(self.Mapper.frame_parameters['size_pixels'][0])

        def offset_finished(text):
            if len(text) > 0:
                try:
                    offset = float(text)
                except ValueError:
                    pass
                else:
                    self.Mapper.offset = offset

                self.total_number_frames()

            offset_line_edit.text = '{:.0f}'.format(self.Mapper.offset)

        def time_finished(text):
            if len(text) > 0:
                try:
                    pixeltime = float(text)
                except ValueError:
                    pass
                else:
                    self.Mapper.frame_parameters['pixeltime'] = pixeltime

                self.total_number_frames()

            time_line_edit.text = '{:.1f}'.format(self.Mapper.frame_parameters['pixeltime'])

        def rotation_finished(text):
            if len(text) > 0:
                try:
                    rotation = float(text)
                except ValueError:
                    pass
                else:
                    self.Mapper.frame_parameters['rotation'] = rotation

            rotation_line_edit.text = '{:.1f}'.format(self.Mapper.frame_parameters['rotation'])

        def number_of_images_finished(text):
            if len(text) > 0:
                try:
                    number_of_images = int(text)
                except ValueError:
                    pass
                else:
                    self.Mapper.number_of_images = number_of_images

                self.total_number_frames()

            number_line_edit.text = '{:.0f}'.format(self.Mapper.number_of_images)

        def sleeptime_finished(text):
            if len(text) > 0:
                try:
                    sleeptime = float(text)
                except ValueError:
                    pass
                else:
                    self.Mapper.sleeptime = sleeptime

                self.total_number_frames()

            sleeptime_line_edit.text = '{:.1f}'.format(self.Mapper.sleeptime)

        def dirt_area_finished(text):
            if len(text) > 0:
                try:
                    dirt_area = float(text)/100
                except ValueError:
                    pass
                else:
                    self.Mapper.dirt_area = dirt_area

            dirt_area_line_edit.text = '{:.0f}'.format(self.Mapper.dirt_area*100)

        def saving_finished(text):
            if len(text) > 0:
                if os.path.isabs(text):
                    self.Mapper.savepath = text
                else:
                    logging.warn(text+' is not an absolute path. Please enter a complete pathname starting from root.')

            savepath_line_edit.text = self.Mapper.savepath

        def average_number_finished(text):
            if len(text) > 0:
                try:
                    average_number = int(text)
                except ValueError:
                    pass
                else:
                    self.Mapper.average_number = average_number

            average_number_line_edit.text = '{:.0f}'.format(self.Mapper.average_number)



        def max_align_dist_finished(text):
            if len(text) > 0:
                try:
                    max_align_dist = float(text)
                except ValueError:
                    pass
                else:
                    self.Mapper.max_align_dist = max_align_dist

            max_align_dist_line_edit.text = '{:.3f}'.format(self.Mapper.max_align_dist)

        def checkbox_changed(check_state):
            if not self.sync_gui_working:
                for key, value in self._checkboxes.items():
                    self.Mapper.switches[key] = value.checked

        def tl_button_clicked():
            self.save_coords('top-left')
            self.total_number_frames()
        def tr_button_clicked():
            self.save_coords('top-right')
            self.total_number_frames()
        def bl_button_clicked():
            self.save_coords('bottom-left')
            self.total_number_frames()
        def br_button_clicked():
            self.save_coords('bottom-right')
            self.total_number_frames()
        def drive_tl_button_clicked():
            self.drive_coords('top-left')
        def drive_tr_button_clicked():
            self.drive_coords('top-right')
        def drive_bl_button_clicked():
            self.drive_coords('bottom-left')
        def drive_br_button_clicked():
            self.drive_coords('bottom-right')

        def method_combo_box_changed(item):
            self.Mapper.retuning_mode[0] = item.replace(' ', '_')
        def mode_combo_box_changed(item):
            self.Mapper.retuning_mode[1] = item.replace(' ', '_')

        def browse_button_clicked():
            existing_directory, directory = self.document_controller._document_controller.ui.get_existing_directory_dialog('Select the savepath', self.Mapper.savepath)
            if len(existing_directory) > 0:
                self.Mapper.savepath = os.path.normpath(existing_directory)
                sync_gui()

        def save_button_clicked():
            self.Mapper.coord_dict = self.coord_dict.copy()
            self.Mapper.save_mapping_config()
            logging.info('Saved config file to: ' + os.path.join(self.Mapper.savepath, self.Mapper.foldername,
                                                                 'configs_map.txt'))

        def load_button_clicked():
            configfilepath = self.Mapper.savepath
            if not os.path.isfile(configfilepath):
                configfilepath = os.path.join(configfilepath, 'configs_map.txt')

            if not os.path.isfile(configfilepath):
                logging.warn('Please type the path to the config file into the \'savepath\' field to load configs.')
            else:
                self.Mapper.load_mapping_config(configfilepath)
                self.coord_dict = self.Mapper.coord_dict.copy()
                sync_gui()
                logging.info('Loaded all mapping configs successfully.')

        def test_button_clicked():
            if self.Mapper is not None and self.Mapper.is_running:
                self.Mapper.handle_retuning()
            else:
                if None in self.frame_parameters.values():
                    logging.warn('You must specify all scan parameters (e.g. FOV, framesize, rotation, pixeltime) ' +
                                 'before acquiring a test image.')
                    return
                reload(autotune)
                Image = autotune.Imaging(frame_parameters=self.frame_parameters, as2=self.as2, superscan=self.superscan,
                                         document_controller=document_controller)
                testimage = Image.image_grabber()[0]
                di=self.__api.library.create_data_item_from_data(testimage, 'testimage_'+ time.strftime('%Y_%m_%d_%H_%M'))
                calibration = self.__api.create_calibration(scale=self.frame_parameters['fov']/
                                                           self.frame_parameters['size_pixels'][0],
                                                           units='nm')
                di.set_dimensional_calibrations([calibration, calibration])

        def done_button_clicked():

#            if self.thread is not None and self.thread.is_alive():
#            self.low_level_event_occured('map_started')
#            return
            if self.Mapper.is_running:
#                if self.tune_event is not None and self.tune_event.is_set():
                if self.waiting_for_focus:
                    self.thread_communication['new_EHTFocus'] = self.as2.get_property_as_float('EHTFocus')
                    self.thread_communication['new_z'] = self.as2.get_property_as_float('StageOutZ')
                    self.Mapper.tune_event.clear()
                    return
                else:
                    logging.warn('There is already a mapping going on. Please abort it or wait for it to terminate.')
                    return

#            saving_finished(savepath_line_edit.text)
#            checkbox_changed('obligatory string argument')
#            self.total_number_frames()

#            try:
#                reload(mapper)
#                reload(autotune)
#            except:
#                logging.warn('Couldn\'t reload mapper and autotune.')

#            if None in self.frame_parameters.values():
#                logging.warn('You must specify all scan parameters (e.g. FOV, framesize, rotation, pixeltime) ' +
#                             'before starting the map.')
#                return
            if None in self.coord_dict.values():
                logging.warn('You must save all four corners before starting the map.')
                return
#            if self.savepath is None:
#                logging.warn('You must input a valid savepath to start the map.')
#                return
#            if self.retuning_mode[0] == 'reference' and self.peak_intensity_reference is None:
#                logging.warn('You must record a peak intensity reference when using "reference" mode.')
#                return

            self.Mapper.coord_dict = self.coord_dict.copy()

#            self.Mapper.number_of_images = self.number_of_images
#            self.Mapper.dirt_area = self.dirt_area
#            self.Mapper.offset = self.offset
#            self.Mapper.savepath = self.savepath
#            self.Mapper.peak_intensity_reference = self.peak_intensity_reference
#            self.Mapper.frame_parameters = self.frame_parameters.copy()
#            self.isotope_mapping_settings['frame_parameters'] = self.isotope_frame_parameters.copy()
#            self.Mapper.isotope_mapping_settings = self.isotope_mapping_settings.copy()
#            self.Mapper.retuning_mode = self.retuning_mode.copy()
#            self.Mapper.average_number = self.average_number
#            self.Mapper.max_align_dist = self.max_align_dist
#            self.Mapper.sleeptime = self.sleeptime
            self.thread_communication = self.Mapper.gui_communication
#            self.thread_communication['abort_button'] = abort_button
#            self.thread_communication['done_button'] = done_button
#            self.thread_communication['analyze_button'] = analyze_button
#            self.tune_now_event = threading.Event()
#            self.Mapper.tune_now_event = self.tune_now_event
#            self.tune_event = threading.Event()
#            self.Mapper.tune_event = self.tune_event
#            if self.switches.get('isotope_mapping') or self.number_of_images > 1:
#                self.abort_series_event = threading.Event()
#                self.Mapper.abort_series_event = self.abort_series_event
#
#            logging.info('FOV: ' + str(self.frame_parameters['fov'])+' nm')
#            logging.info('Offset: ' + str(self.offset)+' x image size')
#            logging.info('Frame Rotation: ' + str(self.frame_parameters['rotation'])+' deg')
#            logging.info('Size: ' + str(self.frame_parameters['size_pixels'])+' px')
#            logging.info('Time: ' + str(self.frame_parameters['pixeltime'])+' us')
#            logging.info('Number of images per location: ' + str(self.number_of_images))

#            self.event = threading.Event()
#            self.Mapper.event = self.event
            self.Mapper.start()

#            self.thread = threading.Thread(target=Mapper.SuperScan_mapping)
#            self.thread.start()

        def abort_button_clicked():
#            self.low_level_event_occured('map_finished')
#            return
            #self.stop_tuning()
            if self.Mapper.number_of_images < 2 or (hasattr(self, 'last_time_abort_clicked') and
                                             time.time() - self.last_time_abort_clicked < 1):
                self.Mapper.abort()
                logging.info('Aborting map.')
            else:
                self.last_time_abort_clicked = time.time()
                self.Mapper.abort_series()
                logging.info('Aborting series. Click again to abort the whole map.')
#            if self.thread_communication.get('series_running'):
#                self.abort_series_event.set()
#                self.thread_communication['series_running'] = False
#                self.thread_communication['abort_button'].text = 'Abort map'
##                    self.document_controller.queue_task(lambda: self.update_abort_button('Abort map'))
#            else:
#                logging.info('Aborting after current frame is finished. (May take a short while until actual abort)')
#                self.event.set()

        def analyze_button_clicked():
            selected_data_item = document_controller.target_data_item
            xdata = copy.deepcopy(selected_data_item.xdata)
            imsize = xdata.dimensional_calibrations[0].scale * xdata.data.shape[0]
            result_data_item = self.document_controller.create_data_item_from_data_and_metadata(xdata)
            result_data_item.title = 'Dirt masked ' + selected_data_item.title
            def analyze_image(dirt_threshold):
                Peak = autotune.Peaking(image=xdata.data.copy(), imsize=imsize, integration_radius=1)
                if dirt_threshold is not None:
                    dirt_mask = Peak.dirt_detector(dirt_threshold=float(dirt_threshold))
                else:
                    dirt_mask = Peak.dirt_detector()
                graphene_mean = np.mean(Peak.image[dirt_mask==0])
                Peak.image[dirt_mask==1] = graphene_mean
                try:
                    peaks = Peak.find_peaks(half_line_thickness=2, position_tolerance = 10, second_order=True)
                except (RuntimeError, IndexError) as detail:
                    intensities_sum = None
                    logging.info(str(detail))
                else:
                    intensities_sum = np.sum(peaks[0][:,-1])+np.sum(peaks[1][:,-1])
                    #self.peak_intensity_reference = intensities_sum
                    logging.info('Measured peak intensities in {} from {} to: {:.0f}.'.format(
                                            selected_data_item._data_item.title,
                                            str(selected_data_item.data_and_metadata.timestamp).split('.')[0],
                                            intensities_sum))
                Peak.image[dirt_mask==1] = 1
                result_data_item.set_data(Peak.image)
                def finished_analysis():
                    self.Mapper.dirt_threshold = Peak.dirt_threshold
                    self.Mapper.peak_intensity_reference = intensities_sum
                self.document_controller.show_get_string_message_box('Dirt threshold', str(Peak.dirt_threshold),
                                                                     analyze_image, rejected_fn=finished_analysis,
                                                                     accepted_text='Try new dirt threshold',
                                                                     rejected_text='Done')
            analyze_image(self.Mapper.dirt_threshold)

        def sync_gui():
            self.sync_gui_working = True

            for key, value in self._checkboxes.items():
                value.checked = self.Mapper.switches.get(key, False)

            for key, value in self._text_fields.items():
                value.on_editing_finished('')

            method_combo_box._ComboBoxWidget__combo_box_widget.current_item = self.Mapper.retuning_mode[0].replace('_', ' ')
            mode_combo_box._ComboBoxWidget__combo_box_widget.current_item = self.Mapper.retuning_mode[1].replace('_', ' ')
#            fov_line_edit.text = str(self.Mapper.frame_parameters.get('fov'))
#            size_line_edit.text = str(self.Mapper.frame_parameters.get('size_pixels')[0])
#            rotation_line_edit.text = str(self.Mapper.frame_parameters.get('rotation'))
#            offset_line_edit.text = str(self.Mapper.offset)
#            time_line_edit.text = str(self.Mapper.frame_parameters.get('pixeltime'))
#            number_line_edit.text = str(self.Mapper.number_of_images)
#            savepath_line_edit.text = str(self.savepath)
#            dirt_area_line_edit.text = '{:.0f}'.format(self.dirt_area*100)
#            average_number_line_edit.text = '{:.0f}'.format(self.average_number)
#            max_align_dist_line_edit.text = '{:.3f}'.format(self.max_align_dist)
#            sleeptime_line_edit.text = '{:.1f}'.format(self.sleeptime)

            self.sync_gui_working = False

        mode_row = ui.create_row_widget()
        fields_row = ui.create_row_widget()

        left_fields_column = ui.create_column_widget()
        right_fields_column = ui.create_column_widget()
        left_edit_row1 = ui.create_row_widget()
        left_edit_row2 = ui.create_row_widget()
        left_edit_row3 = ui.create_row_widget()
        left_edit_row4 = ui.create_row_widget()
        right_edit_row1 = ui.create_row_widget()
        right_edit_row2 = ui.create_row_widget()
        right_edit_row3 = ui.create_row_widget()
        left_buttons_row1 = ui.create_row_widget()
        left_buttons_row2 = ui.create_row_widget()
        right_buttons_row1 = ui.create_row_widget()
        right_buttons_row2 = ui.create_row_widget()
        checkbox_row1 = ui.create_row_widget()
        checkbox_row2 = ui.create_row_widget()
        checkbox_row3 = ui.create_row_widget()
        checkbox_row4 = ui.create_row_widget()
        #checkbox_row5 = ui.create_row_widget()
        checkbox_row6 = ui.create_row_widget()
        checkbox_row7 = ui.create_row_widget()
        savepath_row = ui.create_row_widget()
        save_button_row = ui.create_row_widget()
        done_button_row = ui.create_row_widget()

        column.add_spacing(10)
        column.add(mode_row)
        mode_label = ui.create_label_widget(_("Low-dose mapping settings:"))
        mode_row.add(mode_label)
        column.add_spacing(5)
        column.add(fields_row)
        fields_row.add(left_fields_column)
        fields_row.add_spacing(5)
        fields_row.add(right_fields_column)
        left_fields_column.add(left_edit_row1)
        left_fields_column.add_spacing(5)
        left_fields_column.add(left_edit_row2)
        left_fields_column.add_spacing(5)
        left_fields_column.add(left_edit_row3)
        left_fields_column.add_spacing(5)
        left_fields_column.add(left_edit_row4)
        left_fields_column.add_spacing(10)
        right_fields_column.add(right_edit_row1)
        right_fields_column.add_spacing(5)
        right_fields_column.add(right_edit_row2)
        right_fields_column.add_spacing(5)
        right_fields_column.add(right_edit_row3)
        right_fields_column.add_spacing(10)
        left_fields_column.add_stretch()
        left_fields_column.add(ui.create_label_widget(_("Save Coordinates")))
        left_fields_column.add_spacing(5)
        right_fields_column.add_stretch()
        right_fields_column.add(ui.create_label_widget(_("Goto Coordinates")))
        right_fields_column.add_spacing(5)
        left_fields_column.add(left_buttons_row1)
        left_fields_column.add(left_buttons_row2)
        right_fields_column.add(right_buttons_row1)
        right_fields_column.add(right_buttons_row2)
        column.add_spacing(10)
        column.add(checkbox_row1)
        column.add_spacing(5)
        column.add(checkbox_row2)
        column.add_spacing(5)
        column.add(checkbox_row3)
        column.add_spacing(5)
        column.add(checkbox_row4)
        column.add_spacing(5)
#        column.add(checkbox_row5)
#        column.add_spacing(5)
        column.add(checkbox_row6)
        column.add_spacing(5)
        column.add(checkbox_row7)
        column.add_spacing(5)
        column.add(savepath_row)
        column.add_spacing(5)
        column.add(save_button_row)
        column.add_spacing(10)
        column.add(done_button_row)
        column.add_stretch()

        #####################################################
        #                   mode_row                        #
        #####################################################
        #                  fields_row                       #
        #   left_fields_column  ###    right_fields_column  #
        #   left_edit_row1      ###     right_edit_row1     #
        #####################################################
        #   left_edit_row2      ###     right_edit_row2     #
        #####################################################
        #   left_edit_row3      ###     right_edit_row3     #
        #####################################################
        #   left_edit_row4      ###     right_edit_row3     #
        #####################################################
        #   save_descriptor     ###     goto_descriptor     #
        #####################################################
        #   left_buttons_row1   ###  right_buttons_row1     #
        #####################################################
        #   left_buttons_row2   ###  right_buttons_row2     #
        #####################################################
        #                   checkbox_row1                   #
        #                   checkbox_row2                   #
        #                   checkbox_row3                   #
        #                   checkbox_row4                   #
        #                   checkbox_row5                   #
        #                   checkbox_row6                   #
        #                   checkbox_row7                   #
        #                    savepath_row                   #
        #                   done_button_row                 #
        #####################################################


        left_edit_row1.add(ui.create_label_widget(_("FOV (nm): ")))
        fov_line_edit = ui.create_line_edit_widget()
        fov_line_edit.on_editing_finished = fov_finished
        left_edit_row1.add(fov_line_edit)

        right_edit_row1.add(ui.create_label_widget(_("Framesize (px): ")))
        size_line_edit = ui.create_line_edit_widget()
        size_line_edit.on_editing_finished = size_finished
        right_edit_row1.add(size_line_edit)

        rotation_label = ui.create_label_widget(_("Rotation (deg): "))
        left_edit_row2.add(rotation_label)
        rotation_line_edit = ui.create_line_edit_widget()
        rotation_line_edit.on_editing_finished = rotation_finished
        left_edit_row2.add(rotation_line_edit)

        offset_label = ui.create_label_widget(_("Offset (images): "))
        right_edit_row2.add(offset_label)
        offset_line_edit = ui.create_line_edit_widget()
        offset_line_edit.on_editing_finished = offset_finished
        right_edit_row2.add(offset_line_edit)

        left_edit_row3.add(ui.create_label_widget(_("Pixeltime (us): ")))
        time_line_edit = ui.create_line_edit_widget()
        time_line_edit.on_editing_finished = time_finished
        left_edit_row3.add(time_line_edit)

        left_edit_row4.add(ui.create_label_widget(_("Sleeptime (s): ")))
        sleeptime_line_edit = ui.create_line_edit_widget()
        sleeptime_line_edit.on_editing_finished = sleeptime_finished
        left_edit_row4.add(sleeptime_line_edit)

        series_label = ui.create_label_widget(_("Series length: "))
        right_edit_row3.add(series_label)
        number_line_edit = ui.create_line_edit_widget()
        number_line_edit.on_editing_finished = number_of_images_finished
        right_edit_row3.add(number_line_edit)

        savepath_row.add(ui.create_label_widget(_("Savepath: ")))
        savepath_line_edit = ui.create_line_edit_widget()
        savepath_line_edit.on_editing_finished = saving_finished
        savepath_row.add(savepath_line_edit)
        savepath_row.add_spacing(5)
        browse_button = ui.create_push_button_widget("Browse...")
        browse_button.on_clicked = browse_button_clicked
        savepath_row.add(browse_button)

        dirt_area_line_edit = ui.create_line_edit_widget()
        dirt_area_line_edit.on_editing_finished = dirt_area_finished
        average_number_line_edit = ui.create_line_edit_widget()
        average_number_line_edit.on_editing_finished = average_number_finished
        max_align_dist_line_edit = ui.create_line_edit_widget()
        max_align_dist_line_edit.on_editing_finished = max_align_dist_finished

        tl_button = ui.create_push_button_widget(_("Top\nLeft"))
        tr_button = ui.create_push_button_widget(_("Top\nRight"))
        bl_button = ui.create_push_button_widget(_("Bottom\nLeft"))
        br_button = ui.create_push_button_widget(_("Bottom\nRight"))
        drive_tl = ui.create_push_button_widget(_("Top\nLeft"))
        drive_tr = ui.create_push_button_widget(_("Top\nRight"))
        drive_bl = ui.create_push_button_widget(_("Bottom\nLeft"))
        drive_br = ui.create_push_button_widget(_("Bottom\nRight"))
        save_button = ui.create_push_button_widget(_("Save Configs"))
        load_button = ui.create_push_button_widget(_("Load Configs"))
        test_button = ui.create_push_button_widget(_("Test image"))
        done_button = ui.create_push_button_widget(_("Start map"))
        abort_button = ui.create_push_button_widget(_("Abort map"))
        analyze_button = ui.create_push_button_widget(_("Analyze image"))

        retune_checkbox = ui.create_check_box_widget(_("Retune live "))
        retune_checkbox.on_check_state_changed = checkbox_changed
        method_combo_box = ui.create_combo_box_widget()
        method_combo_box.items = ['at every position', 'on dirt']
        method_combo_box.on_current_item_changed = method_combo_box_changed
        mode_combo_box = ui.create_combo_box_widget()
        mode_combo_box.items = ['manual', 'auto']
        mode_combo_box.on_current_item_changed = mode_combo_box_changed
        overview_checkbox = ui.create_check_box_widget(_("Acquire Overview"))
        overview_checkbox.check_state = 'checked'
        overview_checkbox.on_check_state_changed = checkbox_changed
        blank_checkbox = ui.create_check_box_widget(_("Blank beam between images"))
        blank_checkbox.on_check_state_changed = checkbox_changed
        correct_stage_errors_checkbox = ui.create_check_box_widget(_("Correct Stage Movement"))
        correct_stage_errors_checkbox.on_check_state_changed = checkbox_changed
        z_drive_checkbox = ui.create_check_box_widget(_("Use Z Drive"))
        z_drive_checkbox.on_check_state_changed = checkbox_changed
        abort_series_on_dirt_checkbox = ui.create_check_box_widget(_("Abort series on more than "))
        correct_stage_errors_checkbox.on_check_state_changed = checkbox_changed
        show_average_checkbox = ui.create_check_box_widget(_("Show average of last "))
        show_average_checkbox.on_check_state_changed = checkbox_changed
        align_average_checkbox = ui.create_check_box_widget(_("Align average. Maximum distance (images): "))
        align_average_checkbox.on_check_state_changed = checkbox_changed

        tl_button.on_clicked = tl_button_clicked
        tr_button.on_clicked = tr_button_clicked
        bl_button.on_clicked = bl_button_clicked
        br_button.on_clicked = br_button_clicked
        drive_tl.on_clicked = drive_tl_button_clicked
        drive_tr.on_clicked = drive_tr_button_clicked
        drive_bl.on_clicked = drive_bl_button_clicked
        drive_br.on_clicked = drive_br_button_clicked
        save_button.on_clicked = save_button_clicked
        load_button.on_clicked = load_button_clicked
        test_button.on_clicked = test_button_clicked
        done_button.on_clicked = done_button_clicked
        abort_button.on_clicked = abort_button_clicked
        analyze_button.on_clicked = analyze_button_clicked

        left_buttons_row1.add(tl_button)
        left_buttons_row1.add_spacing(2)
        left_buttons_row1.add(tr_button)

        left_buttons_row2.add(bl_button)
        left_buttons_row2.add_spacing(2)
        left_buttons_row2.add(br_button)

        right_buttons_row1.add(drive_tl)
        right_buttons_row1.add_spacing(2)
        right_buttons_row1.add(drive_tr)

        right_buttons_row2.add(drive_bl)
        right_buttons_row2.add_spacing(2)
        right_buttons_row2.add(drive_br)

        checkbox_row1.add(retune_checkbox)
        checkbox_row1.add(method_combo_box)
        #checkbox_row1.add(ui.create_label_widget(_(' mode: ')))
        checkbox_row1.add(mode_combo_box)
        checkbox_row1.add_stretch()

        checkbox_row2.add(overview_checkbox)
        checkbox_row2.add_spacing(3)
        #checkbox_row2.add(blank_checkbox)
        checkbox_row2.add_stretch()

        checkbox_row3.add(correct_stage_errors_checkbox)
        checkbox_row3.add_spacing(3)
        checkbox_row3.add(z_drive_checkbox)
        checkbox_row3.add_stretch()

        checkbox_row4.add(abort_series_on_dirt_checkbox)
        checkbox_row4.add(dirt_area_line_edit)
        checkbox_row4.add(ui.create_label_widget(_('% dirt in image')))
        checkbox_row4.add_stretch()

#        checkbox_row5.add(isotope_mapping_checkbox)
#        checkbox_row5.add_stretch()

        checkbox_row6.add(show_average_checkbox)
        checkbox_row6.add(average_number_line_edit)
        checkbox_row6.add(ui.create_label_widget(_(' images')))
        checkbox_row6.add_stretch()
        checkbox_row6.add_stretch()
        checkbox_row6.add_stretch()
        checkbox_row6.add_stretch()
        checkbox_row6.add_stretch()

        checkbox_row7.add(align_average_checkbox)
        checkbox_row7.add(max_align_dist_line_edit)
        checkbox_row7.add_stretch()

        save_button_row.add(save_button)
        save_button_row.add_spacing(4)
        save_button_row.add(load_button)
        save_button_row.add_spacing(4)
        save_button_row.add(test_button)

        done_button_row.add(done_button)
        done_button_row.add_spacing(4)
        done_button_row.add(abort_button)
        done_button_row.add_spacing(4)
        done_button_row.add(analyze_button)

        self._checkboxes['do_retuning'] = retune_checkbox
        self._checkboxes['use_z_drive'] = z_drive_checkbox
        self._checkboxes['acquire_overview'] = overview_checkbox
        self._checkboxes['compensate_stage_error'] = correct_stage_errors_checkbox
        self._checkboxes['abort_series_on_dirt'] = abort_series_on_dirt_checkbox
        self._checkboxes['show_last_frames_average'] = show_average_checkbox
        self._checkboxes['aligned_average'] = align_average_checkbox

        self._buttons['test'] = test_button
        self._buttons['done'] = done_button
        self._buttons['load'] = load_button
        self._buttons['save'] = save_button
        self._buttons['abort'] = abort_button
        self._buttons['analyze'] = analyze_button
        self._buttons['tl'] = tl_button
        self._buttons['tr'] = tr_button
        self._buttons['bl'] = bl_button
        self._buttons['br'] = br_button
        self._buttons['drive_bl'] = drive_bl
        self._buttons['drive_br'] = drive_br
        self._buttons['drive_tl'] = drive_tl
        self._buttons['drive_tr'] = drive_tr

        self._text_fields['sleeptime'] = sleeptime_line_edit
        self._text_fields['fov'] = fov_line_edit
        self._text_fields['size'] = size_line_edit
        self._text_fields['rotation'] = rotation_line_edit
        self._text_fields['offset'] = offset_line_edit
        self._text_fields['time'] = time_line_edit
        self._text_fields['number'] = number_line_edit
        self._text_fields['dirt_area'] = dirt_area_line_edit
        self._text_fields['average_number'] = average_number_line_edit
        self._text_fields['max_align_dist'] = max_align_dist_line_edit
        self._text_fields['savepath'] = savepath_line_edit

        self._dropdowns['mode'] = mode_combo_box
        self._dropdowns['method'] = method_combo_box

        sync_gui()

        return column

    def low_level_event_occured(self, name):
        if name == 'map_started':
            def update_button():
                self._buttons['test'].text = 'Retune now'
                if self.Mapper.number_of_images > 1:
                    self._buttons['abort'].text = 'Abort series'
            self.document_controller.queue_task(update_button)

            def disable_text_fields():
                for key, value in self._text_fields.items():
                    if key in ['fov', 'size', 'rotation', 'offset', 'time', 'savepath', 'sleeptime']:
                        value._widget.enabled = False
            self.document_controller.queue_task(disable_text_fields)

            def disable_buttons():
                for key, value in self._buttons.items():
                    if key in ['tl', 'tr', 'bl', 'br', 'drive_tl', 'drive_tr', 'drive_bl', 'drive_br', 'save', 'load']:
                        value._widget.enabled = False
            self.document_controller.queue_task(disable_buttons)

            def disable_checkboxes():
                for key, value in self._checkboxes.items():
                    if key in ['compensate_stage_error', 'use_z_drive']:
                        value._widget.enabled = False
            self.document_controller.queue_task(disable_checkboxes)

        elif name == 'map_finished':
            def update_button():
                self._buttons['test'].text = 'Test image'
                self._buttons['abort'].text = 'Abort map'
            self.document_controller.queue_task(update_button)

            def enable_text_fields():
                for key, value in self._text_fields.items():
                    if key in ['fov', 'size', 'rotation', 'offset', 'time', 'savepath', 'sleeptime']:
                        value._widget.enabled = True
            self.document_controller.queue_task(enable_text_fields)

            def enable_buttons():
                for key, value in self._buttons.items():
                    if key in ['tl', 'tr', 'bl', 'br', 'drive_tl', 'drive_tr', 'drive_bl', 'drive_br', 'save', 'load']:
                        value._widget.enabled = True
            self.document_controller.queue_task(enable_buttons)

            def enable_checkboxes():
                for key, value in self._checkboxes.items():
                    if key in ['compensate_stage_error', 'use_z_drive']:
                        value._widget.enabled = True
            self.document_controller.queue_task(enable_checkboxes)

        elif name == 'waiting_for_focus':
            def update_button():
                self._buttons['done'].text = 'Done tuning'
            self.document_controller.queue_task(update_button)
            self.waiting_for_focus = True
        elif name == 'finished_focus':
            def update_button():
                self._buttons['done'].text = 'Start map'
            self.document_controller.queue_task(update_button)
            self.waiting_for_focus = False

    def save_coords(self, position):
        self.coord_dict[position] = (self.as2.get_property_as_float('StageOutX'),
                                     self.as2.get_property_as_float('StageOutY'),
                                     self.as2.get_property_as_float('StageOutZ'),
                                     self.as2.get_property_as_float('EHTFocus'))
        logging.info('Saved x: ' +
                     str(self.coord_dict[position][0]) + ', y: ' +
                     str(self.coord_dict[position][1]) + ', z: ' +
                     str(self.coord_dict[position][2]) + ', focus: ' +
                     str(self.coord_dict[position][3]) + ' as ' + position + ' corner.')

    def drive_coords(self, position):
        if self.coord_dict[position] is None:
            logging.warn('You haven\'t set the '+position+' corner yet.')
        else:
            logging.info('Going to ' + str(position) + ' corner: x: ' + str(self.coord_dict[position][0]) + ', y: ' +
                         str(self.coord_dict[position][1]) + ', z: '+str(self.coord_dict[position][2]) + ', focus: ' +
                         str(self.coord_dict[position][3]))
            self.as2.set_property_as_float('StageOutX', self.coord_dict[position][0])
            self.as2.set_property_as_float('StageOutY', self.coord_dict[position][1])
            self.as2.set_property_as_float('EHTFocus', self.coord_dict[position][3])
            if self.Mapper.switches['use_z_drive']:
                self.as2.set_property_as_float('StageOutZ', self.coord_dict[position][2])

    def total_number_frames(self):
        if not None in self.coord_dict.values() and not None in self.Mapper.frame_parameters.values():
            corners = ('top-left', 'top-right', 'bottom-right', 'bottom-left')
            coords = []
            for corner in corners:
                coords.append(self.coord_dict[corner])
            Mpr = mapper.Mapping(coord_dict=self.coord_dict.copy())
            self.coord_dict_sorted = Mpr.sort_quadrangle()
            coords = []
            for corner in corners:
                coords.append(self.coord_dict_sorted[corner])
            imsize = self.Mapper.frame_parameters['fov']*1e-9
            distance = self.Mapper.offset*imsize
            leftX = np.min((coords[0][0],coords[3][0]))
            rightX = np.max((coords[1][0],coords[2][0]))
            topY = np.max((coords[0][1],coords[1][1]))
            botY = np.min((coords[2][1],coords[3][1]))
            num_subframes = (int(np.abs(rightX-leftX)/(imsize+distance))+1, int(np.abs(topY-botY)/(imsize+distance))+1)

            logging.info('With the current settings, %dx%d frames (%d in total) will be taken.'
                         % (num_subframes[0], num_subframes[1], num_subframes[0]*num_subframes[1]))
            logging.info('A total area of %.4f um2 will be scanned.'
                         % (num_subframes[0]*num_subframes[1]*(self.Mapper.frame_parameters['fov']*1e-3)**2))
            logging.info('Approximate mapping time: %.0f s'
                         % (num_subframes[0]*num_subframes[1]*(self.Mapper.frame_parameters['size_pixels'][0]**2
                            *np.mean(self.Mapper.frame_parameters['pixeltime'])*1e-6*self.Mapper.number_of_images + self.Mapper.sleeptime)))


class ScanMapExtension(object):
    extension_id = 'univie.scanmap'

    def __init__(self, api_broker):
        api = api_broker.get_api(version='1', ui_version='1')
        self.__panel_ref = api.create_panel(ScanMapPanelDelegate(api))

    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None