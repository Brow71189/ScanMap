# standard libraries
import gettext
import logging
import numpy as np
import threading
import warnings
import os

try:
    from importlib import reload
except:
    pass

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import ViennaTools.ViennaTools as vt

from .maptools import mapper

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
        self.coord_dict = {'top-left': None, 'top-right': None, 'bottom-right': None, 'bottom-left': None}
        self.switches = {}
        self.frame_parameters = {'size_pixels': None, 'pixeltime': None, 'fov': None, 'rotation': 0}
        self.offset = 0
        self.number_of_images = 1
        self.savepath = None
        self._checkboxes = {'do_autotuning': False, 'use_z_drive': False, 'auto_offset': False, 'auto_rotation': False,
                            'compensate_stage_error': False, 'acquire_overview': True, 'blank_beam': False}
    
    def create_panel_widget(self, ui, document_controller):
        
        self.superscan = self.__api.get_hardware_source_by_id('scan_controller', '1')
        self.as2 = self.__api.get_instrument_by_id('autostem_controller', '1')        
        
        column = ui.create_column_widget()        
        
        def fov_finished(text):
            if len(text) > 0:
                try:
                    fov = float(text)
                    self.frame_parameters['fov'] = fov
                    logging.info('Setting FOV to: ' + str(fov) + ' nm.')
                except ValueError:
                    logging.warn(text + ' is not a valid FOV. Please input a floating point number.')
                    fov_line_edit.select_all()
                    
                self.total_number_frames()
            
        def size_finished(text):
            if len(text) > 0:
                try:
                    size = int(text)
                    self.frame_parameters['size_pixels'] = (size, size)
                    logging.info('Setting Image Size to: ' + str(size))
                except ValueError:
                    logging.warn(text + ' is not a valid size. Please input an integer number.')
                    size_line_edit.select_all()
                    
                self.total_number_frames()
            
        def offset_finished(text):
            if len(text) > 0:
                try:
                    offset = float(text)
                    self.offset = offset
                    logging.info('Setting Offset to: ' + str(offset))
                except ValueError:
                    logging.warn(text + ' is not a valid Offset. Please input a floating point number.')
                    offset_line_edit.select_all()
                    
                self.total_number_frames()
        
        def time_finished(text):
            if len(text) > 0:
                try:
                    time = float(text)
                    self.frame_parameters['pixeltime'] = time
                    logging.info('Setting pixeltime to: ' + str(time) + ' us.')
                except (TypeError, ValueError):
                    try:
                        time = [float(s) for s in text.split(',')]
                        self.frame_parameters['pixeltime'] = time
                        logging.info('Pixel times will be (in this order): ' + str(time) + ' us.')
                    except ValueError:
                        logging.warn(text + ' is not a valid Time. Please input a floating point number or a comma-seperated list of floats')
                        time_line_edit.select_all()
                
                self.total_number_frames()                
            
        def rotation_finished(text):
            if len(text) > 0:
                try:
                    rotation = float(text)
                    self.frame_parameters['rotation'] = rotation
                    logging.info('Setting frame rotation to: ' + str(rotation))
                except ValueError:
                    logging.warn(text + ' is not a valid Frame Rotation. Please input a floating point number.')
                    rotation_line_edit.select_all()
                
        def number_of_images_finished(text):
            if len(text) > 0:
                try:
                    self.number_of_images = int(text)
                    if self.number_of_images > 1:
                        logging.info(str(self.number_of_images)+ ' images will be recorded at each location.')
                    else:
                        logging.info('One image will be recorded at each location.')
                except ValueError:
                    logging.warn(text + ' is not a valid number. Please input an integer number.')
                    number_line_edit.select_all()
                    
                self.total_number_frames()
                
        def saving_finished(text):
            if len(text)>0:
                if os.path.isabs(text):
                    self.savepath = text
                else:
                    logging.warn(text+' is not an absolute path. Please enter a complete pathname starting from root.')
            else:
                self.savepath = None
                
        def checkbox_changed(check_state):
            for key, value in self._checkboxes.items():
                self.switches[key] = value.checked
        
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
            
        def save_button_clicked():
            Mapper = mapper.Mapping()            
            Mapper.coord_dict=self.coord_dict.copy()
            Mapper.switches=self.switches.copy()
            Mapper.number_of_images = self.number_of_images
            Mapper.offset = self.offset
            Mapper.savepath = self.savepath
            Mapper.frame_parameters = self.frame_parameters.copy()
            
            Mapper.save_mapping_config()
            logging.info('Saved config file to: ' + os.path.join(Mapper.savepath, Mapper.foldername, 'configs_map.txt'))
        
        def load_button_clicked():
            if not os.path.isfile(self.savepath):
                logging.warn('Please type the path to the config file into the \'savepath\' field to load configs.')
            else:
                Mapper = mapper.Mapping()
                Mapper.load_mapping_config(self.savepath)
                self.coord_dict = Mapper.coord_dict.copy()
                self.switches = Mapper.switches.copy()
                self.number_of_images = Mapper.number_of_images
                self.offset = Mapper.offset
                self.savepath = Mapper.savepath
                self.frame_parameters = Mapper.frame_parameters.copy()
                
                sync_gui()
                logging.info('Loaded all mapping configs successfully.')
       
        def done_button_clicked():
            saving_finished(savepath_line_edit.text)
            checkbox_changed('obligatory string argument')
            self.total_number_frames()
            
            try:
                reload(mapper)
            except:
                logging.warn('Couldn\'t reload mapper')
                
            if None in self.frame_parameters.values():
                logging.warn('You must specify all scan parameters (e.g. FOV, framesize, rotation, pixeltime) ' +
                             'before starting the map.')
                return
            if None in self.coord_dict.values():
                logging.warn('You must save all four corners before starting the map.')
                return
            if self.savepath is None:
                logging.warn('You must input a valid savepath to start the map.')
                return
            
            Mapper = mapper.Mapping(superscan=self.superscan, as2=self.as2, document_controller=document_controller,
                                    coord_dict=self.coord_dict.copy(), switches=self.switches.copy())
                                    
            Mapper.number_of_images = self.number_of_images
            Mapper.offset = self.offset
            Mapper.savepath = self.savepath
            Mapper.frame_parameters = self.frame_parameters.copy()
            
            logging.info('FOV: ' + str(self.frame_parameters['fov'])+' nm')
            logging.info('Offset: ' + str(self.offset)+' x image size')
            logging.info('Frame Rotation: ' + str(self.frame_parameters['rotation'])+' deg')
            logging.info('Size: ' + str(self.frame_parameters['size_pixels'])+' px')
            logging.info('Time: ' + str(self.frame_parameters['pixeltime'])+' us')
            logging.info('Number of images per location: ' + str(self.number_of_images))
            
            self.event = threading.Event()
            Mapper.event = self.event
            self.thread = threading.Thread(target=Mapper.SuperScan_mapping)
            self.thread.start()

        def abort_button_clicked():
            #self.stop_tuning()
            logging.info('Aborting after current frame is finished. (May take a short while until actual abort)')
            self.event.set()
            
        def sync_gui():
            for key, value in self._checkboxes.items():
                value.checked = self.switches.get(key, False)
                
            fov_line_edit.text = str(self.frame_parameters.get('fov'))
            size_line_edit.text = str(self.frame_parameters.get('size_pixels')[0])
            rotation_line_edit.text = str(self.frame_parameters.get('rotation'))
            offset_line_edit.text = str(self.offset)
            time_line_edit.text = str(self.frame_parameters.get('pixeltime'))
            number_line_edit.text = str(self.number_of_images)
            savepath_line_edit.text = str(self.savepath)
            
        fields_row = ui.create_row_widget()
        
        left_fields_column = ui.create_column_widget()
        right_fields_column = ui.create_column_widget()
        left_edit_row1 = ui.create_row_widget()
        left_edit_row2 = ui.create_row_widget()
        left_edit_row3 = ui.create_row_widget()
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
        savepath_row = ui.create_row_widget()
        save_button_row = ui.create_row_widget()
        done_button_row = ui.create_row_widget()

        column.add_spacing(20)        
        column.add(fields_row)
        fields_row.add(left_fields_column)
        fields_row.add_spacing(10)
        fields_row.add(right_fields_column)
        left_fields_column.add(left_edit_row1)
        left_fields_column.add_spacing(5)
        left_fields_column.add(left_edit_row2)
        left_fields_column.add_spacing(5)
        left_fields_column.add(left_edit_row3)
        left_fields_column.add_spacing(20)
        right_fields_column.add(right_edit_row1)
        right_fields_column.add_spacing(5)
        right_fields_column.add(right_edit_row2)
        right_fields_column.add_spacing(5)
        right_fields_column.add(right_edit_row3)
        right_fields_column.add_spacing(20)
        left_fields_column.add(ui.create_label_widget(_("Save Coordinates")))
        left_fields_column.add_spacing(5)
        right_fields_column.add(ui.create_label_widget(_("Goto Coordinates")))
        right_fields_column.add_spacing(5)
        left_fields_column.add(left_buttons_row1)
        left_fields_column.add(left_buttons_row2)
        right_fields_column.add(right_buttons_row1)
        right_fields_column.add(right_buttons_row2)
        column.add_spacing(20)
        column.add(checkbox_row1)
        column.add_spacing(5)
        column.add(checkbox_row2)
        column.add_spacing(5)
        column.add(checkbox_row3)
        column.add_spacing(5)
        column.add(savepath_row)
        column.add_spacing(5)
        column.add(save_button_row)
        column.add_spacing(20)
        column.add(done_button_row)
        column.add_stretch()

        #####################################################
        #                  fields_row                       #
        #   left_fields_column  ###    right_fields_column  #
        #   left_edit_row1      ###     right_edit_row1     #
        #####################################################
        #   left_edit_row2      ###     right_edit_row2     #
        #####################################################
        #   left_edit_row3      ###     right_edit_row3     #
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
        #                    savepath_row                   #
        #                   done_button_row                 #
        #####################################################
        
        left_edit_row1.add(ui.create_label_widget(_("FOV per Frame (nm): ")))
        fov_line_edit = ui.create_line_edit_widget()
        fov_line_edit.on_editing_finished = fov_finished
        left_edit_row1.add(fov_line_edit)
        
        right_edit_row1.add(ui.create_label_widget(_("Framesize (px): ")))
        size_line_edit = ui.create_line_edit_widget()
        size_line_edit.on_editing_finished = size_finished
        right_edit_row1.add(size_line_edit)
        
        left_edit_row2.add(ui.create_label_widget(_("Scan roation (deg): ")))
        rotation_line_edit = ui.create_line_edit_widget()
        rotation_line_edit.on_editing_finished = rotation_finished
        left_edit_row2.add(rotation_line_edit)
        
        right_edit_row2.add(ui.create_label_widget(_("Offset (images): ")))
        offset_line_edit = ui.create_line_edit_widget()
        offset_line_edit.on_editing_finished = offset_finished
        right_edit_row2.add(offset_line_edit)
        
        left_edit_row3.add(ui.create_label_widget(_("Pixeltime (us): ")))
        time_line_edit = ui.create_line_edit_widget()
        time_line_edit.placeholder_text = 'Number or comma-separated list of numbers'
        time_line_edit.on_editing_finished = time_finished
        left_edit_row3.add(time_line_edit)
        
        right_edit_row3.add(ui.create_label_widget(_("Images per location: ")))
        number_line_edit = ui.create_line_edit_widget()
        number_line_edit.placeholder_text = 'Defaults to 1'
        number_line_edit.on_editing_finished = number_of_images_finished
        right_edit_row3.add(number_line_edit)
        
        savepath_row.add(ui.create_label_widget(_("Savepath: ")))
        savepath_line_edit = ui.create_line_edit_widget()
        savepath_line_edit.text = "Z:/ScanMap/"
        savepath_line_edit.on_editing_finished = saving_finished
        savepath_row.add(savepath_line_edit)

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
        done_button = ui.create_push_button_widget(_("Done"))
        abort_button = ui.create_push_button_widget(_("Abort"))
   
        z_drive_checkbox = ui.create_check_box_widget(_("Use Z Drive"))
        z_drive_checkbox.on_check_state_changed = checkbox_changed
        autotuning_checkbox = ui.create_check_box_widget(_("Autotuning"))
        autotuning_checkbox.on_check_state_changed = checkbox_changed
        auto_offset_checkbox = ui.create_check_box_widget(_("Auto Offset"))
        auto_offset_checkbox.on_check_state_changed = checkbox_changed
        auto_rotation_checkbox = ui.create_check_box_widget(_("Auto Rotation"))
        auto_rotation_checkbox.on_check_state_changed = checkbox_changed
        overview_checkbox = ui.create_check_box_widget(_("Acquire Overview"))
        overview_checkbox.check_state = 'checked'
        overview_checkbox.on_check_state_changed = checkbox_changed
        blank_checkbox = ui.create_check_box_widget(_("Blank beam between acquisitions"))
        blank_checkbox.on_check_state_changed = checkbox_changed
        correct_stage_errors_checkbox = ui.create_check_box_widget(_("Compensate Stage Movement Errors"))
        correct_stage_errors_checkbox.on_check_state_changed = checkbox_changed
        
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
        done_button.on_clicked = done_button_clicked
        abort_button.on_clicked = abort_button_clicked
        
        left_buttons_row1.add(tl_button)
        left_buttons_row1.add_spacing(3)
        left_buttons_row1.add(tr_button)
        
        left_buttons_row2.add(bl_button)
        left_buttons_row2.add_spacing(3)
        left_buttons_row2.add(br_button)
        
        right_buttons_row1.add(drive_tl)
        right_buttons_row1.add_spacing(3)
        right_buttons_row1.add(drive_tr)
        
        right_buttons_row2.add(drive_bl)
        right_buttons_row2.add_spacing(3)
        right_buttons_row2.add(drive_br)
        
        checkbox_row1.add(autotuning_checkbox)
        checkbox_row1.add_spacing(4)
        checkbox_row1.add(auto_rotation_checkbox)        
        checkbox_row1.add_spacing(4)
        checkbox_row1.add(auto_offset_checkbox)
        checkbox_row1.add_spacing(4)
        checkbox_row1.add(z_drive_checkbox)
        checkbox_row1.add_stretch()
        
        checkbox_row2.add(overview_checkbox)
        checkbox_row2.add_spacing(4)
        checkbox_row2.add(blank_checkbox)
        checkbox_row2.add_stretch()

        checkbox_row3.add(correct_stage_errors_checkbox)
        checkbox_row3.add_stretch()
        
        save_button_row.add(save_button)
        save_button_row.add_spacing(15)
        save_button_row.add(load_button)
                
        done_button_row.add(done_button)
        done_button_row.add_spacing(15)
        done_button_row.add(abort_button)
        
        self._checkboxes['use_z_drive'] = z_drive_checkbox
        self._checkboxes['do_autotuning'] = autotuning_checkbox
        self._checkboxes['auto_offset'] = auto_offset_checkbox
        self._checkboxes['auto_rotation'] = auto_rotation_checkbox
        self._checkboxes['acquire_overview'] = overview_checkbox
        self._checkboxes['blank_beam'] = blank_checkbox
        self._checkboxes['compensate_stage_error'] = correct_stage_errors_checkbox

        return column
        
    def save_coords(self, position):
        try:
            reload(vt)
        except:
            logging.warn('Could not reload ViennaTools!')
        self.coord_dict[position] = (vt.as2_get_control(self.as2, 'StageOutX'), vt.as2_get_control(self.as2, 'StageOutY'), 
                                vt.as2_get_control(self.as2, 'StageOutZ'), vt.as2_get_control(self.as2, 'EHTFocus'))
        logging.info('Saved x: ' +
                     str(vt.as2_get_control(self.as2, 'StageOutX')) + ', y: ' +
                     str(vt.as2_get_control(self.as2, 'StageOutY')) + ', z: ' +
                     str(vt.as2_get_control(self.as2, 'StageOutZ')) + ', focus: ' +
                     str(vt.as2_get_control(self.as2, 'EHTFocus')) + ' as ' + position + ' corner.')

    def drive_coords(self, position):
        if self.coord_dict[position] is None:
            logging.warn('You haven\'t set the '+position+' corner yet.')
        else:
            logging.info('Going to '+str(position)+' corner: x: '+str(self.coord_dict[position][0])+', y: '+str(self.coord_dict[position][1])+', z: '+str(self.coord_dict[position][2])+', focus: '+str(self.coord_dict[position][3]))
            vt.as2_set_control(self.as2, 'StageOutX', self.coord_dict[position][0])
            vt.as2_set_control(self.as2, 'StageOutY', self.coord_dict[position][1])
            vt.as2_set_control(self.as2, 'EHTFocus', self.coord_dict[position][3])
            if self.switches['use_z_drive']:          
                vt.as2_set_control(self.as2, 'StageOutZ', self.coord_dict[position][2])
            
    def total_number_frames(self):
        if not None in self.coord_dict.values() and not None in self.frame_parameters.values():
            corners = ('top-left', 'top-right', 'bottom-right', 'bottom-left')
            coords = []
            for corner in corners:
                coords.append(self.coord_dict[corner])
            Mpr = mapper.Mapping(coord_dict=self.coord_dict.copy())
            self.coord_dict_sorted = Mpr.sort_quadrangle()
            coords = []
            for corner in corners:
                coords.append(self.coord_dict_sorted[corner])
            imsize = self.frame_parameters['fov']*1e-9
            distance = self.offset*imsize    
            leftX = np.min((coords[0][0],coords[3][0]))
            rightX = np.max((coords[1][0],coords[2][0]))
            topY = np.max((coords[0][1],coords[1][1]))
            botY = np.min((coords[2][1],coords[3][1]))
            num_subframes = ( int(np.abs(rightX-leftX)/(imsize+distance))+1, int(np.abs(topY-botY)/(imsize+distance))+1 )
            
            logging.info('With the current settings, %dx%d frames (%d in total) will be taken.'
                         % (num_subframes[0], num_subframes[1], num_subframes[0]*num_subframes[1]))
            logging.info('A total area of %.4f um2 will be scanned.'
                         % (num_subframes[0]*num_subframes[1]*(self.frame_parameters['fov']*1e-3)**2))
            logging.info('Approximate mapping time: %.0f s' 
                         % (num_subframes[0]*num_subframes[1]*(self.frame_parameters['size_pixels'][0]**2
                            *np.sum(self.frame_parameters['pixeltime'])*1e-6 + 4)))
    

class ScanMapExtension(object):
    extension_id = 'univie.scanmap'
    
    def __init__(self, api_broker):
        api = api_broker.get_api(version='1', ui_version='1')
        self.__panel_ref = api.create_panel(ScanMapPanelDelegate(api))
    
    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None