# standard libraries
import gettext
import logging
import numpy as np
import threading
from importlib import reload
# third party libraries
# None
try:
    import ViennaTools.ViennaTools as vt
except:
    try:
        import ViennaTools as vt
    except:
        logging.warn('Could not import Vienna tools!')

# local libraries
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import DataItem
from nion.ui import Binding
#document_controller.ui is object of nion.ui.UserInterface.QtUserInterface

from .maptools import mapper

_ = gettext.gettext

coord_dict = {'top-left': None, 'top-right': None, 'bottom-right': None, 'bottom-left': None}
do_autofocus = False
use_z_drive = False
auto_offset=False
auto_rotation=False
acquire_overview = True
FOV = Size = Offset = Time = Rotation = None
Number_of_images = 1

class ScanMap(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super(ScanMap, self).__init__(document_controller, panel_id, "Example")

        ui = document_controller.ui
        # user interface

        column = ui.create_column_widget()
        
        def FOV_finished(text):
            global FOV
            if len(text) > 0:
                try:
                    FOV = float(text)
                    logging.info('Setting FOV to: ' + str(FOV) + ' nm.')
                except:
                    logging.warn(text + ' is not a valid FOV. Please input a floating point number.')
                    FOV_line_edit.select_all()
                    
                total_number_frames()
            
        def Size_finished(text):
            global Size
            if len(text) > 0:
                try:
                    Size = int(text)
                    logging.info('Setting Image Size to: ' + str(Size))
                except:
                    logging.warn(text + ' is not a valid size. Please input an integer number.')
                    Size_line_edit.select_all()
                    
                total_number_frames()
            
        def Offset_finished(text):
            global Offset
            if len(text) > 0:
                try:
                    Offset = float(text)
                    logging.info('Setting Offset to: ' + str(Offset))
                except:
                    logging.warn(text + ' is not a valid Offset. Please input a floating point number.')
                    Offset_line_edit.select_all()
                    
                total_number_frames()
        
        def Time_finished(text):
            global Time
            if len(text) > 0:
                try:
                    Time = float(text)
                    logging.info('Setting pixeltime to: ' + str(Time) + ' us.')
                except:
                    try:
                        Time = [float(s) for s in text.split(',')]
                        logging.info('Pixel times will be (in this order): ' + str(Time) + ' us.')
                    except:
                        logging.warn(text + ' is not a valid Time. Please input a floating point number or a comma-seperated list of floats')
                        Time_line_edit.select_all()
                
                total_number_frames()                
            
        def Rotation_finished(text):
            global Rotation
            if len(text) > 0:
                try:
                    Rotation = float(text)
                    logging.info('Setting frame rotation to: ' + str(Rotation))
                except:
                    logging.warn(text + ' is not a valid Frame Rotation. Please input a floating point number.')
                    Rotation_line_edit.select_all()
                
        def Number_of_images(text):
            global Number_of_images
            if len(text) > 0:
                try:
                    Number_of_images = int(text)
                    if Number_of_images > 1:
                        logging.info(str(Number_of_images)+ ' images will be recorded at each location.')
                    else:
                        logging.info('One image will be recorded at each location.')
                except:
                    logging.warn(text + ' is not a valid number. Please input an integer number.')
                    Number_line_edit.select_all()
                    
                total_number_frames()
        
        edit_row1 = ui.create_row_widget()
        
        edit_row1.add(ui.create_label_widget(_("FOV per Frame (nm)")))
        edit_row1.add_spacing(2)
        FOV_line_edit = ui.create_line_edit_widget()
        FOV_line_edit.on_editing_finished = FOV_finished
        edit_row1.add(FOV_line_edit)

        edit_row1.add_spacing(6)
        
        edit_row1.add(ui.create_label_widget(_("Size in Pixels per Frame")))
        edit_row1.add_spacing(2)
        Size_line_edit = ui.create_line_edit_widget()
        Size_line_edit.on_editing_finished = Size_finished
        edit_row1.add(Size_line_edit)
        
        edit_row1.add_stretch()
        
        edit_row2 = ui.create_row_widget()
        
        edit_row2.add(ui.create_label_widget(_("Scan roation (deg)")))
        edit_row2.add_spacing(2)
        Rotation_line_edit = ui.create_line_edit_widget()
        Rotation_line_edit.on_editing_finished = Rotation_finished
        edit_row2.add(Rotation_line_edit)
        
        edit_row2.add_spacing(6)
        
        edit_row2.add(ui.create_label_widget(_("Offset (images)")))
        edit_row2.add_spacing(2)
        Offset_line_edit = ui.create_line_edit_widget()
        Offset_line_edit.on_editing_finished = Offset_finished
        edit_row2.add(Offset_line_edit)
        
        edit_row2.add_stretch()
        
        edit_row3 = ui.create_row_widget()
        
        edit_row3.add(ui.create_label_widget(_("Pixeltime (us)")))
        edit_row3.add_spacing(2)
        Time_line_edit = ui.create_line_edit_widget()
        Time_line_edit.placeholder_text = 'Number or comma-separated list of numbers'
        Time_line_edit.on_editing_finished = Time_finished
        edit_row3.add(Time_line_edit)
        
        edit_row3.add_spacing(6)
        
        edit_row3.add(ui.create_label_widget(_("Images per location")))
        edit_row3.add_spacing(2)
        Number_line_edit = ui.create_line_edit_widget()
        Number_line_edit.placeholder_text = 'Defaults to 1'
        Number_line_edit.on_editing_finished = Number_of_images
        edit_row3.add(Number_line_edit)
        
        edit_row3.add_stretch()

        bottom_button_row = ui.create_row_widget()
        top_button_row = ui.create_row_widget()
        done_button_row = ui.create_row_widget()
        checkbox_row = ui.create_row_widget()
        checkbox_row2 = ui.create_row_widget()
        tl_button = ui.create_push_button_widget(_("Top Left"))
        tr_button = ui.create_push_button_widget(_("Top Right"))
        bl_button = ui.create_push_button_widget(_("Bottom Left"))
        br_button = ui.create_push_button_widget(_("Bottom Right"))
        drive_tl = ui.create_push_button_widget(_("Top Left"))
        drive_tr = ui.create_push_button_widget(_("Top Right"))
        drive_bl = ui.create_push_button_widget(_("Bottom Left"))
        drive_br = ui.create_push_button_widget(_("Bottom Right"))
        done_button = ui.create_push_button_widget(_("Done"))
        abort_button = ui.create_push_button_widget(_("Abort"))
   
        z_drive_checkbox = ui.create_check_box_widget(_("Use Z Drive"))
        autofocus_checkbox = ui.create_check_box_widget(_("Autofocus"))
        auto_offset_checkbox = ui.create_check_box_widget(_("Auto Offset"))
        auto_rotation_checkbox = ui.create_check_box_widget(_("Auto Rotation"))
        overview_checkbox = ui.create_check_box_widget(_("Acquire Overview image"))
        overview_checkbox.check_state = 'checked'
        
        descriptor_row = ui.create_row_widget()
        descriptor_row.add(ui.create_label_widget(_("Save Coordinates")))
        descriptor_row.add_spacing(12)
        descriptor_row.add(ui.create_label_widget(_("Goto Coordinates")))
        

        def tl_button_clicked():
            save_coords('top-left')
            total_number_frames()
        def tr_button_clicked():
            save_coords('top-right')
            total_number_frames()
        def bl_button_clicked():
            save_coords('bottom-left')
            total_number_frames()
        def br_button_clicked():
            save_coords('bottom-right')
            total_number_frames()
        def drive_tl_button_clicked():
            drive_coords('top-left')
        def drive_tr_button_clicked():
            drive_coords('top-right')
        def drive_bl_button_clicked():
            drive_coords('bottom-left')
        def drive_br_button_clicked():
            drive_coords('bottom-right')
        def done_button_clicked():
            global FOV, Offset, SIze, Time, Rotation, do_autofocus, use_z_drive, auto_offset, auto_rotation, Number_of_images, acquire_overview
            
            total_number_frames()
            
            if z_drive_checkbox.check_state == 'checked':
                logging.info('Using z drive in addition to Fine Focus for focus adjustment.')
                use_z_drive = True
            else:
                logging.info('Using only Fine focus for focus adjustment')
                use_z_drive = False
                
            if autofocus_checkbox.check_state == 'checked':
                logging.info('Autofocus: ON')
                do_autofocus = True
            else:
                logging.info('Autofocus: OFF')
                do_autofocus = False
            
            if auto_rotation_checkbox.check_state == 'checked':
                logging.info('Auto Rotation: ON')
                auto_rotation= True
            else:
                logging.info('Auto Rotation: OFF')
                auto_rotation = False
            
            if auto_offset_checkbox.check_state == 'checked':
                logging.info('Auto Offset: ON')
                auto_offset = True
            else:
                logging.info('Auto Offset: OFF')
                auto_offset = False
            
            if overview_checkbox.check_state == 'checked':
                logging.info('Acquiring an overview image at the end of the mapping process.')
                acquire_overview = True
            else:
                acquire_overview = False
                
            logging.info('FOV: ' + str(FOV)+' nm')
            logging.info('Offset: ' + str(Offset)+' x image size')
            logging.info('Frame Rotation: ' + str(Rotation)+' deg')
            logging.info('Size: ' + str(Size)+' px')
            logging.info('Time: ' + str(Time)+' us')
            logging.info('Number of images per location: ' + str(Number_of_images))
            
#            try:
#                reload(vt)
#            except:
#                logging.warn('Couldn\'t reload ViennaTools!')
            
            try:
                reload(mapper)
            except:
                logging.warn('Couldn\'t reload mapper')
            
            if not None in coord_dict.viewvalues() and not None in (FOV, Size, Offset, Time, Rotation, Number_of_images):
                self.event = threading.Event()
                self.thread = threading.Thread(target=mapper.SuperScan_mapping, args=(coord_dict,), kwargs={'do_autofocus': do_autofocus, 'imsize': FOV, 'offset': Offset,\
                            'rotation': Rotation, 'number_of_images': Number_of_images, 'impix': Size, 'pixeltime': Time, 'use_z_drive': use_z_drive, \
                            'auto_offset': auto_offset, 'auto_rotation': auto_rotation, 'autofocus_pattern': 'testing', 'document_controller': document_controller, \
                            'event': self.event, 'acquire_overview': acquire_overview})
                self.thread.start()
#                mapper.SuperScan_mapping(coord_dict, do_autofocus=do_autofocus, imsize = FOV, offset = Offset, rotation = Rotation, number_of_images = Number_of_images,\
#                        impix = Size, pixeltime = Time, use_z_drive=use_z_drive, auto_offset=auto_offset, auto_rotation=auto_rotation, autofocus_pattern='testing', \
#                        acquire_overview=acquire_overview)
            else:
                logging.warn('You din\'t specify all necessary parameters.')

        def abort_button_clicked():
            #self.stop_tuning()
            logging.info('Aborting after current frame is finished. (May take a short while until actual abort)')
            self.event.set()
            self.thread.join()
            
        tl_button.on_clicked = tl_button_clicked
        tr_button.on_clicked = tr_button_clicked
        bl_button.on_clicked = bl_button_clicked
        br_button.on_clicked = br_button_clicked
        drive_tl.on_clicked = drive_tl_button_clicked
        drive_tr.on_clicked = drive_tr_button_clicked
        drive_bl.on_clicked = drive_bl_button_clicked
        drive_br.on_clicked = drive_br_button_clicked
        done_button.on_clicked = done_button_clicked
        abort_button.on_clicked = abort_button_clicked
        
        bottom_button_row.add(tl_button)
        bottom_button_row.add_spacing(2)
        bottom_button_row.add(tr_button)
        bottom_button_row.add_spacing(14)
        bottom_button_row.add(drive_tl)
        bottom_button_row.add_spacing(2)
        bottom_button_row.add(drive_tr)

        top_button_row.add(bl_button)
        top_button_row.add_spacing(2)
        top_button_row.add(br_button)
        top_button_row.add_spacing(14)
        top_button_row.add(drive_bl)
        top_button_row.add_spacing(2)
        top_button_row.add(drive_br)
        
        checkbox_row.add(z_drive_checkbox)
        checkbox_row.add_spacing(4)
        checkbox_row.add(autofocus_checkbox)
        checkbox_row.add_spacing(4)
        checkbox_row.add(auto_rotation_checkbox)        
        checkbox_row.add_spacing(4)
        checkbox_row.add(auto_offset_checkbox)
        checkbox_row.add_stretch()
        
        checkbox_row2.add(overview_checkbox)
        
        done_button_row.add(done_button)
        done_button_row.add_spacing(15)
        done_button_row.add(abort_button)
        
        column.add_spacing(15)
        column.add(edit_row1)
        column.add_spacing(5)
        column.add(edit_row2)
        column.add_spacing(5)
        column.add(edit_row3)
        column.add_spacing(15)
        column.add(descriptor_row)
        column.add_spacing(5)
        column.add(bottom_button_row)
        column.add_spacing(5)
        column.add(top_button_row)
        column.add_spacing(15)
        column.add(checkbox_row)
        column.add_spacing(5)
        column.add(checkbox_row2)
        column.add_spacing(15)
        column.add(done_button_row) 
        column.add_stretch()

        self.widget = column


workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(ScanMap, "scanmap-panel", _("SuperScan Mapping"), ["left", "right"], "right" )

def save_coords(position):
    global coord_dict
    coord_dict[position] = (vt.as2_get_control('StageOutX'), vt.as2_get_control('StageOutY'), vt.as2_get_control('StageOutZ'), vt.as2_get_control('EHTFocus'))
    logging.info('Saved x: '+str(vt.as2_get_control('StageOutX'))+', y: '+str(vt.as2_get_control('StageOutY'))+', z: '+str(vt.as2_get_control('StageOutZ'))+', focus: '+str(vt.as2_get_control('EHTFocus'))+' as '+position+' corner.')

def drive_coords(position):
    global coord_dict
    if coord_dict[position] is None:
        logging.warn('You haven\'t set the '+position+' corner yet.')
    else:
        logging.info('Going to '+str(position)+' corner: x: '+str(coord_dict[position][0])+', y: '+str(coord_dict[position][1])+', z: '+str(coord_dict[position][2])+', focus: '+str(coord_dict[position][3]))
        vt.as2_set_control('StageOutX', coord_dict[position][0])
        vt.as2_set_control('StageOutY', coord_dict[position][1])
        vt.as2_set_control('StageOutZ', coord_dict[position][2])
        vt.as2_set_control('EHTFocus', coord_dict[position][3])
        
def total_number_frames():
    global Offset, FOV, Size, Time, coord_dict, Number_of_images
    
    if not None in coord_dict.viewvalues() and not None in (Offset, FOV, Size, Time):
        corners = ('top-left', 'top-right', 'bottom-right', 'bottom-left')
        coords = []
        for corner in corners:
            coords.append(coord_dict[corner])
        coord_dict_sorted = mapper.sort_quadrangle(coords)
        coords = []
        for corner in corners:
            coords.append(coord_dict_sorted[corner])
        imsize = FOV*1e-9
        distance = Offset*imsize    
        leftX = np.min((coords[0][0],coords[3][0]))
        rightX = np.max((coords[1][0],coords[2][0]))
        topY = np.max((coords[0][1],coords[1][1]))
        botY = np.min((coords[2][1],coords[3][1]))
        num_subframes = ( int(np.abs(rightX-leftX)/(imsize+distance))+1, int(np.abs(topY-botY)/(imsize+distance))+1 )
        
        logging.info('With the current settings, %dx%d frames (%d in total) will be taken.' % (num_subframes[0], num_subframes[1], num_subframes[0]*num_subframes[1]))
        logging.info('A total area of %.4f um2 will be scanned.' % (num_subframes[0]*num_subframes[1]*(FOV*1e-3)**2))
        logging.info('Approximate mapping time: %.0f s' %(num_subframes[0]*num_subframes[1]*(Size**2*np.sum(Time)*1e-6 + 3.5)))
    