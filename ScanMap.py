# standard libraries
import gettext
import logging
import numpy as np
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

_ = gettext.gettext

coord_dict = {'top-left': None, 'top-right': None, 'bottom-right': None, 'bottom-left': None}
do_autofocus = False
use_z_drive = False
auto_offset=False
auto_rotation=False
FOV = None
Size = None
Offset = None
Time = None
Rotation = None

class ScanMap(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super(ScanMap, self).__init__(document_controller, panel_id, "Example")

        ui = document_controller.ui

        # user interface

        column = ui.create_column_widget()
        
        def FOV_finished(text):
            global FOV
            try:
                FOV = float(text)
                logging.info('Setting FOV to: ' + str(FOV))
            except:
                logging.warn(text + ' is not a valid FOV. Please input a floating point number.')
                
            FOV_line_edit.select_all()
            
        def Size_finished(text):
            global Size
            try:
                Size = int(text)
                logging.info('Setting Image Size to: ' + str(Size))
            except:
                logging.warn(text + ' is not a valid size. Please input an integer number.')
            
            Size_line_edit.select_all()
            
        def Offset_finished(text):
            global Offset
            try:
                Offset = float(text)
                logging.info('Setting Offset to: ' + str(Offset))
            except:
                logging.warn(text + ' is not a valid Offset. Please input a floating point number.')
            
            Offset_line_edit.select_all()
        
        def Time_finished(text):
            global Time
            try:
                Time = float(text)
                logging.info('Setting pixeltime to: ' + str(Time))
            except:
                logging.warn(text + ' is not a valid Time. Please input a floating point number.')
            
            Time_line_edit.select_all()
            
        def Rotation_finished(text):
            global Rotation
            try:
                Rotation = float(text)
                logging.info('Setting frame rotation to: ' + str(Rotation))
            except:
                logging.warn(text + ' is not a valid Frame Rotation. Please input a floating point number.')
            
            Time_line_edit.select_all()
        
        
        edit_row1 = ui.create_row_widget()
        
        edit_row1.add(ui.create_label_widget(_("FOV per Frame (nm)")))
        edit_row1.add_spacing(6)
        FOV_line_edit = ui.create_line_edit_widget()
        FOV_line_edit.on_editing_finished = FOV_finished
        edit_row1.add(FOV_line_edit)

        edit_row1.add_spacing(6)
        
        edit_row1.add(ui.create_label_widget(_("Size in Pixels per Frame")))
        edit_row1.add_spacing(6)
        Size_line_edit = ui.create_line_edit_widget()
        Size_line_edit.on_editing_finished = Size_finished
        edit_row1.add(Size_line_edit)
        
        edit_row1.add_stretch()
        
        edit_row2 = ui.create_row_widget()
        
        edit_row2.add(ui.create_label_widget(_("Scan roation (deg)")))
        edit_row2.add_spacing(6)
        Rotation_line_edit = ui.create_line_edit_widget()
        Rotation_line_edit.on_editing_finished = Rotation_finished
        edit_row2.add(Rotation_line_edit)
        
        edit_row2.add_spacing(6)
        
        edit_row2.add(ui.create_label_widget(_("Offset (images)")))
        edit_row2.add_spacing(6)
        Offset_line_edit = ui.create_line_edit_widget()
        Offset_line_edit.on_editing_finished = Offset_finished
        edit_row2.add(Offset_line_edit)
        
        edit_row2.add_spacing(6)
        
        edit_row2.add(ui.create_label_widget(_("Pixeltime (us)")))
        edit_row2.add_spacing(6)
        Time_line_edit = ui.create_line_edit_widget()
        Time_line_edit.on_editing_finished = Time_finished
        edit_row2.add(Time_line_edit)
        
        edit_row2.add_stretch()
        

        bottom_button_row = ui.create_row_widget()
        top_button_row = ui.create_row_widget()
        #autofocus_z_drive_button_row = ui.create_row_widget()
        done_button_row = ui.create_row_widget()
        checkbox_row = ui.create_row_widget()
        tl_button = ui.create_push_button_widget(_("Top Left"))
        tr_button = ui.create_push_button_widget(_("Top Right"))
        bl_button = ui.create_push_button_widget(_("Bottom Left"))
        br_button = ui.create_push_button_widget(_("Bottom Right"))
        drive_tl = ui.create_push_button_widget(_("Top Left"))
        drive_tr = ui.create_push_button_widget(_("Top Right"))
        drive_bl = ui.create_push_button_widget(_("Bottom Left"))
        drive_br = ui.create_push_button_widget(_("Bottom Right"))
        done_button = ui.create_push_button_widget(_("Done"))
   
        z_drive_checkbox = ui.create_check_box_widget(_("Use Z Drive"))
        autofocus_checkbox = ui.create_check_box_widget(_("Autofocus"))
        auto_offset_checkbox = ui.create_check_box_widget(_("Auto Offset"))
        auto_rotation_checkbox = ui.create_check_box_widget("Auto Rotation")
        auto_rotation_checkbox.check_state = 'checked'
        
        descriptor_row = ui.create_row_widget()
        descriptor_row.add(ui.create_label_widget(_("Save Coordinates")))
        descriptor_row.add_spacing(12)
        descriptor_row.add(ui.create_label_widget(_("Goto Coordinates")))
        

        def tl_button_clicked():
            save_coords('top-left')
        def tr_button_clicked():
            save_coords('top-right')
        def bl_button_clicked():
            save_coords('bottom-left')
        def br_button_clicked():
            save_coords('bottom-right')
        def drive_tl_button_clicked():
            drive_coords('top-left')
        def drive_tr_button_clicked():
            drive_coords('top-right')
        def drive_bl_button_clicked():
            drive_coords('bottom-left')
        def drive_br_button_clicked():
            drive_coords('bottom-right')
        def done_button_clicked():
            global FOV
            global Offset
            global Size
            global Time
            global Rotation
            global do_autofocus
            global use_z_drive
            global auto_offset
            global auto_rotation
            
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
            
            logging.info('FOV: ' + str(FOV)+' nm')
            logging.info('Offset: ' + str(Offset)+' x image size')
            logging.info('Frame Rotation: ' + str(Rotation)+' deg')
            logging.info('Size: ' + str(Size)+' px')
            logging.info('Time: ' + str(Time)+' us')
            
                        
            
            if not None in coord_dict.viewvalues():
                vt.SuperScan_mapping(coord_dict, do_autofocus=do_autofocus, imsize = FOV if FOV != None else 200, offset = Offset if Offset != None else 0.0, rotation = Rotation if Rotation != None else 0.0, impix = Size if Size != None else 512, pixeltime = Time if Time != None else 4, use_z_drive=use_z_drive, auto_offset=auto_offset, auto_rotation=auto_rotation)
            else:
                logging.warn('You din\'t set all 4 corners.')

        def autofocus_button_clicked():
            global do_autofocus
            
            if do_autofocus is False:
                do_autofocus = True
                logging.info('Autofocus is now ON.')
            else:
                do_autofocus = False
                logging.info('Autofocus is now OFF')
        
        def z_drive_button_clicked():
            global use_z_drive
            
            if use_z_drive is False:
                use_z_drive = True
                logging.info('Z Drive will be additionally used for focus adjusting.')
            else:
                use_z_drive = False
                logging.info('Only fine focus adjustment.')
        
        def auto_rotation_checked(state):
            logging.info(str(state))

        tl_button.on_clicked = tl_button_clicked
        tr_button.on_clicked = tr_button_clicked
        bl_button.on_clicked = bl_button_clicked
        br_button.on_clicked = br_button_clicked
        drive_tl.on_clicked = drive_tl_button_clicked
        drive_tr.on_clicked = drive_tr_button_clicked
        drive_bl.on_clicked = drive_bl_button_clicked
        drive_br.on_clicked = drive_br_button_clicked
        done_button.on_clicked = done_button_clicked
        
        bottom_button_row.add(tl_button)
        bottom_button_row.add_spacing(4)
        bottom_button_row.add(tr_button)
        bottom_button_row.add_spacing(8)
        bottom_button_row.add(drive_tl)
        bottom_button_row.add_spacing(4)
        bottom_button_row.add(drive_tr)

        top_button_row.add(bl_button)
        top_button_row.add_spacing(4)
        top_button_row.add(br_button)
        top_button_row.add_spacing(8)
        top_button_row.add(drive_bl)
        top_button_row.add_spacing(4)
        top_button_row.add(drive_br)
        
        checkbox_row.add(z_drive_checkbox)
        checkbox_row.add_spacing(4)
        checkbox_row.add(autofocus_checkbox)
        checkbox_row.add_spacing(4)
        checkbox_row.add(auto_rotation_checkbox)        
        checkbox_row.add_spacing(4)
        checkbox_row.add(auto_offset_checkbox)
        checkbox_row.add_stretch()
        
        done_button_row.add(done_button)
        
        column.add(edit_row1)
        column.add_spacing(8)
        column.add(edit_row2)
        column.add_spacing(8)
        column.add(descriptor_row)
        column.add_spacing(8)
        column.add(bottom_button_row)
        column.add_spacing(8)
        column.add(top_button_row)
        column.add_spacing(8)
        column.add(checkbox_row)
        column.add_spacing(8)
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
        logging.info('Saved x: '+str('Going to '+position+' corner: x: '+str(coord_dict[position][0])+', y: '+str(coord_dict[position][1])+', z: '+str(coord_dict[position][2])+', focus: '+str(coord_dict[position][3])))
        vt.as2_set_control('StageOutX', coord_dict[position][0])
        vt.as2_set_control('StageOutY', coord_dict[position][1])
        vt.as2_set_control('StageOutZ', coord_dict[position][2])
        vt.as2_set_control('EHTFocus', coord_dict[position][3])