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


_ = gettext.gettext

coord_dict = {'top-left': None, 'top-right': None, 'bottom-right': None, 'bottom-left': None}
do_autofocus = False
FOV = None
Size = None
Offset = None
Time = None

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
        
        
        edit_row1 = ui.create_row_widget()
        
        edit_row1.add(ui.create_label_widget(_("FOV per Frame (nm)")))
        edit_row1.add_spacing(12)
        FOV_line_edit = ui.create_line_edit_widget()
        FOV_line_edit.on_editing_finished = FOV_finished
        edit_row1.add(FOV_line_edit)
        
        edit_row1.add(ui.create_label_widget(_("Size in Pixels per Frame")))
        edit_row1.add_spacing(12)
        Size_line_edit = ui.create_line_edit_widget()
        Size_line_edit.on_editing_finished = Size_finished
        edit_row1.add(Size_line_edit)
        
        edit_row1.add_stretch()
        
        edit_row2 = ui.create_row_widget()
        
        edit_row2.add(ui.create_label_widget(_("Offset (images)")))
        edit_row2.add_spacing(12)
        Offset_line_edit = ui.create_line_edit_widget()
        Offset_line_edit.on_editing_finished = Offset_finished
        edit_row2.add(Offset_line_edit)
        
        edit_row2.add(ui.create_label_widget(_("Pixeltime (us)")))
        edit_row2.add_spacing(12)
        Time_line_edit = ui.create_line_edit_widget()
        Time_line_edit.on_editing_finished = Time_finished
        edit_row2.add(Time_line_edit)
        
        edit_row2.add_stretch()
        

        bottom_button_row = ui.create_row_widget()
        top_button_row = ui.create_row_widget()
        done_button_row = ui.create_row_widget()
        tl_button = ui.create_push_button_widget(_("Top Left"))
        tr_button = ui.create_push_button_widget(_("Top Right"))
        bl_button = ui.create_push_button_widget(_("Bottom Left"))
        br_button = ui.create_push_button_widget(_("Bottom Right"))
        done_button = ui.create_push_button_widget(_("Done"))
        autofocus_button = ui.create_push_button_widget(_("Autofocus"))

        def tl_button_clicked():
            save_coords('top-left')
        def tr_button_clicked():
            save_coords('top-right')
        def bl_button_clicked():
            save_coords('bottom-left')
        def br_button_clicked():
            save_coords('bottom-right')
        def done_button_clicked():
            global FOV
            global Offset
            global Size
            global Time
            global do_autofocus
            
            logging.info('FOV: ' + str(FOV))
            logging.info('Offset: ' + str(Offset))
            logging.info('Size: ' + str(Size))
            logging.info('Time: ' + str(Time))
            
            if not None in coord_dict.viewvalues():
                vt.SuperScan_mapping(coord_dict, do_autofocus=do_autofocus, imsize = FOV if FOV != None else 200, offset = Offset if Offset != None else 0.1, impix = Size if Size != None else 512, pixeltime = Time if Time != None else 4)
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

        tl_button.on_clicked = tl_button_clicked
        tr_button.on_clicked = tr_button_clicked
        bl_button.on_clicked = bl_button_clicked
        br_button.on_clicked = br_button_clicked
        done_button.on_clicked = done_button_clicked
        autofocus_button.on_clicked = autofocus_button_clicked

        bottom_button_row.add(tl_button)
        top_button_row.add_spacing(8)
        bottom_button_row.add(tr_button)

        top_button_row.add(bl_button)
        top_button_row.add_spacing(8)
        top_button_row.add(br_button)
        
        done_button_row.add(autofocus_button)
        top_button_row.add_spacing(8)
        done_button_row.add(done_button)

        column.add(edit_row1)
        column.add_spacing(8)
        column.add(edit_row2)
        column.add_spacing(8)
        column.add(bottom_button_row)
        column.add_spacing(8)
        column.add(top_button_row)
        column.add_spacing(8)
        column.add(done_button_row)        
        column.add_stretch()

        self.widget = column


workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(ScanMap, "scanmap-panel", _("Scan a Map"), ["left", "right"], "right" )

def save_coords(position):
    global coord_dict
    coord_dict[position] = (vt.as2_get_control('StageOutX'), vt.as2_get_control('StageOutY'), vt.as2_get_control('StageOutZ'), vt.as2_get_control('EHTFocus'))
    logging.info('Saved x: '+str(vt.as2_get_control('StageOutX'))+', y: '+str(vt.as2_get_control('StageOutY'))+', z: '+str(vt.as2_get_control('StageOutZ'))+', focus: '+str(vt.as2_get_control('EHTFocus'))+' as '+position+' corner.')
    
