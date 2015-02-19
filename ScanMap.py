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

class ScanMap(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super(ScanMap, self).__init__(document_controller, panel_id, "Example")

        ui = document_controller.ui

        # user interface

        column = ui.create_column_widget()

        edit_row = ui.create_row_widget()
        edit_row.add(ui.create_label_widget(_("Edit Field")))
        edit_row.add_spacing(12)
        edit_line_edit = ui.create_line_edit_widget()
        def editing_finished(text):
            logging.info(text)
            edit_line_edit.select_all()
        edit_line_edit.on_editing_finished = editing_finished
        edit_row.add(edit_line_edit)
        edit_row.add_stretch()

        bottom_button_row = ui.create_row_widget()
        top_button_row = ui.create_row_widget()
        done_button_row = ui.create_row_widget()
        tl_button = ui.create_push_button_widget(_("Top Left"))
        tr_button = ui.create_push_button_widget(_("Top Right"))
        bl_button = ui.create_push_button_widget(_("Bottom Left"))
        br_button = ui.create_push_button_widget(_("Bottom Right"))
        done_button = ui.create_push_button_widget(_("Done"))

        def tl_button_clicked():
            save_coords('top-left')
        def tr_button_clicked():
            save_coords('top-right')
        def bl_button_clicked():
            save_coords('bottom-left')
        def br_button_clicked():
            save_coords('bottom-right')
        def done_button_clicked():
            if not None in coord_dict.viewvalues():
                vt.SuperScan_mapping(coord_dict, imsize=200)

        tl_button.on_clicked = tl_button_clicked
        tr_button.on_clicked = tr_button_clicked
        bl_button.on_clicked = bl_button_clicked
        br_button.on_clicked = br_button_clicked
        done_button.on_clicked = done_button_clicked

        bottom_button_row.add(tl_button)
        top_button_row.add_spacing(8)
        bottom_button_row.add(tr_button)

        top_button_row.add(bl_button)
        top_button_row.add_spacing(8)
        top_button_row.add(br_button)
        
        done_button_row.add(done_button)

        column.add(edit_row)
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
    
