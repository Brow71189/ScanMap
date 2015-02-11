# standard libraries
import gettext
import logging

# third party libraries
# None

# local libraries
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import DataItem
from nion.ui import Binding

_ = gettext.gettext


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
        tl_button = ui.create_push_button_widget(_("Top Left"))
        tr_button = ui.create_push_button_widget(_("Top Right"))
        bl_button = ui.create_push_button_widget(_("Bottom Left"))
        br_button = ui.create_push_button_widget(_("Bottom Right"))

        def tl_button_clicked():
            logging.info("TODO: load function to save top left")
        def tr_button_clicked():
            logging.info("TODO: load function to save top right")
        def bl_button_clicked():
            logging.info("TODO: load function to save bottom left")
        def br_button_clicked():
            logging.info("TODO: load function to save bottom right")

        tl_button.on_clicked = tl_button_clicked
        tr_button.on_clicked = tr_button_clicked
        bl_button.on_clicked = bl_button_clicked
        br_button.on_clicked = br_button_clicked

        bottom_button_row.add(tl_button)
        top_button_row.add_spacing(8)
        bottom_button_row.add(tr_button)

        top_button_row.add(bl_button)
        top_button_row.add_spacing(8)
        top_button_row.add(br_button)

        column.add(edit_row)
        column.add_spacing(8)
        column.add(bottom_button_row)
        column.add_spacing(8)
        column.add(top_button_row)
        column.add_stretch()

        self.widget = column


workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(ScanMap, "scanmap-panel", _("Scan a Map"), ["left", "right"], "right" )
