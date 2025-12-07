# FreeCAD Workbench init
from .wb import init_workbench


def Initialize():
    init_workbench()


def GetClassName():
    return "Gui::PythonWorkbench"
