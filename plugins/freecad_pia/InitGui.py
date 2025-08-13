# FreeCAD Workbench init
import FreeCADGui
from .wb import init_workbench

def Initialize():
    init_workbench()

def GetClassName():
    return "Gui::PythonWorkbench"
