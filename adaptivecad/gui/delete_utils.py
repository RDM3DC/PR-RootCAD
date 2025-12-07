from adaptivecad.command_defs import DOCUMENT


def delete_selected_feature(selected_feature):
    """Delete the selected feature from DOCUMENT and update the scene."""
    if selected_feature in DOCUMENT:
        DOCUMENT.remove(selected_feature)
        # Optionally, remove children or related features as well
        # Rebuild the scene to update the display
        return True
    return False
