"""Defines keylogger class
"""
import pyxhook
from collections import defaultdict

class keyboard_logger():
    """Helper class for keylogging
    """
    
    def __init__(self):
        """Initialize keylogging wrapper
        """
        self.reset_log()
        self.hook = pyxhook.HookManager()
        self.hook.KeyDown = lambda ev : self.on_key_press(ev)
        self.hook.HookKeyboard()
        self.hook.start()

    def stop(self):
        """Stops logging thread
        """
        self.hook.cancel()

    def reset_log(self):
        """Resets any recorded key inputs
        """
        self.log = defaultdict(bool)

    def on_key_press(self, event):
        """Automatically called when key input is detected
        
        Args:
            event (pyxhook event): Event of key input
        """
        self.log[str(event.Key)] = True

    def get_key_pressed(self, keys):
        """Query key inputs that were created after the last query, or from beginning if this is the first query
        
        Args:
            keys (list): List of keyboard id in string
        
        Returns:
            list: List of bools that specifies whether the key is pressed for each keys
        """
        key_pressed = [self.log[key] for key in keys]
        self.reset_log()
        return key_pressed
        