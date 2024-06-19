import os
import matplotlib.pyplot as plt
import time

class Utils:

    def __init__(self) -> None:
        self.log_file = "generated-files/log.txt"

    def clear_screen(self):
        print("\033[H\033[J")

    def clear_log(self, dataset_path: str) -> None:
        """Go to the final line and write a separator to the new log."""
        with open(self.log_file, "a+") as file:
            file.write("===========================================================================================================================\n")
            file.write(f"Dataset: {dataset_path}\n")
            
            file.close()



    def debug(self, text, type = "debug"):

        if type == "error":
            print("\033[1;31m" + "[Error]: " + "\033[0m" + text)
                
        elif type == "warning":
            print("\033[1;33m" + "[Warning]: " + "\033[0m" + text)
        
        elif type == "info":
            print("\033[1;34m" + "[Info]: " + "\033[0m" + text)
            
        elif type == "success":
            print("\033[1;32m" + "[Success]: " + "\033[0m" + text)
            
        else:
            print("\033[1;35m" + "[Debug]: " + "\033[0m" + text)
            
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        current_time = f"{[current_time]}"
            
        with open(self.log_file, "a+") as file:
            
            file.write(f"{current_time}-{type}: {text} \n")
            file.close()
