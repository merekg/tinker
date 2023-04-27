import tkinter as tk
import time
from threading import Thread, Condition

# Globals
_current_time = 30

# Functions
def update_timer_loop(cv):
    global _current_time
    while True:
        cv.wait(1)
        if _current_time > 0:
            _current_time -= 1
            label.config(text = _current_time)
        else:
            break

def reset_countdown(cv):
    global _current_time
    _current_time = 30
    cv.notifyAll()

# Main
window = tk.Tk()

# condition variable for thread sync
cv = Condition()

timer_thread = Thread(target=update_timer_loop,args=[cv])
timer_thread.start()

# label
label = tk.Label(window, text = _current_time, font=("Luminari", 36,"bold italic"))
label.pack()

# button
button_image = tk.PhotoImage(file='/home/company/Downloads/IMG_3750.PNG')
button = tk.Button(window, image=button_image, command=lambda: reset_countdown(cv))
button.pack(pady=30)

window.geometry("1600x1600")
window.mainloop()
timer_thread.join()
