import tkinter
from tkinter import messagebox
# Create gui to say yes/no
def guiYesNo(header, question):
    '''
    Creates a gui to ask the user if yes or no

    Parameters
    ----------
    header : str
        The header of the gui
    question : str
        The question to ask the user

    Returns
    -------
    str
        "yes" if yes, "no" if no
    '''
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    answer = messagebox.askquestion(f'{header}', f'{question}')
    return answer

if __name__ == "__main__":
    while True:
        answer = guiYesNo("Question", "Do you want to continue?")
        if answer == "yes":
            print("Yes")
        elif answer == "no":
            print("No")
            break