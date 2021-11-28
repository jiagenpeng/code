import os

path = r"D:\毕设\code_end\code1"

def ui2py(path):
    uilist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.ui')]
    pylist = [os.path.splitext(uifile)[0]+".py" for uifile in uilist]
    [os.system("pyuic5 -o {pyfile} {uifile}".format(pyfile=py, uifile=ui)) for py, ui in zip(pylist, uilist)]

if __name__ == "__main__":
    ui2py(path)
