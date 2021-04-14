import sys
import os


class console:
    buffer = ""
    tmp = "/tmp/_multifil_tmp/"
    os.makedirs(tmp, exist_ok=True)
    tmp += "console.txt"

    @staticmethod
    def load():
        console.dump()
        with open(console.tmp, 'r') as file:
            return file.read()

    @staticmethod
    def dump():
        with open(console.tmp, 'a') as file:
            file.write(console.buffer)
        console.buffer = ""

    @staticmethod
    def clear():
        console.buffer = ""
        try:
            os.remove(console.tmp)
        except FileNotFoundError:
            # good
            pass

    @staticmethod
    def write(*args, sep=" ", end='\n'):
        for i in range(len(args)):
            console.buffer += str(args[i])
            if i < len(args) - 1:
                console.buffer += sep
        console.buffer += end

        if len(console.buffer) > 5000:
            console.dump()

    @staticmethod
    def get():
        console.dump()
        return console.load()

    @staticmethod
    def save(dst):
        import json
        with open(dst, 'w') as file:
            json.dump(console.get(), file)

    @staticmethod
    def show():
        sys.stdout.write(console.get())
        sys.stdout.flush()

    @staticmethod
    def pop():
        result = console.get()
        console.clear()
        return result

    @staticmethod
    def flush():
        console.show()
        console.clear()


console.dump()
