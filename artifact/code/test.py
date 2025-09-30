import os


if __name__ == "__main__":
    print(os.getcwd())
    print(os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0])
    print(os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0] + '/data')