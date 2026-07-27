"""MicroPython 启动入口。"""

from app import ExecutorApplication


def main():
    ExecutorApplication().run()


if __name__ == "__main__":
    main()

