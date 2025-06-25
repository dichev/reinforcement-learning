from datetime import datetime

def now():
    return datetime.now().strftime('%b%d %H-%M-%S')