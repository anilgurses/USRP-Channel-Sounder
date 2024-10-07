
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot)) 

