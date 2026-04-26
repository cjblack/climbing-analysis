
def pixels_to_cm(wall='2mm_basic'):
    # wall distance 6mm average = 25.3px
    # wall distance 6mm std = 1.6px
    if wall == '2mm_basic':
        px_to_cm = (6.0/25.3)*0.1 #ratio to multiply pixels by
    return px_to_cm