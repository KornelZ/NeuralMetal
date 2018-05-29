
def limit_offset(offset):
    if offset <= 0.25:
        return 0.25
    elif offset <= 0.5:
        return 0.5
    elif offset <= 0.75:
        return 0.75
    else:
        return 1