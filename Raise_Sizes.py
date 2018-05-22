sizes = [0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.1,0.12,0.14,0.16,0.2,0.24,0.28,0.32,0.4,0.48,0.56,0.64,0.72,0.8,0.88,1]
def generate_raise_size(min_bet,all_in,n):
    return min_bet +abs(all_in - min_bet)*sizes[n]

def generate_raise_sizes(min_bet,all_in):
    raise_sizes = [generate_raise_size(min_bet,all_in,n) for n in range(len(sizes))]
    return raise_sizes

def find_index(min_bet,all_in,raise_size):
    raise_sizes = generate_raise_sizes(min_bet,all_in)

    for index,r in enumerate(raise_sizes): 
        if raise_size <=r:
            if index == 0:
                return index
            else: #round to the nearest index.
                diff0 = abs(raise_size-raise_sizes[index])
                diff1 = abs(raise_size - raise_sizes[index-1])
                if diff0 <= diff1:
                    return index
                else:
                    return index - 1
    return len(raise_sizes)-1
