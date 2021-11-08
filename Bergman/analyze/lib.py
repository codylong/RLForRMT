def cast_int_then_float(s):
	try:
		k = int(s)
	except ValueError:
		k = float(s)
	return k

def pos(d): return [k for k in d if k > 0]
def min_pos_mean(df): 
    try: 
        return min(pos(df['mean']))
    except ValueError: return 0
def max_pos_mean(df): 
    try:
        return max(pos(df['mean']))
    except ValueError: return 0
