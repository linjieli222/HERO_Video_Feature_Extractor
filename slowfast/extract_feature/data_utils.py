def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


def convert_to_frac(frac_str):
    try:
        return [int(frac_str)]
    except ValueError:
        try:
            num, denom = frac_str.split('/')
            num, denom = int(num), int(denom)
            gcd_ = gcd(num, denom)
            return [int(num/gcd_), int(denom/gcd_)]
        except ValueError:
            return []


def gcd(a, b):
    """Calculate the Greatest Common Divisor of a and b.

    Unless b==0, the result will have the same sign as b (so that when
    b is divided by it, the result comes out positive).
    """
    while b:
        a, b = b, a % b
    return a
