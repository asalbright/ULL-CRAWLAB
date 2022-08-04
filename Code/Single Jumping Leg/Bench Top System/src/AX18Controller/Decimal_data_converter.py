##########################################################################################
# Decimal_data_converter.py
#
# Script to convert decimal data of dynamixel MX-28 servo's velocity
#
# Used for gathering data 
#
# Created: 8/26/2021 Eve Dang	
#
##########################################################################################

NEGATE = {'1': '0', '0': '1'}
def negate(value):
    return ''.join(NEGATE[x] for x in value)

def binaryToDecimal(n):
    return int(n,2)

def DecimalToVelocity(num):
    if num>10000:
        bin_ = bin(num)[2:].zfill(16)
        b='0000000000000001'
  
        # Calculating binary value using function
        sum = bin(int(bin_, 2) - int(b, 2))
        c=sum[2:]  
        d=negate(c)
        dec=binaryToDecimal(d)
        return dec*0.111
    
    else: 
        return num*0.111

