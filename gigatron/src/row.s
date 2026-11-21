# TTLM hot loop in vCPU assembly (glcc/glink syntax).
#
# int row(unsigned char **pp)
#   One ternary matrix row: n = *p, then n index bytes b. The activations
#   live in page XPAGE: x[i] at XPAGE:2i and -x[i] at XPAGE:0x80+2i, so the
#   index byte (0x80 set for a -1 weight) *is* the address: no multiply,
#   no branch on the sign, one 16-bit add per non-zero weight.
#   Returns the sum, advances *pp past the row.

XPAGE = 0x4f

def scope():

    def code0():
        nohop()
        label('row')                       # R8 = pp
        LDW(R8);DEEK();STW(T2)             # T2 = p
        PEEK();ST(T3)                      # T3 = n
        INC(T2)
        LDI(0);STW(T0)                     # T0 = acc
        LDI(XPAGE);ST(T1+1)                # T1 = XPAGE:00
        LD(T3);_BEQ('.done')
        label('.loop')
        LDW(T2);PEEK();ST(T1);INC(T2)      # T1 low = index byte
        LDW(T1);DEEK();ADDW(T0);STW(T0)    # acc += x[b]
        LD(T3);SUBI(1);ST(T3);_BNE('.loop')
        label('.done')
        LDW(T2);DOKE(R8)                   # *pp = p
        LDW(T0);RET()

    return [('EXPORT', 'row'),
            ('CODE', 'row', code0) ]

module(code=scope(), name='row.s');

# Local Variables:
# mode: python
# indent-tabs-mode: ()
# End:
