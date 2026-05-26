#ifndef AXI_HPP
#define AXI_HPP

#define AXI_SIZE_BYTE  0b000
#define AXI_SIZE_HWORD 0b001
#define AXI_SIZE_WORD  0b010

#define AXI_BURST_INC 0x1

#define AXI_STRB_WORD  0b1111
#define AXI_STRB_HWORD 0b0011
#define AXI_STRB_BYTE  0b0001

#define AXI_RESP_OKAY   0x0
#define AXI_RESP_exOKAY 0x1
#define AXI_RESP_SLVERR 0x2
#define AXI_RESP_DECERR 0x3

#endif
