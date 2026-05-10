/* NutShell RV64IM Test Program Linker Script
 *
 * Memory Layout:
 * - 0x80000000: .text (code + entry point)
 * - 0x80010000: .data (initialized data)
 * - 0x80020000: .bss (uninitialized data)
 * - 0x807F0000: Stack top (grows downward)
 */

OUTPUT_ARCH( "riscv" )

MEMORY {
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 8M
}

SECTIONS {
  .text 0x80000000 : {
    *(.text.entry)
    *(.text.*)
    *(.text)
    *(.rodata.*)
    *(.rodata)
  } > RAM

  .data 0x80010000 : {
    *(.data.*)
    *(.data)
  } > RAM

  .bss : {
    __bss_start = .;
    *(.bss.*)
    *(.bss)
    *(COMMON)
    __bss_end = .;
  } > RAM

  /DISCARD/ : {
    *(.note.GNU-stack)
    *(.gnu_attributes)
    *(.comment)
    *(.eh_frame)
    *(.eh_frame_hdr)
  }
}

/* Symbols */
__data_start = ADDR(.data);
__data_size = SIZEOF(.data);
__bss_size = __bss_end - __bss_start;

/* Stack top: 0x807F0000 (high end of 8MB RAM, grows downward) */
__stack_top = 0x807F0000;
