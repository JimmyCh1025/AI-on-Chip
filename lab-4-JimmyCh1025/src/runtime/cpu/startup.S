/* startup.S - NutShell RV64IM Boot Code
 *
 * Initializes stack pointer and BSS before calling main().
 */

    .section .text.entry, "ax", @progbits
    .global _start
    .type   _start, @function
_start:
    /* Initialize stack pointer to __stack_top (defined in linker.ld) */
    la      sp, __stack_top

    /* Clear BSS section */
    la      t0, __bss_start
    la      t1, __bss_size
    beqz    t1, .Lbss_done
    add     t1, t0, t1
.Lbss_clear:
    sd      zero, 0(t0)
    addi    t0, t0, 8
    bltu    t0, t1, .Lbss_clear
.Lbss_done:

    /* Call main() */
    call    main

    /* Halt if main() returns */
.Lhalt:
    j       .Lhalt

    .size _start, . - _start
