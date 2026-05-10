
`include "DifftestMacros.svh"
module DiffExtCommitData(
  input         clock,
  input         enable,
  input         io_valid,
  input  [63:0] io_data,
  input  [ 7:0] io_coreid,
  input  [ 7:0] io_index
);
  wire _dummy_unused = 1'b1;
`ifndef SYNTHESIS
`ifdef DIFFTEST
`ifndef CONFIG_DIFFTEST_FPGA

import "DPI-C" function void v_difftest_CommitData (
  input   longint io_data,
  input      byte io_coreid,
  input      byte io_index
);


  always @(posedge clock) begin
    if (enable)
      v_difftest_CommitData (io_data, io_coreid, io_index);
  end
`endif // CONFIG_DIFFTEST_FPGA
`endif // DIFFTEST
`endif // SYNTHESIS
endmodule