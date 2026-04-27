#ifndef CONFIG_H
#define CONFIG_H

#ifndef TB_PPU
#define TB_PPU 0
#endif
using std::string;

const int CYCLE = 10;

#if (TB_PPU == 0)
const string DATA_SRC = "./testbench/PPU_test_data/tb0.txt";
const string GOLDEN_SRC = "./testbench/PPU_test_data/tb0_golden.txt";
const int RELU_EN = 1;
const int MAXPOOL_EN = 0;
const int delay_cycle = 1;

#elif (TB_PPU == 1)
const string DATA_SRC = "./testbench/PPU_test_data/tb1.txt";
const string GOLDEN_SRC = "./testbench/PPU_test_data/tb1_golden.txt";
const int RELU_EN = 1;
const int MAXPOOL_EN = 1;
const int delay_cycle = 4;
#elif (TB_PPU == 2)
const string DATA_SRC = "./testbench/PPU_test_data/tb2.txt";
const string GOLDEN_SRC = "./testbench/PPU_test_data/tb2_golden.txt";
const int RELU_EN = 1;
const int MAXPOOL_EN = 1;
const int delay_cycle = 4;
#endif

#endif  // CONFIG_H