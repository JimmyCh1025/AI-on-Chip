#ifndef CONFIG_H
#define CONFIG_H

#ifndef TB_PE
#define TB_PE 0
#endif
using std::string;

const int CYCLE = 10;

#if (TB_PE == 0)
const string IFMAP_FILE = "./testbench/PE_test_data/tb0/ifmap_tb0.txt";
const string FILT_FILE = "./testbench/PE_test_data/tb0/filter_tb0.txt";
const string IPSUM_FILE = "./testbench/PE_test_data/tb0/ipsum_tb0.txt";
const string OPSUM_FILE = "./testbench/PE_test_data/tb0/ofmap_tb0.txt";
const int IFMAP_COL = 34;
const int OFMAP_COL = 32;
const int FILT_COL = 3;
const int I_CH = 3;
const int OFMAP_CH = 4;
#elif (TB_PE == 1)
const string IFMAP_FILE = "./testbench/PE_test_data/tb1/ifmap_tb1.txt";
const string FILT_FILE = "./testbench/PE_test_data/tb1/filter_tb1.txt";
const string IPSUM_FILE = "./testbench/PE_test_data/tb1/ipsum_tb1.txt";
const string OPSUM_FILE = "./testbench/PE_test_data/tb1/ofmap_tb1.txt";
const int IFMAP_COL = 34;
const int OFMAP_COL = 32;
const int FILT_COL = 3;
const int I_CH = 3;
const int OFMAP_CH = 4;
#elif (TB_PE == 2)
const string IFMAP_FILE = "./testbench/PE_test_data/tb2/ifmap_tb2.txt";
const string FILT_FILE = "./testbench/PE_test_data/tb2/filter_tb2.txt";
const string IPSUM_FILE = "./testbench/PE_test_data/tb2/ipsum_tb2.txt";
const string OPSUM_FILE = "./testbench/PE_test_data/tb2/ofmap_tb2.txt";
const int IFMAP_COL = 18;
const int OFMAP_COL = 16;
const int FILT_COL = 3;
const int I_CH = 4;
const int OFMAP_CH = 4;
#endif

#endif  // CONFIG_H