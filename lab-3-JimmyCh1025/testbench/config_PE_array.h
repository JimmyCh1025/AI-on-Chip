#ifndef CONFIG_PE_ARRAY_H
#define CONFIG_PE_ARRAY_H

#ifndef TBA
#error "TBA is NOT defined!"
#endif

using std::string;

const int CYCLE = 10;
const int PE_ARRAY_H = 6;
const int PE_ARRAY_W = 8;
const int FILT_ROW = 3;
const int FILT_COL = 3;
const int DEFAULT_TAG_X = 31;
const int DEFAULT_TAG_Y = 7;

#if (TBA == 0)
const string DATA_SRC = "./testbench/PE_array_test_data/tb0/";
const string CONFIG_SRC = "./testbench/PE_array_test_data/tb0/";
const string IFMAP_CONFIG_XID_FILE = CONFIG_SRC + "ifmap_config_chain_XID_tb0.txt";
const string IFMAP_CONFIG_YID_FILE = CONFIG_SRC + "ifmap_config_chain_YID_tb0.txt";
const string FILTER_CONFIG_XID_FILE = CONFIG_SRC + "filter_config_chain_XID_tb0.txt";
const string FILTER_CONFIG_YID_FILE = CONFIG_SRC + "filter_config_chain_YID_tb0.txt";
const string IPSUM_CONFIG_XID_FILE = CONFIG_SRC + "ipsum_config_chain_XID_tb0.txt";
const string IPSUM_CONFIG_YID_FILE = CONFIG_SRC + "ipsum_config_chain_YID_tb0.txt";
const string OPSUM_CONFIG_XID_FILE = CONFIG_SRC + "opsum_config_chain_XID_tb0.txt";
const string OPSUM_CONFIG_YID_FILE = CONFIG_SRC + "opsum_config_chain_YID_tb0.txt";
const string IFMAP_FILE = DATA_SRC + "ifmap_tb0.txt";
const string FILTER_FILE = DATA_SRC + "filter_tb0.txt";
const string IPSUM_FILE = DATA_SRC + "ipsum_tb0.txt";
const string OPSUM_FILE = DATA_SRC + "opsum_tb0.txt";
const int p = 4;
const int q = 4;
const int r = 1;
const int t = 2;
const int e = 8;
const int IFMAP_COL = 18;
const int OFMAP_COL = IFMAP_COL - FILT_ROW + 1;
const int LN_CONFIG = 27;

const unsigned long long PE_EN = (1ULL << (PE_ARRAY_H * PE_ARRAY_W)) - 1;
const int PE_CONFIG = ((p - 1) << 7) + ((OFMAP_COL - 1) << 2) + (q - 1);

#elif (TBA == 1)
const string DATA_SRC = "./testbench/PE_array_test_data/tb1/";
const string CONFIG_SRC = "./testbench/PE_array_test_data/tb1/";
const string IFMAP_CONFIG_XID_FILE = CONFIG_SRC + "ifmap_config_chain_XID_tb1.txt";
const string IFMAP_CONFIG_YID_FILE = CONFIG_SRC + "ifmap_config_chain_YID_tb1.txt";
const string FILTER_CONFIG_XID_FILE = CONFIG_SRC + "filter_config_chain_XID_tb1.txt";
const string FILTER_CONFIG_YID_FILE = CONFIG_SRC + "filter_config_chain_YID_tb1.txt";
const string IPSUM_CONFIG_XID_FILE = CONFIG_SRC + "ipsum_config_chain_XID_tb1.txt";
const string IPSUM_CONFIG_YID_FILE = CONFIG_SRC + "ipsum_config_chain_YID_tb1.txt";
const string OPSUM_CONFIG_XID_FILE = CONFIG_SRC + "opsum_config_chain_XID_tb1.txt";
const string OPSUM_CONFIG_YID_FILE = CONFIG_SRC + "opsum_config_chain_YID_tb1.txt";
const string IFMAP_FILE = DATA_SRC + "ifmap_tb1.txt";
const string FILTER_FILE = DATA_SRC + "filter_tb1.txt";
const string IPSUM_FILE = DATA_SRC + "ipsum_tb1.txt";
const string OPSUM_FILE = DATA_SRC + "opsum_tb1.txt";
const int p = 4;
const int q = 3;
const int r = 1;
const int t = 2;
const int e = 8;
const int IFMAP_COL = 34;
const int OFMAP_COL = IFMAP_COL - FILT_ROW + 1;
const int LN_CONFIG = 27;
const unsigned long long PE_EN = (1ULL << (PE_ARRAY_H * PE_ARRAY_W)) - 1;
const int PE_CONFIG = ((p - 1) << 7) + ((OFMAP_COL - 1) << 2) + (q - 1);

#elif (TBA == 2)
const string DATA_SRC = "./testbench/PE_array_test_data/tb2/";
const string CONFIG_SRC = "./testbench/PE_array_test_data/tb2/";
const string IFMAP_CONFIG_XID_FILE = CONFIG_SRC + "ifmap_config_chain_XID_tb2.txt";
const string IFMAP_CONFIG_YID_FILE = CONFIG_SRC + "ifmap_config_chain_YID_tb2.txt";
const string FILTER_CONFIG_XID_FILE = CONFIG_SRC + "filter_config_chain_XID_tb2.txt";
const string FILTER_CONFIG_YID_FILE = CONFIG_SRC + "filter_config_chain_YID_tb2.txt";
const string IPSUM_CONFIG_XID_FILE = CONFIG_SRC + "ipsum_config_chain_XID_tb2.txt";
const string IPSUM_CONFIG_YID_FILE = CONFIG_SRC + "ipsum_config_chain_YID_tb2.txt";
const string OPSUM_CONFIG_XID_FILE = CONFIG_SRC + "opsum_config_chain_XID_tb2.txt";
const string OPSUM_CONFIG_YID_FILE = CONFIG_SRC + "opsum_config_chain_YID_tb2.txt";
const string IFMAP_FILE = DATA_SRC + "ifmap_tb2.txt";
const string FILTER_FILE = DATA_SRC + "filter_tb2.txt";
const string IPSUM_FILE = DATA_SRC + "ipsum_tb2.txt";
const string OPSUM_FILE = DATA_SRC + "opsum_tb2.txt";
const int p = 4;
const int q = 4;
const int r = 2;
const int t = 1;
const int e = 8;
const int IFMAP_COL = 18;
const int OFMAP_COL = IFMAP_COL - FILT_ROW + 1;
const int LN_CONFIG = 31;
const unsigned long long PE_EN = (1ULL << (PE_ARRAY_H * PE_ARRAY_W)) - 1;
const int PE_CONFIG = ((p - 1) << 7) + ((OFMAP_COL - 1) << 2) + (q - 1);

#elif (TBA == 3)
const string DATA_SRC = "./testbench/PE_array_test_data/tb3/";
const string CONFIG_SRC = "./testbench/PE_array_test_data/tb3/";
const string IFMAP_CONFIG_XID_FILE = CONFIG_SRC + "ifmap_config_chain_XID_tb3.txt";
const string IFMAP_CONFIG_YID_FILE = CONFIG_SRC + "ifmap_config_chain_YID_tb3.txt";
const string FILTER_CONFIG_XID_FILE = CONFIG_SRC + "filter_config_chain_XID_tb3.txt";
const string FILTER_CONFIG_YID_FILE = CONFIG_SRC + "filter_config_chain_YID_tb3.txt";
const string IPSUM_CONFIG_XID_FILE = CONFIG_SRC + "ipsum_config_chain_XID_tb3.txt";
const string IPSUM_CONFIG_YID_FILE = CONFIG_SRC + "ipsum_config_chain_YID_tb3.txt";
const string OPSUM_CONFIG_XID_FILE = CONFIG_SRC + "opsum_config_chain_XID_tb3.txt";
const string OPSUM_CONFIG_YID_FILE = CONFIG_SRC + "opsum_config_chain_YID_tb3.txt";
const string IFMAP_FILE = DATA_SRC + "ifmap_tb3.txt";
const string FILTER_FILE = DATA_SRC + "filter_tb3.txt";
const string IPSUM_FILE = DATA_SRC + "ipsum_tb3.txt";
const string OPSUM_FILE = DATA_SRC + "opsum_tb3.txt";
const int p = 4;
const int q = 4;
const int r = 2;
const int t = 2;
const int e = 4;
const int IFMAP_COL = 18;
const int OFMAP_COL = IFMAP_COL - FILT_ROW + 1;
const int LN_CONFIG = 31;
const unsigned long long PE_EN = (1ULL << (PE_ARRAY_H * PE_ARRAY_W)) - 1;
const int PE_CONFIG = ((p - 1) << 7) + ((OFMAP_COL - 1) << 2) + (q - 1);

#elif (TBA == 4)
const string DATA_SRC = "./testbench/PE_array_test_data/tb4/";
const string CONFIG_SRC = "./testbench/PE_array_test_data/tb4/";
const string IFMAP_CONFIG_XID_FILE = CONFIG_SRC + "ifmap_config_chain_XID_tb4.txt";
const string IFMAP_CONFIG_YID_FILE = CONFIG_SRC + "ifmap_config_chain_YID_tb4.txt";
const string FILTER_CONFIG_XID_FILE = CONFIG_SRC + "filter_config_chain_XID_tb4.txt";
const string FILTER_CONFIG_YID_FILE = CONFIG_SRC + "filter_config_chain_YID_tb4.txt";
const string IPSUM_CONFIG_XID_FILE = CONFIG_SRC + "ipsum_config_chain_XID_tb4.txt";
const string IPSUM_CONFIG_YID_FILE = CONFIG_SRC + "ipsum_config_chain_YID_tb4.txt";
const string OPSUM_CONFIG_XID_FILE = CONFIG_SRC + "opsum_config_chain_XID_tb4.txt";
const string OPSUM_CONFIG_YID_FILE = CONFIG_SRC + "opsum_config_chain_YID_tb4.txt";
const string IFMAP_FILE = DATA_SRC + "ifmap_tb4.txt";
const string FILTER_FILE = DATA_SRC + "filter_tb4.txt";
const string IPSUM_FILE = DATA_SRC + "ipsum_tb4.txt";
const string OPSUM_FILE = DATA_SRC + "opsum_tb4.txt";
const int p = 4;
const int q = 4;
const int r = 1;
const int t = 4;
const int e = 4;
const int IFMAP_COL = 18;
const int OFMAP_COL = IFMAP_COL - FILT_ROW + 1;
const int LN_CONFIG = 27;
const unsigned long long PE_EN = (1ULL << (PE_ARRAY_H * PE_ARRAY_W)) - 1;
const int PE_CONFIG = ((p - 1) << 7) + ((OFMAP_COL - 1) << 2) + (q - 1);

#elif (TBA == 5)
const string DATA_SRC = "./testbench/PE_array_test_data/tb5/";
const string CONFIG_SRC = "./testbench/PE_array_test_data/tb5/";
const string IFMAP_CONFIG_XID_FILE = CONFIG_SRC + "ifmap_config_chain_XID_tb5.txt";
const string IFMAP_CONFIG_YID_FILE = CONFIG_SRC + "ifmap_config_chain_YID_tb5.txt";
const string FILTER_CONFIG_XID_FILE = CONFIG_SRC + "filter_config_chain_XID_tb5.txt";
const string FILTER_CONFIG_YID_FILE = CONFIG_SRC + "filter_config_chain_YID_tb5.txt";
const string IPSUM_CONFIG_XID_FILE = CONFIG_SRC + "ipsum_config_chain_XID_tb5.txt";
const string IPSUM_CONFIG_YID_FILE = CONFIG_SRC + "ipsum_config_chain_YID_tb5.txt";
const string OPSUM_CONFIG_XID_FILE = CONFIG_SRC + "opsum_config_chain_XID_tb5.txt";
const string OPSUM_CONFIG_YID_FILE = CONFIG_SRC + "opsum_config_chain_YID_tb5.txt";
const string IFMAP_FILE = DATA_SRC + "ifmap_tb5.txt";
const string FILTER_FILE = DATA_SRC + "filter_tb5.txt";
const string IPSUM_FILE = DATA_SRC + "ipsum_tb5.txt";
const string OPSUM_FILE = DATA_SRC + "opsum_tb5.txt";
const int p = 4;
const int q = 4;
const int r = 1;
const int t = 1;
const int e = 16;
const int IFMAP_COL = 18;
const int OFMAP_COL = IFMAP_COL - FILT_ROW + 1;
const int LN_CONFIG = 27;
const unsigned long long PE_EN = (1ULL << (PE_ARRAY_H * PE_ARRAY_W)) - 1;
const int PE_CONFIG = ((p - 1) << 7) + ((OFMAP_COL - 1) << 2) + (q - 1);

#endif

#endif