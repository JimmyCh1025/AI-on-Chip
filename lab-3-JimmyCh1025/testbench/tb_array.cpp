#include <verilated.h>  // Include common routines

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "VPE_array.h"
#include "config_PE_array.h"
#include "verilated_fst_c.h"

using namespace std;

#define MAX_CYCLE 100000

#ifndef CLASSROOM_MODE
#define CLASSROOM_MODE 0
#endif

#define step(dut, fp, time)   \
    (fp)->dump(time);         \
    (time) += CYCLE / 2;      \
    (dut)->clk = !(dut)->clk; \
    (dut)->eval();

#define set_signal(dut, signal, value) \
    (signal) = (value);                \
    (dut)->eval();

struct Index {
    int count_ifmap_col;
    int count_ifmap_row;
    int count_ifmap_ch;
    int count_filter_col;
    int count_filter_row;
    int count_filter_ch;
    int count_filter_num;
    int count_ipsum_col;
    int count_ipsum_row;
    int count_ipsum_ch;
    int count_ofmap_col;
    int count_ofmap_row;
    int count_ofmap_ch;
    int count_r;
    int count_t_H;
    int count_t_W;
};

void load_config_data(vector<int>& GIN_ifmap_XID_config, vector<int>& GIN_ifmap_YID_config,
                      vector<int>& GIN_filter_XID_config, vector<int>& GIN_filter_YID_config,
                      vector<int>& GIN_ipsum_XID_config, vector<int>& GIN_ipsum_YID_config,
                      vector<int>& GON_opsum_XID_config, vector<int>& GON_opsum_YID_config) {
    // Implementation to load configuration data into the vectors
    string line;
    string value;
    stringstream ss;

    ifstream ifmap_XID_config_file(IFMAP_CONFIG_XID_FILE);
    getline(ifmap_XID_config_file, line);
    ifmap_XID_config_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < PE_ARRAY_H * PE_ARRAY_W; i++) {
        if (getline(ss, value, ',')) {
            GIN_ifmap_XID_config[i] = stoi(value);
        }
    }

    ifstream ifmap_YID_config_file(IFMAP_CONFIG_YID_FILE);
    getline(ifmap_YID_config_file, line);
    ifmap_YID_config_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < PE_ARRAY_H; i++) {
        if (getline(ss, value, ',')) {
            GIN_ifmap_YID_config[i] = stoi(value);
        }
    }

    ifstream filter_XID_config_file(FILTER_CONFIG_XID_FILE);
    getline(filter_XID_config_file, line);
    filter_XID_config_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < PE_ARRAY_H * PE_ARRAY_W; i++) {
        if (getline(ss, value, ',')) {
            GIN_filter_XID_config[i] = stoi(value);
        }
    }

    ifstream filter_YID_config_file(FILTER_CONFIG_YID_FILE);
    getline(filter_YID_config_file, line);
    filter_YID_config_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < PE_ARRAY_H; i++) {
        if (getline(ss, value, ',')) {
            GIN_filter_YID_config[i] = stoi(value);
        }
    }

    ifstream ipsum_XID_config_file(IPSUM_CONFIG_XID_FILE);
    getline(ipsum_XID_config_file, line);
    ipsum_XID_config_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < PE_ARRAY_H * PE_ARRAY_W; i++) {
        if (getline(ss, value, ',')) {
            GIN_ipsum_XID_config[i] = stoi(value);
        }
    }

    ifstream ipsum_YID_config_file(IPSUM_CONFIG_YID_FILE);
    getline(ipsum_YID_config_file, line);
    ipsum_YID_config_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < PE_ARRAY_H; i++) {
        if (getline(ss, value, ',')) {
            GIN_ipsum_YID_config[i] = stoi(value);
        }
    }

    ifstream opsum_XID_config_file(OPSUM_CONFIG_XID_FILE);
    getline(opsum_XID_config_file, line);
    opsum_XID_config_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < PE_ARRAY_H * PE_ARRAY_W; i++) {
        if (getline(ss, value, ',')) {
            GON_opsum_XID_config[i] = stoi(value);
        }
    }

    ifstream opsum_YID_config_file(OPSUM_CONFIG_YID_FILE);
    getline(opsum_YID_config_file, line);
    opsum_YID_config_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < PE_ARRAY_H; i++) {
        if (getline(ss, value, ',')) {
            GON_opsum_YID_config[i] = stoi(value);
        }
    }
}

void load_data(vector<vector<vector<int>>>& ifmap, vector<vector<vector<vector<int>>>>& filter,
               vector<vector<vector<int>>>& ipsum, vector<vector<vector<int>>>& opsum_golden) {
    // Implementation to load data into the vectors
    string line;
    string value;
    stringstream ss;

    ifstream ifmap_file(IFMAP_FILE);
    getline(ifmap_file, line);
    ifmap_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < (e + FILT_ROW - 1); i++) {
        for (int j = 0; j < IFMAP_COL; j++) {
            for (int k = 0; k < q * r; k++) {
                if (getline(ss, value, ',')) {
                    ifmap[i][j][k] = stoi(value);
                }
            }
        }
    }
    ifstream filter_file(FILTER_FILE);
    getline(filter_file, line);
    filter_file.close();
    ss.str(line);
    ss.clear();
    for (int l = 0; l < p * t; l++) {
        for (int i = 0; i < FILT_ROW; i++) {
            for (int j = 0; j < FILT_COL; j++) {
                for (int k = 0; k < q * r; k++) {
                    if (getline(ss, value, ',')) {
                        filter[l][i][j][k] = stoi(value);
                    }
                }
            }
        }
    }
    ifstream ipsum_file(IPSUM_FILE);
    getline(ipsum_file, line);
    ipsum_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < e; i++) {
        for (int j = 0; j < OFMAP_COL; j++) {
            for (int k = 0; k < p * t; k++) {
                if (getline(ss, value, ',')) {
                    ipsum[i][j][k] = stoi(value);
                }
            }
        }
    }
    ifstream opsum_golden_file(OPSUM_FILE);
    getline(opsum_golden_file, line);
    opsum_golden_file.close();
    ss.str(line);
    ss.clear();
    for (int i = 0; i < e; i++) {
        for (int j = 0; j < OFMAP_COL; j++) {
            for (int k = 0; k < p * t; k++) {
                if (getline(ss, value, ',')) {
                    opsum_golden[i][j][k] = stoi(value);
                }
            }
        }
    }
}

void send_LN(VPE_array* dut, bool& set_LN) {
    dut->set_LN = 0;
    if (!set_LN) {
        dut->set_LN = 1;
        set_LN = 1;
    }
    if (set_LN)
        dut->LN_config_in = LN_CONFIG;
    else
        dut->LN_config_in = 0;
}

void send_XID_config(VPE_array* dut, vector<int>& GIN_ifmap_XID_config, vector<int>& GIN_filter_XID_config,
                     vector<int>& GIN_ipsum_XID_config, vector<int>& GON_opsum_XID_config, int& count_XID,
                     bool& ID_config_end) {
    if (count_XID < PE_ARRAY_H * PE_ARRAY_W) {
        dut->set_XID = 1;
        dut->ifmap_XID_scan_in = GIN_ifmap_XID_config[count_XID];
        dut->filter_XID_scan_in = GIN_filter_XID_config[count_XID];
        dut->ipsum_XID_scan_in = GIN_ipsum_XID_config[count_XID];
        dut->opsum_XID_scan_in = GON_opsum_XID_config[count_XID];
        count_XID++;
    } else {
        ID_config_end = 1;
        dut->set_XID = 0;
        dut->ifmap_XID_scan_in = 0;
        dut->filter_XID_scan_in = 0;
        dut->ipsum_XID_scan_in = 0;
        dut->opsum_XID_scan_in = 0;
    }
}

void send_YID_config(VPE_array* dut, vector<int>& GIN_ifmap_YID_config, vector<int>& GIN_filter_YID_config,
                     vector<int>& GIN_ipsum_YID_config, vector<int>& GON_opsum_YID_config, int& count_YID) {
    if (count_YID < PE_ARRAY_H) {
        set_signal(dut, dut->set_YID, 1);
        set_signal(dut, dut->ifmap_YID_scan_in, GIN_ifmap_YID_config[count_YID]);
        set_signal(dut, dut->filter_YID_scan_in, GIN_filter_YID_config[count_YID]);
        set_signal(dut, dut->ipsum_YID_scan_in, GIN_ipsum_YID_config[count_YID]);
        set_signal(dut, dut->opsum_YID_scan_in, GON_opsum_YID_config[count_YID]);
        count_YID++;
    } else {
        set_signal(dut, dut->set_YID, 0);
        set_signal(dut, dut->ifmap_YID_scan_in, 0);
        set_signal(dut, dut->filter_YID_scan_in, 0);
        set_signal(dut, dut->ipsum_YID_scan_in, 0);
        set_signal(dut, dut->opsum_YID_scan_in, 0);
    }
}

void send_data(VPE_array* dut, vector<vector<vector<int>>>& ifmap, vector<vector<vector<vector<int>>>>& filter,
               vector<vector<vector<int>>>& ipsum, vector<vector<vector<int>>>& opsum_golden, bool& send_PE_config_end,
               bool& send_ifmap_end, bool& send_filter_end, bool& send_ipsum_end, Index* index, int t_H, int t_W) {
    int data_in = 0;
    set_signal(dut, dut->PE_en, 0);
    set_signal(dut, dut->PE_config, 0);

    if (!send_PE_config_end) {
        set_signal(dut, dut->PE_en, PE_EN);
        set_signal(dut, dut->PE_config, PE_CONFIG);
        set_signal(dut, dut->GLB_ifmap_valid, 0);
        set_signal(dut, dut->GLB_filter_valid, 0);
        set_signal(dut, dut->GLB_ipsum_valid, 0);
        set_signal(dut, dut->GLB_opsum_ready, 0);
        send_PE_config_end = 1;
    } else if (!send_filter_end) {
        set_signal(dut, dut->GLB_ifmap_valid, 0);
        set_signal(dut, dut->GLB_filter_valid, rand() % 2);
        set_signal(dut, dut->GLB_ipsum_valid, 0);
        set_signal(dut, dut->GLB_opsum_ready, 0);
        // data
        for (int i = 0; i < q; i++) {
            data_in += (filter[index->count_filter_num][index->count_filter_row][index->count_filter_col]
                              [index->count_filter_ch + i] &
                        0xFF)
                       << (8 * i);
        }
        if (dut->GLB_filter_valid) {
            set_signal(dut, dut->GLB_data_in, data_in);
        } else {
            set_signal(dut, dut->GLB_data_in, 0);
        }

        // data tag
        set_signal(dut, dut->filter_tag_X, index->count_filter_row + FILT_ROW * (index->count_t_W));
        set_signal(dut, dut->filter_tag_Y, (index->count_r + index->count_t_H));

        // count
        if (dut->GLB_filter_valid && dut->GLB_filter_ready) {
            if (index->count_filter_ch == q * (r - 1)) {
                index->count_filter_ch = 0;
                if (index->count_filter_col == FILT_COL - 1) {
                    index->count_filter_col = 0;
                    if (index->count_filter_row == FILT_ROW - 1) {
                        index->count_filter_row = 0;
                        // count t
                        if (index->count_filter_num % p == p - 1) {
                            if (index->count_t_H == t_H - 1) {
                                index->count_t_H = 0;
                                if (index->count_t_W == t_W - 1) {
                                    index->count_t_W = 0;
                                } else {
                                    index->count_t_W++;
                                }
                            } else {
                                index->count_t_H++;
                            }
                        }
                        // count filter
                        if (index->count_filter_num == p * t - 1) {
                            index->count_filter_num = 0;
                            send_filter_end = 1;
                        } else
                            index->count_filter_num++;
                    } else
                        index->count_filter_row++;
                } else
                    index->count_filter_col++;
            } else {
                index->count_filter_ch = index->count_filter_ch + q;
            }

            if (index->count_r == r - 1) {
                index->count_r = 0;
            } else {
                index->count_r++;
            }
        }
    } else if (!send_ifmap_end) {
        set_signal(dut, dut->GLB_ifmap_valid, rand() % 2);
        set_signal(dut, dut->GLB_filter_valid, 0);
        set_signal(dut, dut->GLB_ipsum_valid, 0);
        set_signal(dut, dut->GLB_opsum_ready, 0);

        // data
        for (int i = 0; i < q; i++) {
            data_in += (ifmap[index->count_ifmap_row][index->count_ifmap_col][index->count_ifmap_ch + i] & 0xFF)
                       << (8 * i);
        }
        if (dut->GLB_ifmap_valid) {
            set_signal(dut, dut->GLB_data_in, data_in);
        } else {
            set_signal(dut, dut->GLB_data_in, 0);
        }

        // data tag
        set_signal(dut, dut->ifmap_tag_X, index->count_ifmap_row);
        set_signal(dut, dut->ifmap_tag_Y, index->count_r);

        // count
        if (dut->GLB_ifmap_valid && dut->GLB_ifmap_ready) {
            if (index->count_ifmap_ch == q * (r - 1)) {
                index->count_ifmap_ch = 0;
                if (index->count_ifmap_row == e + FILT_ROW - 2) {
                    index->count_ifmap_row = 0;
                    if (index->count_ifmap_col >= FILT_COL - 1) {
                        send_ifmap_end = 1;
                        if (index->count_ifmap_col == IFMAP_COL - 1) {
                            index->count_ifmap_col = 0;
                        } else
                            index->count_ifmap_col++;
                    } else
                        index->count_ifmap_col++;
                } else
                    index->count_ifmap_row++;
            } else {
                index->count_ifmap_ch = index->count_ifmap_ch + q;
            }

            if (index->count_r == r - 1) {
                index->count_r = 0;
            } else {
                index->count_r++;
            }
        }
    } else if (!send_ipsum_end) {
        set_signal(dut, dut->GLB_ifmap_valid, 0);
        set_signal(dut, dut->GLB_filter_valid, 0);
        set_signal(dut, dut->GLB_ipsum_valid, rand() % 2);
        set_signal(dut, dut->GLB_opsum_ready, 0);
        // data
        if (dut->GLB_ipsum_valid) {
            set_signal(dut, dut->GLB_data_in,
                       ipsum[index->count_ipsum_row][index->count_ipsum_col][index->count_ipsum_ch]);
        } else {
            set_signal(dut, dut->GLB_data_in, 0);
        }
        // data tag
        set_signal(dut, dut->ipsum_tag_X, index->count_ipsum_row + e * index->count_t_W);
        set_signal(dut, dut->ipsum_tag_Y, index->count_t_H);

        // count
        if (dut->GLB_ipsum_valid && dut->GLB_ipsum_ready) {
            // count t
            if (index->count_ipsum_ch % p == p - 1) {
                if (index->count_t_H == t_H - 1) {
                    index->count_t_H = 0;
                    if (index->count_t_W == t_W - 1) {
                        index->count_t_W = 0;
                    } else {
                        index->count_t_W++;
                    }
                } else {
                    index->count_t_H++;
                }
            }
            // count ipsum
            if (index->count_ipsum_ch == p * t - 1) {
                index->count_ipsum_ch = 0;
                if (index->count_ipsum_row == e - 1) {
                    index->count_ipsum_row = 0;
                    send_ipsum_end = 1;
                    if (index->count_ipsum_col == OFMAP_COL - 1) {
                        index->count_ifmap_col = 0;
                        index->count_ipsum_col = 0;
                    } else
                        index->count_ipsum_col++;
                } else
                    index->count_ipsum_row++;
            } else {
                index->count_ipsum_ch++;
            }
        }
    } else {
        set_signal(dut, dut->ifmap_tag_X, DEFAULT_TAG_X);
        set_signal(dut, dut->ifmap_tag_Y, DEFAULT_TAG_Y);
        set_signal(dut, dut->filter_tag_X, DEFAULT_TAG_X);
        set_signal(dut, dut->filter_tag_Y, DEFAULT_TAG_Y);
        set_signal(dut, dut->ipsum_tag_X, DEFAULT_TAG_X);
        set_signal(dut, dut->ipsum_tag_Y, DEFAULT_TAG_Y);
        set_signal(dut, dut->opsum_tag_X, DEFAULT_TAG_X);
        set_signal(dut, dut->opsum_tag_Y, DEFAULT_TAG_Y);
        set_signal(dut, dut->GLB_ifmap_valid, 0);
        set_signal(dut, dut->GLB_filter_valid, 0);
        set_signal(dut, dut->GLB_ipsum_valid, 0);
        set_signal(dut, dut->GLB_data_in, 0);
    }
}

void store_data(VPE_array* dut, vector<vector<vector<int>>>& opsum, Index* index, bool& send_opsum_end,
                bool& send_ifmap_end, bool& send_ipsum_end, int t_H, int t_W) {
    set_signal(dut, dut->GLB_ifmap_valid, 0);
    set_signal(dut, dut->GLB_filter_valid, 0);
    set_signal(dut, dut->GLB_ipsum_valid, 0);
    set_signal(dut, dut->GLB_opsum_ready, rand() % 2);  // randomize opsum_ready 0 or 1

    set_signal(dut, dut->opsum_tag_X, index->count_ofmap_row + e * index->count_t_W);
    set_signal(dut, dut->opsum_tag_Y, index->count_t_H);

    if (dut->GLB_opsum_ready && dut->GLB_opsum_valid) {
        opsum[index->count_ofmap_row][index->count_ofmap_col][index->count_ofmap_ch] = dut->GLB_data_out;

        if (index->count_ofmap_ch % p == p - 1) {
            if (index->count_t_H == t_H - 1) {
                index->count_t_H = 0;
                if (index->count_t_W == t_W - 1) {
                    index->count_t_W = 0;
                } else {
                    index->count_t_W++;
                }
            } else {
                index->count_t_H++;
            }
        }

        if (index->count_ofmap_ch == p * t - 1) {
            index->count_ofmap_ch = 0;
            if (index->count_ofmap_row == e - 1) {
                index->count_ofmap_row = 0;
                if (index->count_ofmap_col == OFMAP_COL - 1) {
                    send_opsum_end = 1;
                    index->count_ofmap_col = 0;
                } else {
                    send_ifmap_end = 0;
                    send_ipsum_end = 0;
                    index->count_ofmap_col++;
                }
            } else {
                index->count_ofmap_row++;
            }
        } else {
            index->count_ofmap_ch++;
        }
    }
}

bool check_output(const vector<vector<vector<int>>>& opsum_data, const vector<vector<vector<int>>>& opsum_data_golden,
                  int time) {
    bool pass = true;
    for (int i = 0; i < e; i++) {
        for (int j = 0; j < OFMAP_COL; j++) {
            for (int k = 0; k < p * t; k++) {
                if (opsum_data[i][j][k] != opsum_data_golden[i][j][k]) {
                    pass = false;
                    cout << "opsum[" << i << "][" << j << "][" << k << "]: your result = " << setw(6)
                         << opsum_data[i][j][k] << ", expect = " << setw(6) << opsum_data_golden[i][j][k] << endl;
                } else {
                    cout << "opsum[" << i << "][" << j << "][" << k << "]: pass" << endl;
                }
            }
        }
    }
    if (pass) {
        ifstream inFile("./art/pass.txt");
        string line;
        while (getline(inFile, line)) {
            cout << line << endl;
        }
        inFile.close();
    } else {
        ifstream inFile("./art/fail.txt");
        string line;
        while (getline(inFile, line)) {
            cout << line << endl;
        }
        inFile.close();
    }
    if (time >= MAX_CYCLE) cout << "Simulation timeout" << endl;
    return pass;
}

void print_config_data(vector<int>& GIN_ifmap_XID_config, vector<int>& GIN_ifmap_YID_config,
                       vector<int>& GIN_filter_XID_config, vector<int>& GIN_filter_YID_config,
                       vector<int>& GIN_ipsum_XID_config, vector<int>& GIN_ipsum_YID_config,
                       vector<int>& GON_opsum_XID_config, vector<int>& GON_opsum_YID_config) {
    // cout << "GIN_ifmap_XID_config:" << endl;
    // for (int i = 0; i < PE_ARRAY_H * PE_ARRAY_W; i++) {
    //     cout << "GIN_ifmap_XID_config[" << i << "] = " << GIN_ifmap_XID_config[i] << endl;
    // }
    // cout << "GIN_ifmap_YID_config:" << endl;
    // for (int i = 0; i < PE_ARRAY_H; i++) {
    //     cout << "GIN_ifmap_YID_config[" << i << "] = " << GIN_ifmap_YID_config[i] << endl;
    // }
    // cout << "GIN_filter_XID_config:" << endl;
    // for (int i = 0; i < PE_ARRAY_H * PE_ARRAY_W; i++) {
    //     cout << "GIN_filter_XID_config[" << i << "] = " << GIN_filter_XID_config[i] << endl;
    // }
    // cout << "GIN_filter_YID_config:" << endl;
    // for (int i = 0; i < PE_ARRAY_H; i++) {
    //     cout << "GIN_filter_YID_config[" << i << "] = " << GIN_filter_YID_config[i] << endl;
    // }
    // cout << "GIN_ipsum_XID_config:" << endl;
    // for (int i = 0; i < PE_ARRAY_H * PE_ARRAY_W; i++) {
    //     cout << "GIN_ipsum_XID_config[" << i << "] = " << GIN_ipsum_XID_config[i] << endl;
    // }
    // cout << "GIN_ipsum_YID_config:" << endl;
    // for (int i = 0; i < PE_ARRAY_H; i++) {
    //     cout << "GIN_ipsum_YID_config[" << i << "] = " << GIN_ipsum_YID_config[i] << endl;
    // }
    cout << "GON_opsum_XID_config:" << endl;
    for (int i = 0; i < PE_ARRAY_H * PE_ARRAY_W; i++) {
        cout << "GON_opsum_XID_config[" << i << "] = " << GON_opsum_XID_config[i] << endl;
    }
    cout << "GON_opsum_YID_config:" << endl;
    for (int i = 0; i < PE_ARRAY_H; i++) {
        cout << "GON_opsum_YID_config[" << i << "] = " << GON_opsum_YID_config[i] << endl;
    }
}

void print_data(vector<vector<vector<int>>>& ifmap, vector<vector<vector<vector<int>>>>& filter,
                vector<vector<vector<int>>>& ipsum, vector<vector<vector<int>>>& opsum_golden) {
    cout << "ifmap:" << endl;
    for (int i = 0; i < (e + FILT_ROW - 1); i++) {
        for (int j = 0; j < IFMAP_COL; j++) {
            for (int k = 0; k < q * r; k++) {
                cout << "ifmap[" << i << "][" << j << "][" << k << "] = " << ifmap[i][j][k] << " ";
            }
            cout << endl;
        }
    }
    cout << "filter:" << endl;
    for (int l = 0; l < p * t; l++) {
        for (int i = 0; i < FILT_ROW; i++) {
            for (int j = 0; j < FILT_COL; j++) {
                for (int k = 0; k < q * r; k++) {
                    cout << "filter[" << l << "][" << i << "][" << j << "][" << k << "] = " << filter[l][i][j][k]
                         << " ";
                }
                cout << endl;
            }
        }
    }
    cout << "ipsum:" << endl;
    for (int i = 0; i < e; i++) {
        for (int j = 0; j < OFMAP_COL; j++) {
            for (int k = 0; k < p * t; k++) {
                cout << "ipsum[" << i << "][" << j << "][" << k << "] = " << ipsum[i][j][k] << " ";
            }
            cout << endl;
        }
    }
    cout << "opsum_golden:" << endl;
    for (int i = 0; i < e; i++) {
        for (int j = 0; j < OFMAP_COL; j++) {
            for (int k = 0; k < p * t; k++) {
                cout << "opsum_golden[" << i << "][" << j << "][" << k << "] = " << opsum_golden[i][j][k] << " ";
            }
            cout << endl;
        }
    }
}

void reset_tag(VPE_array* dut) {
    set_signal(dut, dut->ifmap_tag_X, DEFAULT_TAG_X);
    set_signal(dut, dut->ifmap_tag_Y, DEFAULT_TAG_Y);
    set_signal(dut, dut->filter_tag_X, DEFAULT_TAG_X);
    set_signal(dut, dut->filter_tag_Y, DEFAULT_TAG_Y);
    set_signal(dut, dut->ipsum_tag_X, DEFAULT_TAG_X);
    set_signal(dut, dut->ipsum_tag_Y, DEFAULT_TAG_Y);
    set_signal(dut, dut->opsum_tag_X, DEFAULT_TAG_X);
    set_signal(dut, dut->opsum_tag_Y, DEFAULT_TAG_Y);
}

void map_para_analysis(int& t_H, int& t_W) {
    int merge_num = (e + PE_ARRAY_W - 1) / PE_ARRAY_W;

    int merged_PE_ARRAY_W = PE_ARRAY_W * merge_num;
    int merged_PE_ARRAY_H = PE_ARRAY_H / merge_num;

    int array_H_tile = merged_PE_ARRAY_H / FILT_ROW;
    int array_W_tile = merged_PE_ARRAY_W / e;

    t_H = array_H_tile / r;
    t_W = t / t_H;
    return;
}

int main(int argc, char** argv) {
    vector<int> GIN_ifmap_XID_config(PE_ARRAY_H * PE_ARRAY_W);
    vector<int> GIN_ifmap_YID_config(PE_ARRAY_H);
    vector<int> GIN_filter_XID_config(PE_ARRAY_H * PE_ARRAY_W);
    vector<int> GIN_filter_YID_config(PE_ARRAY_H);
    vector<int> GIN_ipsum_XID_config(PE_ARRAY_H * PE_ARRAY_W);
    vector<int> GIN_ipsum_YID_config(PE_ARRAY_H);
    vector<int> GON_opsum_XID_config(PE_ARRAY_H * PE_ARRAY_W);
    vector<int> GON_opsum_YID_config(PE_ARRAY_H);

    vector<vector<vector<int>>> ifmap((e + FILT_ROW - 1), vector<vector<int>>(IFMAP_COL, vector<int>(q * r)));
    vector<vector<vector<vector<int>>>> filter(
        p * t, vector<vector<vector<int>>>(FILT_ROW, vector<vector<int>>(FILT_COL, vector<int>(q * r))));
    vector<vector<vector<int>>> ipsum(e, vector<vector<int>>(OFMAP_COL, vector<int>(p * t)));
    vector<vector<vector<int>>> opsum(e, vector<vector<int>>(OFMAP_COL, vector<int>(p * t)));
    vector<vector<vector<int>>> opsum_golden(e, vector<vector<int>>(OFMAP_COL, vector<int>(p * t)));

    load_config_data(GIN_ifmap_XID_config, GIN_ifmap_YID_config, GIN_filter_XID_config, GIN_filter_YID_config,
                     GIN_ipsum_XID_config, GIN_ipsum_YID_config, GON_opsum_XID_config, GON_opsum_YID_config);

    load_data(ifmap, filter, ipsum, opsum_golden);

    Verilated::traceEverOn(true);
    VerilatedFstC* fp = new VerilatedFstC();

    auto dut = new VPE_array;
    dut->trace(fp, 99);
    fp->open("wave/PE_array_wave.fst");
    int time = 0;
    dut->rst = 1;
    dut->clk = 0;

    int count_XID = 0;
    int count_YID = 0;
    bool set_LN = 0;
    bool ID_config_end = 0;
    bool send_PE_config_end = 0;
    bool send_filter_end = 0;
    bool send_ifmap_end = 0;
    bool send_ipsum_end = 0;
    bool store_opsum_end = 0;
    Index index = {0};

    int t_H, t_W;
    map_para_analysis(t_H, t_W);

    // print_data(ifmap, filter, ipsum, opsum_golden);

    while (time <= MAX_CYCLE) {
        step(dut, fp, time);
        dut->rst = 0;
        if (!dut->clk) {
            reset_tag(dut);
            if (!ID_config_end) {
                send_LN(dut, set_LN);
                send_XID_config(dut, GIN_ifmap_XID_config, GIN_filter_XID_config, GIN_ipsum_XID_config,
                                GON_opsum_XID_config, count_XID, ID_config_end);
                send_YID_config(dut, GIN_ifmap_YID_config, GIN_filter_YID_config, GIN_ipsum_YID_config,
                                GON_opsum_YID_config, count_YID);
            } else if (!send_ipsum_end) {
                send_data(dut, ifmap, filter, ipsum, opsum_golden, send_PE_config_end, send_ifmap_end, send_filter_end,
                          send_ipsum_end, &index, t_H, t_W);
            } else {
                store_data(dut, opsum, &index, store_opsum_end, send_ifmap_end, send_ipsum_end, t_H, t_W);
            }
        }
        if (store_opsum_end) break;
    }

    bool pass = check_output(opsum, opsum_golden, time);

    fp->close();
    delete dut;
    delete fp;

    if (CLASSROOM_MODE == 1 && !pass) return 1;
    else return 0;
}