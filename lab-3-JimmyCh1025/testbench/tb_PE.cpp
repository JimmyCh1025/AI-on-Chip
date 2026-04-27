#include <verilated.h>  // Include common routines

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "VPE.h"
#include "config_PE.h"
#include "verilated_fst_c.h"

using namespace std;

#define MAX_CYCLE 100000

#ifndef CLASSROOM_MODE
#define CLASSROOM_MODE 0
#endif

#define SEND_CONFIG 0
#define SEND_FILT 1
#define SEND_IFMAP 2
#define SEND_IPSUM 3
#define STORE_OPSUM 4

#define step(dut, fp, time)          \
    (fp)->dump((time) += CYCLE / 2); \
    (dut)->clk = !(dut)->clk;

#define set_signal(dut, signal, value) \
    (signal) = (value);                \
    (dut)->eval();

struct Index {
    int count_ifmap_col;
    int count_filter_col;
    int count_filter_num;
    int count_ipsum_ch;
    int count_ofmap_ch;
};

void print_config() {
    cout << "using TB" << TB_PE << endl;
    cout << "IFMAP_FILE: " << IFMAP_FILE << endl;
    cout << "FILT_FILE: " << FILT_FILE << endl;
    cout << "IPSUM_FILE: " << IPSUM_FILE << endl;
    cout << "OPSUM_FILE: " << OPSUM_FILE << endl;
    cout << "IFMAP_COL: " << IFMAP_COL << endl;
    cout << "OFMAP_COL: " << OFMAP_COL << endl;
    cout << "FILT_COL: " << FILT_COL << endl;
    cout << "I_CH: " << I_CH << endl;
    cout << "OFMAP_CH: " << OFMAP_CH << endl;
}

void load_test_data(vector<vector<uint8_t>>& ifmap_data, vector<vector<vector<int8_t>>>& filter_data,
                    vector<vector<int>>& ipsum_data, vector<vector<int>>& opsum_data_golden) {
    string line;
    string value;
    stringstream ss;

    ifstream ifmap_file(IFMAP_FILE);
    getline(ifmap_file, line);
    ifmap_file.close();
    ss.str(line);  // 重設 stringstream 的內容
    ss.clear();    // 清除 stringstream 的狀態
    for (int i = 0; i < IFMAP_COL; i++) {
        for (int j = 0; j < I_CH; j++) {
            if (getline(ss, value, ',')) {
                ifmap_data[i][j] = stoi(value);
            }
        }
    }

    ifstream filter_file(FILT_FILE);
    getline(filter_file, line);
    filter_file.close();
    ss.str(line);  // 重設 stringstream 的內容
    ss.clear();    // 清除 stringstream 的狀態
    for (int i = 0; i < OFMAP_CH; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < I_CH; k++) {
                if (getline(ss, value, ',')) {
                    filter_data[i][j][k] = stoi(value);
                }
            }
        }
    }

    ifstream ipsum_file(IPSUM_FILE);
    getline(ipsum_file, line);
    ipsum_file.close();
    ss.str(line);  // 重設 stringstream 的內容
    ss.clear();    // 清除 stringstream 的狀態
    for (int i = 0; i < OFMAP_COL; i++) {
        for (int j = 0; j < OFMAP_CH; j++) {
            if (getline(ss, value, ',')) {
                ipsum_data[i][j] = stoi(value);
            }
        }
    }

    ifstream opsum_file(OPSUM_FILE);
    getline(opsum_file, line);  // 用 ifstream 讀取整行數據
    opsum_file.close();         // 檔案讀取完成後關閉檔案
    ss.str(line);               // 重設 stringstream 的內容
    ss.clear();                 // 清除 stringstream 的狀態
    for (int i = 0; i < OFMAP_COL; i++) {
        for (int j = 0; j < OFMAP_CH; j++) {
            if (getline(ss, value, ',')) {
                opsum_data_golden[i][j] = stoi(value);  // 將字串轉成整數
            }
        }
    }
}

void transaction(VPE* dut, int& send_data_type, Index* index, const vector<vector<vector<int8_t>>>& filter_data,
                 const vector<vector<uint8_t>>& ifmap_data, const vector<vector<int>>& ipsum_data,
                 vector<vector<int>>& opsum_data, bool& opsum_end) {
    int send_data = 0;
    if (send_data_type != SEND_FILT) dut->filter_valid = 0;
    if (send_data_type != SEND_IFMAP) dut->ifmap_valid = 0;
    if (send_data_type != SEND_IPSUM) dut->ipsum_valid = 0;

    switch (send_data_type) {
        case SEND_CONFIG:
            set_signal(dut, dut->PE_en, 1);
            set_signal(dut, dut->i_config, ((OFMAP_CH - 1) << 7) + ((OFMAP_COL - 1) << 2) + (I_CH - 1));
            send_data_type = SEND_FILT;
            break;
        case SEND_FILT:
            set_signal(dut, dut->PE_en, 0);
            set_signal(dut, dut->i_config, 0);
            set_signal(dut, dut->filter_valid, rand() % 2);  // randomize idata_valid 0 or 1
            for (int i = 0; i < I_CH; i++)
                send_data += (filter_data[index->count_filter_num][index->count_filter_col][i] & 0xFF) << (8 * i);
            if (dut->filter_valid) {
                set_signal(dut, dut->filter, send_data);
            } else {
                set_signal(dut, dut->filter, 0);
            }
            if (dut->filter_valid && dut->filter_ready) {
                if (index->count_filter_col == (FILT_COL - 1) && index->count_filter_num == (OFMAP_CH - 1)) {
                    send_data_type = SEND_IFMAP;
                    index->count_filter_num = 0;
                    index->count_filter_col = 0;
                } else if (index->count_filter_col == FILT_COL - 1) {
                    index->count_filter_col = 0;
                    index->count_filter_num++;
                } else
                    index->count_filter_col++;
            }
            break;

        case SEND_IFMAP:
            set_signal(dut, dut->ifmap_valid, rand() % 2);  // randomize idata_valid 0 or 1

            // send_data
            for (int i = 0; i < I_CH; i++) send_data += ifmap_data[index->count_ifmap_col][i] << (8 * i);
            for (int i = I_CH; i < 4; i++) send_data += 128 << (8 * i);
            if (dut->ifmap_valid) {
                set_signal(dut, dut->ifmap, send_data);
            } else {
                set_signal(dut, dut->ifmap, 0);
            }

            // hand shake
            if (dut->ifmap_valid && dut->ifmap_ready) {
                if (index->count_ifmap_col < (FILT_COL - 1)) {
                    index->count_ifmap_col++;
                } else {
                    send_data_type = SEND_IPSUM;
                }
            }
            break;

        case SEND_IPSUM:
            set_signal(dut, dut->ipsum_valid, rand() % 2);  // randomize ipsum_valid 0 or 1
            if (dut->ipsum_valid && dut->ipsum_ready) {
                set_signal(dut, dut->ipsum, ipsum_data[index->count_ifmap_col - FILT_COL + 1][index->count_ipsum_ch]);
                if (index->count_ipsum_ch == (OFMAP_CH - 1)) {
                    index->count_ipsum_ch = 0;
                    send_data_type = STORE_OPSUM;
                } else
                    index->count_ipsum_ch++;
            } else
                dut->ipsum = 0;
            break;
        case STORE_OPSUM:
            set_signal(dut, dut->opsum_ready, rand() % 2);  // randomize opsum_ready 0 or 1
            if (dut->opsum_ready && dut->opsum_valid) {
                opsum_data[index->count_ifmap_col - FILT_COL + 1][index->count_ofmap_ch] = dut->opsum;
                if (index->count_ofmap_ch == (OFMAP_CH - 1)) {
                    index->count_ofmap_ch = 0;
                    send_data_type = SEND_IFMAP;
                    if (index->count_ifmap_col == (IFMAP_COL - 1))
                        opsum_end = 1;
                    else
                        index->count_ifmap_col++;
                } else
                    index->count_ofmap_ch++;
            }
            break;
        default:
            dut->ifmap = 0;
            dut->filter = 0;
            dut->ipsum = 0;
            dut->ifmap_valid = 0;
            dut->filter_valid = 0;
            dut->ipsum_valid = 0;
            dut->opsum_ready = 0;
            break;
    }

    // Store opsum
}

bool check_output(const vector<vector<int>>& opsum_data, const vector<vector<int>>& opsum_data_golden, int OFMAP_COL,
                  int OFMAP_CH, int time) {
    bool pass = true;
    for (int i = 0; i < OFMAP_COL; i++) {
        for (int j = 0; j < OFMAP_CH; j++) {
            if (opsum_data[i][j] != opsum_data_golden[i][j]) {
                pass = false;
                cout << "opsum[" << i << "][" << j << "]: your result = " << setw(6) << opsum_data[i][j]
                     << ", expect = " << setw(6) << opsum_data_golden[i][j] << endl;
            } else {
                cout << "opsum[" << i << "][" << j << "]: pass" << endl;
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

int main(int argc, char** argv) {
    // print_config();
    // Load test data
    vector<vector<uint8_t>> ifmap_data(IFMAP_COL, vector<uint8_t>(I_CH));
    vector<vector<vector<int8_t>>> filter_data(OFMAP_CH, vector<vector<int8_t>>(3, vector<int8_t>(I_CH)));
    vector<vector<int>> ipsum_data(OFMAP_COL, vector<int>(OFMAP_CH));
    vector<vector<int>> opsum_data(OFMAP_COL, vector<int>(OFMAP_CH));
    vector<vector<int>> opsum_data_golden(OFMAP_COL, vector<int>(OFMAP_CH));

    load_test_data(ifmap_data, filter_data, ipsum_data, opsum_data_golden);

    int time = 0;

    Verilated::traceEverOn(true);
    VerilatedFstC* fp = new VerilatedFstC();

    auto dut = new VPE;
    dut->trace(fp, 99);
    fp->open("wave/PE_wave.fst");

    int send_data_type = SEND_CONFIG;

    dut->rst = 1;
    dut->clk = 0;

    bool opsum_end = 0;

    Index index = {0};

    // Simulate until time_out or opsum_end
    while (time <= 100000) {
        step(dut, fp, time);
        dut->rst = 0;
        if (!dut->clk) {
            transaction(dut, send_data_type, &index, filter_data, ifmap_data, ipsum_data, opsum_data, opsum_end);
        }
        dut->eval();
        if (opsum_end) break;
    }
    for (int i = 0; i < 10; i++) {
        step(dut, fp, time);
        dut->eval();
    }
    // Check if the output is correct
    bool pass = check_output(opsum_data, opsum_data_golden, OFMAP_COL, OFMAP_CH, time);

    fp->close();
    dut->final();
    delete dut;

    if (CLASSROOM_MODE == 1 && !pass) return 1;
    else return 0;
}
