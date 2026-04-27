#include <verilated.h>  // Include common routines

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "VPPU.h"
#include "config_PPU.h"
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

void read_file(vector<int>& data_in, vector<int>& golden, int& scaling_factor) {
    // read data_in and golden from file
    ifstream data_infile(DATA_SRC);
    if (!data_infile) {
        cerr << "Error opening file: " << DATA_SRC << endl;
        exit(1);
    }

    string line;
    getline(data_infile, line);
    scaling_factor = stoi(line);

    getline(data_infile, line);
    stringstream ss_value(line);
    string value;
    while (getline(ss_value, value, ',')) {
        data_in.push_back(stoi(value));
    }
    data_infile.close();

    // read golden from file
    ifstream golden_infile(GOLDEN_SRC);
    if (!golden_infile) {
        cerr << "Error opening file: " << GOLDEN_SRC << endl;
        exit(1);
    }
    getline(golden_infile, line);
    stringstream ss_golden(line);
    while (getline(ss_golden, value, ',')) {
        golden.push_back(stoi(value));
    }
    golden_infile.close();
}

void transaction(VPPU* dut, vector<int>& data_in, vector<int>& golden, vector<int>& cal_res, int scaling_factor,
                 int& count_data, int& count_delay, bool& test_end) {
    // cout << "count_data = " << count_data << "  count_delay = " << count_delay << endl;
    if (count_delay < delay_cycle) {
        set_signal(dut, dut->data_in, data_in[count_data]);
        set_signal(dut, dut->scaling_factor, scaling_factor);
        set_signal(dut, dut->maxpool_en, MAXPOOL_EN);
        set_signal(dut, dut->relu_sel, MAXPOOL_EN);
        set_signal(dut, dut->relu_en, RELU_EN);
        if (count_delay == 0 && MAXPOOL_EN) {
            set_signal(dut, dut->maxpool_init, 1);
        } else {
            set_signal(dut, dut->maxpool_init, 0);
        }
        count_data++;
        count_delay++;
        return;
    } else {
        cal_res.push_back(dut->data_out);
        count_delay = 0;
        if (count_data == data_in.size()) {
            test_end = true;
        }
        return;
    }
}

void check(vector<int> cal_res, vector<int> golden, int time, bool& pass) {
    if (time > MAX_CYCLE) {
        cout << "Error: Time out!" << endl;
        pass = false;
    }

    if (cal_res.size() != golden.size()) {
        cout << "Error: cal_res.size() != golden.size()" << endl;
        cout << "cal_res.size() = " << cal_res.size() << "  golden.size() = " << golden.size() << endl;
        pass = false;
    }
    if (pass) {
        for (int i = 0; i < cal_res.size(); i++) {
            cout << "cal_res[" << setw(2) << i << "] = " << setw(3) << cal_res[i];
            if (cal_res[i] != golden[i]) {
                pass = false;
                cout << ", Expected: " << setw(3) << golden[i] << endl;
            } else {
                cout << "  pass" << endl;
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
}

int main(int argc, char** argv) {
    Verilated::traceEverOn(true);
    VerilatedFstC* fp = new VerilatedFstC();
    bool test_end = false;

    auto dut = new VPPU;
    dut->trace(fp, 99);
    fp->open("wave/PPU_wave.fst");
    int time = 0;
    dut->rst = 1;
    dut->clk = 0;

    vector<int> data_in;
    vector<int> golden;
    vector<int> cal_res;
    int scaling_factor;
    int count_data = 0;
    int count_delay = 0;
    read_file(data_in, golden, scaling_factor);

    while (time <= MAX_CYCLE) {
        step(dut, fp, time);
        dut->rst = 0;
        if (!dut->clk) {
            transaction(dut, data_in, golden, cal_res, scaling_factor, count_data, count_delay, test_end);
        }
        if (test_end) break;
    }

    bool pass = true;
    check(cal_res, golden, time, pass);

    fp->close();
    delete dut;
    delete fp;
    if (CLASSROOM_MODE == 1 && !pass) return 1;
    else return 0;
}