// -----------------------
// - Hidden Markov Model -
// -----------------------
// - File:    train.cpp
// - Author:  Tao, Tu
// - Date:    2018/1/3
// - Description:
//      Train HMM models.
// ----------------------
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "HMM.h"

#define INIT_MODEL "../data/model_init.txt"

HMM& train(HMM &hmm, const char* train_path, int epoch);
double test(HMM* hmms, const char* test_path, const char* ans_path);

int main() {
    int epoch = 200;
    int n_state = 6;
    int n_observ = 6;
    double accurarcy;
    char train_data[5][100] = {
        "../data/seq_model_01.txt", 
        "../data/seq_model_02.txt",
        "../data/seq_model_03.txt", 
        "../data/seq_model_04.txt", 
        "../data/seq_model_05.txt"
    }; 
    char save_path[5][100] = {
        "../data/model_01.txt", 
        "../data/model_02.txt",
        "../data/model_03.txt", 
        "../data/model_04.txt", 
        "../data/model_05.txt"
    }; 
    // build HMM model
    HMM* hmms = new HMM[5];
    for(int i = 0; i < 5; i++) {
        hmms[i] = HMM(n_state, n_observ);
        hmms[i].load(INIT_MODEL);
    }
    
    for(int ep = 0; ep < epoch; ep++) {
        for(int i = 0; i < 5; i++) {
            train(hmms[i], train_data[i], 1);
        }
        accurarcy = test(hmms, "../data/testing_data1.txt", "../data/testing_answer.txt");
        printf("[%d]: %f\n", ep, accurarcy);
    }
    
    for(int i = 0; i < 5; i++)
        hmms[i].dump(save_path[i]);
    return 0;
}

HMM& train(HMM &hmm, const char* train_path, int epoch) {
    int ex_len, num_example;
    int n_state = hmm.get_n_state();
    int n_observ = hmm.get_n_observ();
    double **_prob_o, **prob_o, **a, **b, **_r, **r, ***_e, ***e;
    char example[MAX_LINE];
    std::fstream f_dat;
    f_dat.open(train_path, std::ios::in);
    for(int ep = 0; ep < epoch; ep++) {
        // initialize
        r = new2dArr(n_state, MAX_LINE);
        e = new3dArr(n_state, n_state, MAX_LINE);
        prob_o = new2dArr(n_observ, n_state);
        num_example = 0;
        // get training data
        while(f_dat >> example) {
            ex_len = strlen(example);
            a = hmm.forward(example, ex_len);
            b = hmm.backward(example, ex_len);
            // calculate r, e, prob_o
            _r = hmm.gamma(a, b, ex_len);
            _e = hmm.epsilon(a, b, example, ex_len);
            _prob_o = hmm.observ_per_state(_r, example, ex_len);
            
            // accumulate r
            for(int t = 0; t < ex_len; t++) 
                for(int s = 0; s < n_state; s++)
                    r[s][t] += _r[s][t];
            // accumulate e
            for(int t = 0; t < ex_len; t++) 
                for(int s1 = 0; s1 < n_state; s1++)
                for(int s2 = 0; s2 < n_state; s2++)
                    e[s1][s2][t] += _e[s1][s2][t];
            // accumulate prob_o
            for(int k = 0; k < n_observ; k++) 
                for(int s = 0; s < n_state; s++)
                    prob_o[k][s] += _prob_o[k][s];

            free2dTable(a, n_state);
            free2dTable(b, n_state);
            free2dTable(_r, n_state);
            free3dTable(_e, n_state, n_state);
            free2dTable(_prob_o, n_observ);
            num_example++;
        }
        hmm.update(r, e, prob_o, num_example);
        free2dTable(r, n_state);
        free3dTable(e, n_state, n_state);
        free2dTable(prob_o, n_observ);
        // re-read the file
        f_dat.seekg(0, std::ios::beg);
    }
    return hmm;
}

double test(HMM* hmms, const char* test_path, const char* ans_path) {
    double prob[5];
    int pred[2500];
    char example[MAX_LINE];
    char ans[MAX_LINE];
    int n = 0;
    int hit = 0;
    std::fstream f_dat;
    f_dat.open(test_path, std::ios::in);
    while(f_dat >> example) {
        for(int i = 0; i < 5; i++) {
            prob[i] = hmms[i].viterbi(example, strlen(example));
        }
        pred[n] = argmax(prob, 5) + 1;
        n++;
    }
    f_dat.close();
    n = 0;
    f_dat.open(ans_path, std::ios::in);
    while(f_dat >> ans) {
        if((int)(ans[7] - '0') == pred[n])
            hit++;
        n++;
    }
    f_dat.close();
    return (double)hit / n;
}





