// -----------------------
// - Hidden Markov Model -
// -----------------------
// - File:    HMM.h
// - Author:  Tao, Tu
// - Date:    2018/1/3
// - Description:
//      Implement Hidden Markov Model.
// ----------------------

#ifndef __HMM_MODEL__
#define __HMM_MODEL__

#define MAX_STATE 10
#define MAX_OBSERV 26
#define MAX_SEQ 200
#define MAX_LINE 256

#include <iostream>

class HMM {
private:
    int n_state;                                // number of states
    int n_observ;                               // number of observations
    double prob_init[MAX_STATE];                // initial initial propability
    double prob_trans[MAX_STATE][MAX_STATE];    // transition probability, [state_current][state_next]
    double prob_observ[MAX_OBSERV][MAX_STATE];  // observation probability
    void write(std::ostream &f);
    
public:
    HMM(int n_state=6, int n_observ=6);
    int get_n_state();
    int get_n_observ();
    double** forward(char* observ, int length);
    double** backward(char* observ, int length);
    double** gamma(double** alpha, double** beta, int o_len);
    double*** epsilon(double** alpha, double** beta, char* observ, int o_len);
    double** observ_per_state(double** gamma, char* observ, int o_len);
    double viterbi(char* observ, int length);
    void update(double** gamma, double*** epsilon, double** prob_o, int n_sample);
    void load(const char* filename);
    void dump(const char* filename);
    void printInfo();
};

// some utils
double** new2dArr(int h, int w);
double*** new3dArr(int h, int w, int d);
void free2dTable(double** table, int h);
void free3dTable(double*** table, int h, int w);
double max(double* list, int length);
int argmax(double* list, int length);
int observ2idx(char observ);
void printTable(double** table, int h, int w);

#endif
