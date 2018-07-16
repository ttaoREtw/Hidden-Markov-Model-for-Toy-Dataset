// -----------------------
// - Hidden Markov Model -
// -----------------------
// - File:    HMM.cpp
// - Author:  Tao, Tu
// - Date:    2018/1/3
// - Description:
//      Implement Hidden Markov Model.
// ----------------------

// #include <cmath>
#include <iostream>
#include <cstdio>
#include <fstream>
#include "HMM.h"


HMM :: HMM(int n_state, int n_observ) {
    this->n_state = n_state;
    this->n_observ = n_observ;
}

int HMM::get_n_state() {
    return this->n_state;
}

int HMM::get_n_observ() {
    return this->n_observ;
}

double** HMM :: forward(char* observ, int length) {
    int o = observ2idx(observ[0]);
    double** alpha = new2dArr(this->n_state, MAX_LINE);
    // initialize the alpha table at t = 0 (ch 4.0, p11)
    for(int s = 0; s < this->n_state; s++)
        alpha[s][0] = this->prob_init[s] * this->prob_observ[o][s];
    // fill up the table
    for(int t = 1; t < length; t++) {
        o = observ2idx(observ[t]);
        for(int s = 0; s < this->n_state; s++) {
            alpha[s][t] = 0.0;
            for(int _s = 0; _s < this->n_state; _s++)
                alpha[s][t] += alpha[_s][t-1] * this->prob_trans[_s][s];
            alpha[s][t] *= this->prob_observ[o][s];
        }
    }
    return alpha;
}

double** HMM :: backward(char* observ, int length) {
    int T = length - 1;
    int o_;
    double** beta = new2dArr(this->n_state, MAX_LINE);
    // initialize the beta table at t = T (ch 4.0, p15)
    for(int s = 0; s < this->n_state; s++)
        beta[s][T] = 1.0;
    // fill up the table
    for(int t = T-1; t >= 0; t--) {
        o_ = observ2idx(observ[t+1]);
        for(int s = 0; s < this->n_state; s++) {
            beta[s][t] = 0.0;
            for(int s_ = 0; s_ < this->n_state; s_++)
                beta[s][t] += beta[s_][t+1] * this->prob_trans[s][s_] * this->prob_observ[o_][s_];
        }
    }
    return beta;
}

double** HMM :: gamma(double** alpha, double** beta, int o_len) {
    double **r = new2dArr(this->n_state, MAX_LINE);
    double sum;
    // calculate r
    for(int t = 0; t < o_len; t++) {
        sum = 0.0;
        for(int s = 0; s < this->n_state; s++)
            sum += alpha[s][t] * beta[s][t];
        for(int s = 0; s < this->n_state; s++)
            r[s][t] = alpha[s][t] * beta[s][t] / sum;
    }
    return r;
}

double*** HMM :: epsilon(double** alpha, double** beta, char* observ, int o_len) {
    double ***e = new3dArr(this->n_state, this->n_state, MAX_LINE);
    double sum;
    int o_;
    for(int t = 0; t < o_len-1; t++) {
        sum = 0.0;
        o_ = observ2idx(observ[t+1]);
        for(int s1 = 0; s1 < this->n_state; s1++)
        for(int s2 = 0; s2 < this->n_state; s2++) {
            sum += alpha[s1][t] * this->prob_trans[s1][s2] * \
                this->prob_observ[o_][s2] * beta[s2][t+1];
        }
        for(int s1 = 0; s1 < this->n_state; s1++)
        for(int s2 = 0; s2 < this->n_state; s2++) {
            e[s1][s2][t] = alpha[s1][t] * this->prob_trans[s1][s2] * \
                this->prob_observ[o_][s2] * beta[s2][t+1] / sum;
        }
    }
    return e;
}

double** HMM :: observ_per_state(double** gamma, char* observ, int o_len) {
    int o;
    double r_sum;
    double** prob_o = new2dArr(this->n_observ, this->n_state);
    for(int s = 0; s < this->n_state; s++) {
        for(int t = 0; t < o_len; t++) {
            o = observ2idx(observ[t]);
            prob_o[o][s] += gamma[s][t];
        }
    }
    return prob_o;
}

double HMM :: viterbi(char* observ, int length) {
    int o = observ2idx(observ[0]);
    double list[MAX_STATE];
    double** delta = new2dArr(this->n_state, MAX_LINE);
    // initialize delta at t = 0 (ch 4.0, p23)
    for(int s = 0; s < this->n_state; s++)
        delta[s][0] = this->prob_init[s] * this->prob_observ[o][s];
    // fill up the table
    for(int t = 1; t < length; t++) {
        o = observ2idx(observ[t]);
        for(int s = 0; s < this->n_state; s++) {
            for(int _s = 0; _s < this->n_state; _s++)
                list[_s] = delta[_s][t-1] * this->prob_trans[_s][s];
            delta[s][t] = max(list, this->n_state) * this->prob_observ[o][s];
        }
    }
    for(int s = 0; s < this->n_state; s++)
        list[s] = delta[s][length-1];
    return max(list, this->n_state);
}

void HMM :: update(double** gamma, double*** epsilon, double** prob_o, int n_sample) {
    double e_sum, r_sum;
    // update prob_init
    for(int s = 0; s < this->n_state; s++)
        this->prob_init[s] = gamma[s][0] / n_sample;
    // update prob_trans
    for(int s1 = 0; s1 < this->n_state; s1++)
    for(int s2 = 0; s2 < this->n_state; s2++) {
        e_sum = 0.0;
        r_sum = 0.0;
        for(int t = 0; t < MAX_LINE-1; t++) {
            e_sum += epsilon[s1][s2][t];
            r_sum += gamma[s1][t];
        }
        this->prob_trans[s1][s2] = e_sum / r_sum;
    }
    // update prob_observ
    for(int k = 0; k < this->n_observ; k++)
    for(int s = 0; s < this->n_state; s++) {
        r_sum = 0.0;
        for(int t = 0; t < MAX_LINE; t++)
            r_sum += gamma[s][t];
        this->prob_observ[k][s] = prob_o[k][s] / r_sum ;
    }
}

void HMM :: load(const char* filename) {
    char dontCare[MAX_LINE];
    std::fstream f_hmm;
    f_hmm.open(filename, std::ios::in);
    
    f_hmm >> dontCare;  // eat "initial"
    f_hmm >> this->n_state;
    for(int i = 0; i < this->n_state; i++)
        f_hmm >> this->prob_init[i];

    f_hmm >> dontCare >> dontCare;  // eat "transition: <int>"
    for(int i = 0; i < this->n_state; i++)
        for(int j = 0; j < this->n_state; j++)
            f_hmm >> this->prob_trans[i][j];

    f_hmm >> dontCare;  // eat "observation:"
    f_hmm >> this->n_observ;
    for(int i = 0; i < this->n_observ; i++)
        for(int j = 0; j < this->n_state; j++)
            f_hmm >> this->prob_observ[i][j];
    f_hmm.close();
}

void HMM :: write(std::ostream &f) {
    f << "initial: " << this->n_state << std::endl;
    for(int i = 0; i < this->n_state; i++) {
         f << this->prob_init[i];
         if (i != this->n_state-1) 
            f << ' ';        
    }
    f << std::endl << std::endl << "transition: " << this->n_state << std::endl;
    for(int i = 0; i < this->n_state; i++) {
        for(int j = 0; j < this->n_state; j++) {
            f << this->prob_trans[i][j];
            if (j != this->n_state-1) 
                f << ' ';
        }
        f << std::endl;
    }
    f << std::endl << "observation: " << this->n_observ << std::endl;
    for(int i = 0; i < this->n_observ; i++) {
        for(int j = 0; j < this->n_state; j++) {
            f << this->prob_observ[i][j];
            if (j != this->n_state-1) 
                f << ' ';
        }
        f << std::endl;
    }
}

void HMM :: dump(const char* filename) {
    std::fstream f_hmm;
    f_hmm.open(filename, std::ios::out);
    this->write(f_hmm);
    f_hmm.close();
}

void HMM :: printInfo() {
    this->write(std::cout);
}

double** new2dArr(int h, int w) {
    double** table;
    table = new double*[h];
    for(int i = 0; i < h; i++) {
        table[i] = new double[w];
        for(int j = 0; j < w; j++)
            table[i][j] = 0.0; // initial value
    }
    return table;
}

double*** new3dArr(int h, int w, int d) {
    double*** table;
    table = new double**[h];
    for(int i = 0; i < h; i++)
        table[i] = new2dArr(w, d);
    return table;
}

void free2dTable(double** table, int h) {
    for(int i = 0; i < h; i++)
        delete[] table[i];
    delete[] table;
}

void free3dTable(double*** table, int h, int w) {
    for(int i = 0; i < h; i++)
        free2dTable(table[i], w);
    delete[] table;
}

double max(double* list, int length) {
    double x = -1;
    for(int i = 0; i < length; i++)
        if(list[i] > x)
            x = list[i];
    return x;
}

int argmax(double* list, int length) {
    int idx = 0;
    for(int i = 1; i < length; i++)
        if(list[i] > list[idx])
            idx = i;
    return idx;
}

int observ2idx(char observ) {
    // observ -> idx
    //      A -> 0
    //      B -> 1
    //      C -> 2
    //      D -> 3
    //      E -> 4
    //      F -> 5
    return (int)(observ - 'A');
}

void printTable(double** table, int h, int w) {
    for(int i = h-1; i >= 0; i--) {
        for(int j = 0; j < w; j++)
            printf("%.2f ", table[i][j]);
        std::cout << std::endl;
    }
}


