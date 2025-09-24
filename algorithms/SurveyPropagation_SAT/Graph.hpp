//
//  Graph.hpp
//  TheBacktrackingSurveyPropagation
/*
 Copyright 2018 Raffaele Marino
 This file is part of BSP.
 
 BSP is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 BSP is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with BSP; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

//
//  Created by Raffaele Marino on 26/11/2018.
//  Copyright © 2018 Raffaele Marino. All rights reserved.
//

#ifndef Graph_hpp
#define Graph_hpp
#define MAX_CLEN 1000
#include "Header.h"
#include "Vertex.hpp"
#define Div(a) = 1/(1-a)

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***************************** START CLAUSE CLASS **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

/*Class clause describes functional nodes into factor graphs. Each object describe a logical operator and stores all information that the algorithm needs.*/

/*class clause*/

class Clause{
public:
    
    /*public members*/
    
    Clause(void){
        _size_cl=0;
        _size_cl_init=0;
    };/*clause constructor default*/
    
    ~Clause(){};/*clause destructor*/
    
    unsigned int * _ptr_c(){ /*pointer to variable _c*/
        return &_c;
    };
    
    unsigned int * _ptr_pos(unsigned int &i){/*pointer to position i*/
        return &_vecpos[i];
    };
    
    int * _ptr_lit_int(unsigned int &i){/*pointer to literal i*/
        return &cp_lit_int[i];
    }
    
    
    bool empty(); /*clause empty*/
    
    unsigned long size(); /*clause size*/
    
    void clean(Vertex * &p_V, unsigned int &j);/*clean a clause*/
    
    void build(Vertex  * &p_V, unsigned int &j);/*build a clause*/
    
    bool _logic_operator(); /*logical operator*/
    
    bool _check();/*check clause*/
    
    friend ostream& operator<<(ostream &out, Clause &C){/*print clause on file in CNF form.*/
        list<bool>::iterator b=C._lit.begin();
        for(list<Vertex *>::iterator ___it=C.V.begin(); ___it!=C.V.end(); ++___it){
            out<<(*___it)->_var_int(*b)<<" ";
            ++b;
        }

        out<<0<<endl;
        return out;
    }
    
    
    
    Clause * operator = (Clause *C){ /*operator equal*/
        this->V=C->V;
        this->survey_cl_to_i=C->survey_cl_to_i;
        this->_lit=C->_lit;
        this->survey_cl_to_i_f_ptr=C->survey_cl_to_i_f_ptr;
        this->_vecpos=C->_vecpos;
        this->cpV=C->cpV;
        this->_v_ref=C->_v_ref;
        this->cp_lit_int=C->cp_lit_int;
        this->_vb=C->_vb;
        this->cp_lit=C->cp_lit;
        this->_c=C->_c;
        this->_l=C->_l;
        this->I_am_white=C->I_am_white;
        this->I_am_white=C->I_am_white;
        return this;
    }
    
    /*ricorda i vettori cambiano solo i valori ma non le posizioni mentre le liste cambiano anche le posizioni*/
    
    /*public variables*/
    list<Vertex *> V; /*list of variable nodes into a clause*/
    vector<Vertex *> v_V; /*vector of variable nodes into a clause*/
    list<Vertex *>::iterator _it_list_V_begin; /*iterator begin list V*/
    list<Clause *>::iterator _it_list; /*iterator in list graph*/
    list<double *> survey_cl_to_i; /*list of pointers to vector in Vertex*/
    vector<double *> v_survey_cl_to_i; /*vector of pointers to vector in Vertex*/
    list<double *>::iterator survey_begin; /*list of iterator*/
    list<bool> _lit; /*list of literal into a clause*/
    vector<bool> v_lit; /*vector of boolean variables*/
    vector<int> _go_forward;
    vector<double  *> survey_cl_to_i_f_ptr; /*vector of pointers to vector in Vertex*/
    vector<Vertex *> cpV;/*vector copy of list of variable node into a clause*/
    vector<double > survey_cl_to_i_f; /*vector of surveys frozen*/
    vector<double> update;
    vector<double> old_s;
    vector<double> div_s;
    vector<unsigned int> _vecpos; /*vector of integer that describes literal position*/
    vector<long int> _var;
    vector<int> _v_ref;/*vector of Boolean variables in integer*/
    vector<int> cp_lit_int;/*vector of Boolean variable in integer 0 AND 1*/
    vector<bool> _vb; /*vector of fixed Boolean variables */
    vector<bool> cp_lit; /*vector copy list of literal*/
    unsigned int _c; /*clause labeling*/
    unsigned int _l; /*position in vec_list*/
    unsigned int _size_cl; /*size of cl*/
    unsigned long _size_cl_init; /*size of cl*/
    bool _I_am_in_list_unsat;/*variables that verifies that a clause is not satisfied*/
    bool I_am_a_cl_true; /*variable which identifies if the clause is satisfied*/
    bool I_am_white;/*boolean variable thatt identifies if a clause has became "white" or not*/
    
private:
    /*private mebers*/
    int  _cast_b_to_i(bool flag);/*cast boolean variable into an integer*/
    
};




/*private member which casts a Boolean variable into an integer variable*/
inline int  Clause::_cast_b_to_i(bool flag){/*cast a Boolean variable to an integer value*/
    if (flag) {
        return 1;/*true*/
    }else{
        return 0;/*false*/
    }
}


/*public member which check if a clause is satisfied or not*/
inline  bool Clause::_check(){
    bool flag=false;/*boolean variable*/
    for (unsigned int i=0; i<cpV.size(); ++i) {/*loop for checking if a clause is satisfied or not*/
        if(cpV[i]->_I_am_a_fixed_variable)/*all the variables shall be fixed when the algorithm stops*/
            flag=flag or (!(cp_lit[i] xor cpV[i]->_who_I_am));/*logical operator or between literals into a clause*/
    }
    if(flag)return true;/* the function returns true or false*/
    else return false;
}

/*public member which describe the logical operator into a clause, defining, therefore, the problem*/
inline  bool Clause::_logic_operator(){/*logical operator in each clause*/
    bool flag=false;/*boolean varoiable*/
    
#ifdef SAT/*if SAT is defined than SAT problems can be solved using this code*/
    for (unsigned int i=0; i<_vb.size(); ++i) {/*loop for checking if a clause is satisfied or not*/
        flag=flag or _vb[i];/*logical operator or between literals into a clause*/
    }
#endif
    
    /*xor sat deve essere costruito con un altro programma, penso che le eqauzioni cambiano*/
    
#ifdef XORSAT/*if XORSAT is defined than XORSAT problems can be solved using this code*/
    if(V.empty()){/*this statement it true only when a clause is completely empty. When the condition is true, we can check if the clause is satisfied or not*/
        for (unsigned int i=0; i<_vb.size(); ++i) {/*loop for checking if a clause is satisfied or not*/
            flag=flag xor _vb[i];/*logical operator xor between literals into a clause*/
        }
    }
#endif
    
    
    /*per il naesat devo invece usare l'or e poi quando una e' singola e soddisfatta devo metterla a 0*/
#ifdef NAESAT/*if NAESAT is defined than NAESAT problems can be solved using this code*/
    if(V.size()<=_vb.size()-2){/*this statement becomes true when at least two literas have been fixed in the same clause*/
        bool _flag_t=false;/*Boolean variable for checking if a true variable has been fixed*/
        bool _flag_f=false;/*Boolean variable for checking if a false variable has been fixed*/
        for (unsigned int i=0; i<_vb.size(); ++i) {/*loop for checking if the a clause is satisfied or not*/
            if(_v_ref[i]==0)_flag_f=true;/*checking if at least two literals in the same clause are different*/
            if(_v_ref[i]==1)_flag_t=true;/*checking if at least two literals in the same clause are different*/
            flag=flag or _vb[i];/*logical operator or between literals into a clause*/
        }
        if(!_flag_f or !_flag_t)flag=false;/*checking if NAE condition is satisfied*/
    }
#endif
    
    if(flag)return true; /* the function returns true or false*/
    else return false;
}

/*public member that returns the size of a cluase*/
inline unsigned long Clause::size(){ /*return clause degree*/
    return _size_cl;
}

/*public member that returns if a clause is empty or not*/
inline  bool Clause::empty(){ /*return if the clause is empty*/
    return (_size_cl==0) ? true : false;
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/******************************  END CLAUSE CLASS **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/*****************************  START GRAPH CLASS **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

/*class graph*/
class Graph{
public:
    
    /*public members*/
    
    Graph(int argc, char * const argv[]){
        _argc=argc;/*copy of argc*/
        _argv.resize(_argc);/*copy of vector argv*/
        for(int i=0; i<_argc; ++i)_argv[i]=static_cast<string>(argv[i]);
        fl_bsp=false;/*flag for BSP set to zero*/
        _numb_of_dec_moves=1.;/*initialization number of decimation moves*/
        _numb_of_back_moves=1.;/*initialization number of backtracking moves*/
        _last_certitude=1.;/*initialization value of the last certitude*/
        _alpha=0.;/*density of clause set to 0.*/
        _N=0;/*number of variables set to 0*/
        _M=0;/*number of clauses set to zero*/
        _K=0;/*number of literals in a cluase set to 0. This is used for K-SAT problems*/
        _N_t=0;/*number of variables un-fixed set to 0*/
        _M_t=0;/*number of clauses un-fixed set to 0*/
        _m_t_m_1=0;
        _counter_conv=0;/*counter convergence surveys set to 0*/
        complexity_clauses=0.;/*clause complexity set to 0*/
        complexity_variables=0.;/*variable complexity set to 0*/
        complexity=0.;/*total complexity set to 0*/
        _unit_prop=0;/*unit propagation counter set to 0*/
        WellRandomInitialization();/*Initialization seed random number generator*/
    };
    
    ~Graph(){
        
    };/*default destructor*/
    
    unsigned int N(){return _N;} /*public values of N*/
    
    unsigned int M(){return _M;} /*public values of M*/
    
    double alpha(){return _alpha;} /*public values of alpha*/
    
    unsigned int K(){return _K;} /*public values of K*/
    
    void print(); /*print on file*/
    
    void read_from_file_graph();/*read an instance given as INPUT*/
    
    void write_on_file_graph(); /*build a graph and write a CNF formula*/
    
    void split_and_collect_information(); /*graph splitted*/
    
    void update_clause_messages(unsigned int  &c); /*update messages for SP algorithm*/
    
    void unit_propagation(unsigned int  & c); /*fix the variable into clause c to true and clean the graph*/
    
    void unit_propagation(); /*fix the variable into clause c to true and clean the graph*/
    
    void update_products();/*update products for new iteration*/
    
    void update_complexity_clauses(); /*update clauses complexity*/
    
    void convergence_messages(); /*compute fix points all messages*/
    
    void surveys(); /*compute surveys for each variable node*/
    
    void backtrack();
    
    void get_V(vector<Vertex> &V){ /*copy of V addresses*/
        ptrV.resize(_N);
        _cl.resize(_M);
        for(unsigned int i=0; i<_N; ++i) ptrV[i]=&V[i];
    };
    
    void print_on_file_residual_formula();/*print on file residual formula in CNF form*/
    
    void sort_V_Dec_move(); /*sort ptrV for decimation move*/
    
    void sort_V_Back_move(); /*sort ptrV for backtracking move*/
    
    void choose_var_to_fix_and_clean(); /*choose a variable to fix and clean the graph*/
    
    void clean(Vertex * &V_to_clean); /*clean the graph lists of unsitisfied clause*/
    
    void build(Vertex * &V_to_build); /*build the graph lists of unsitisfied clause*/
    
#ifdef WALKSAT/*if WALKSAT is defined then WalkSAT is called*/
    bool WalkSAT();/*call WalkSAT*/
#endif
    
    void Whitening_Solution();/*whitening procedure on a solution*/
    
    void print_surveys(); /*print surveys*/
    
    void print_only_one_sol(); /*print one solution on file */
    
    friend ostream& operator<<(ostream& out, Graph& B){ /*operator << for object graph*/
        out<<B._N_t<<" "<<B._M_t<<" "<<B.complexity<<" "<<B.complexity/(double)B._N<<" "<<(double)(B._M_t)/(double)B._N_t<<endl;
        return out;
    }
    
    void save(){/*save values for printing*/
        _v_Nt.push_back(_N_t);
        _v_M_t.push_back(_M_t);
        _v_c.push_back(complexity);
        _v_time.push_back((double)_time_conv_print);
    };
    
    void save_surveys(){/*save values for printing*/
        for(unsigned long i=0; i<ptrV.size(); ++i){
            _v_sT.push_back(ptrV[i]->_sT);
            _v_sF.push_back(ptrV[i]->_sF);
            _v_sI.push_back(ptrV[i]->_sI);
        }
        
        
    }
    
    
    /*public variables*/
    //    int CORES = static_cast<int>(thread::hardware_concurrency());
    vector<Clause> _cl; /*clauses list*/
    list<Clause *> _cl_list; /*unsitisfied clauses list*/
    vector<Vertex *> ptrV; /*vector of Vertex pointers */
    list<Vertex *> _list_fixed_element; /*list of Vertex pointers, i.e fixed variables are stored here*/
    vector<double> _v_Nt, _v_M_t, _v_c, _v_time; //vector for printing on file
    vector<double> _v_sT, _v_sF, _v_sI; //vector for printing on file
    double complexity_clauses;/*clauses complexity*/
    double complexity_variables;/*variable nodes complexity*/
    double complexity;/*graph total complexity*/
    double _comp_init;
    double _last_certitude; /*min value of certutude fixed at time t*/
    double _numb_of_dec_moves;/*number of decimation moves*/
    double _numb_of_back_moves;/*number of backtracking moves*/
    unsigned int _last_position_elment;/*position of element fixed*/
    unsigned int _counter_conv;/*counter for messages convergence*/
    bool fl_bsp;/*boolean variable for choosing between becktracking or decimation move*/
    
    
private:
    
    vector<Clause *> vec_list_cl;
    vector<long int> _ivec; /*CNF formula*/
    vector<string> _argv;/*copy of argv*/
    double _alpha; /* cluase density*/
    double div;
    unsigned int _seed;/*seed*/
    double _prod_U,_prod_S; /*product PU and PS*/
    double _pU_j_to_a, _pS_j_to_a, _pI_j_to_a; /*surveys for each variable into the clause*/
    unsigned int _seed_out; /*seed of CNF formula from INPUT file*/
    unsigned int _seed_in; /*seed of CNF formula for OUTPUT file*/
    unsigned int _N, _M, _K; /* order of graph and numebr of clauses*/
    unsigned int _N_t, _M_t; /*number of variable and clauses after t decimations*/
    unsigned int _unit_prop;/*unit propagation counter*/
    unsigned int _m; /*satisfied clauses counter*/
    unsigned int _m_t_m_1; /*satisfied clauses counter at time t-1*/
    unsigned int _time_conv_print; /*convergence time*/
    int _argc;/*copy of argc*/
    unsigned long s;
    double _new;
    double norm;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_TIME = chrono::high_resolution_clock::now();
    
    /*private members*/
    
    unsigned int GetRandom(){/*inizialzation seed*/
        unsigned int ris; /*definition seed*/
        FILE *devran = fopen("/dev/urandom","r");
        fread(&ris, 4, 1, devran);
        fclose(devran);
        return ris;/*obtain a random seed*/
    }
    
    void WellRandomInitialization(){/*initialization random number generator*/
        _seed=GetRandom();/*call and store seed*/
        if (_seed < 1) exit(-1);/*check if seed is ok*/
        srandom(_seed);/*initialize default random number generator*/
        unsigned int init[INITLEN];/*store first 32 random number*/
        for (int i=0;i<INITLEN;++i) {
            init[i] = (unsigned int) random();
        }
    }
    
    
    void _update_message(unsigned int  &c, list<double *>::iterator &s_i, list<Vertex *>::iterator &it_cl);
    
    double  compute_message(double &s, double &u);
    
    bool _flip();/*flip literal with probability 0.5*/
    
    double  _p_U(double  &PU, double  &PS);/*compute survey sF clause to variable*/
    
    double  _p_S(double  &PU, double &PS);/*compute survey sT clause to variable*/
    
    double  _p_I(double &PU, double &PS);/*compute survey sI clause to variable*/
    
    double Div_s(double &s);
    
    bool _conv(double &a, double &b);
    
    double _Pr_U(Vertex *V, bool b);
    
    double _Pr_S(Vertex *V, bool b, double &s);
    
    double __pu();
    double __norm();
    
    struct _Vertex_greater_pred {/*predicate for sort on a vector of Vertex*/
        bool operator()(Vertex * x, Vertex * y){
            return x->_sC > y->_sC;
        }
    };
    
    struct _Vertex_smaller_labeled_vertex_pred {/*predicate for sort on a vector of Vertex*/
        bool operator()(Vertex * x, Vertex * y){
            return x->_vertex < y->_vertex;
        }
    };
    
    
};

inline double Graph::_Pr_U(Vertex *V, bool b){/* product of unsitisfied messages*/
    return (b)?(V->prod_V_minus):(V->prod_V_plus);
}

/*public member which returns, depending by the literal into a clause, the right value of products of (1-message) for SP and BP equations*/
inline double Graph::_Pr_S(Vertex *V, bool b, double &s){ /*product of sitisfied messages*/
    return (b)?((V->prod_V_plus)*s):((V->prod_V_minus)*s);
}


inline bool Graph::_conv(double &a, double &b){
    return (abs(a-b)>epsilon) ? true:false;
}

inline double Graph::Div_s(double &s){
    return 1./(1.-s);
}


inline  bool Graph::_flip(){/*flipping literal*/
    
    if ((random() % 10000) < 5000) return false;
    
    return true;
}


inline double Graph::_p_U(double &PU, double &PS){/*compute survey sF clause to variable*/
    return ((1.- PU)*PS);
}

inline double Graph::_p_S(double &PU, double &PS){/*compute survey sT clause to variable*/
    return ((1.- PS)*PU);
}

inline double Graph::_p_I(double &PU, double &PS){/*compute survey sI clause to variable*/
    return (PU*PS);
}
//e' sbagliata!!!!
inline void Graph::_update_message(unsigned int &c, list<double *>::iterator &s_i, list<Vertex *>::iterator &it_cl){
    exit(-1);
    double a= **s_i;
    list<double *>::iterator s_j;/*complex pointer iterator for literals j into a clause*/
    list<bool>::iterator _lit_it_j;/*bool iterator for literals*/
    double _new=1.; /*new message from  clause to variable*/
    _lit_it_j=_cl[c]._lit.begin();/*iterator to first literal into the clause*/
    s_j=_cl[c].survey_cl_to_i.begin();/*iterator to first message into the clause*/
    for (list<Vertex *>::iterator it_cl_j=_cl[c].V.begin(); it_cl_j!=_cl[c].V.end(); ++it_cl_j) {
        if((*it_cl)->_vertex!=(*it_cl_j)->_vertex){
            // _new*=compute_message(it_cl_j,s_j, _lit_it_j);/*new message from cl to variable is computed*/
        }
        ++s_j;/*survey iterator updating*/
        ++_lit_it_j;/*literal iterator updating*/
    }
    
    _new=(_new<ZERO)?(0.):(_new);/*set to 0 a message iff the message is smaller than 1e-16*/
    *(*s_i)=_new;/*update new message value*/
    if(abs(*(*s_i)-a)>epsilon){
        ++_counter_conv; //not-converged messages counter
    }
}


// quste sono le equazioni che mi fregano, infatti questo e' il mio bottleneck.
// devo capire bene come affrontare questo punto. Devo usare le equazioni scrittee su mezared et al.

inline double Graph::compute_message(double &s, double &u){
    _new=(s/u);
    return (_new<ZERO)?(0.):(_new);/*new message from cl to variable is computed*/
    
}

inline double Graph::__pu(){
    return (1.-_prod_U)*_prod_S;
}

inline double Graph::__norm(){
    return (_prod_S+_prod_U-(rho_SP*_prod_S*_prod_U));
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/******************************  END GRAPH CLASS  **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
#endif /* Graph_hpp */

