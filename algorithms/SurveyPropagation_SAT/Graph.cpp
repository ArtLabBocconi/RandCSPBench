//
//  Graph.cpp
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

#include "Graph.hpp"

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***************************** START CLAUSE CLASS **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


/*public member class Clause. This member helps to clean the graph from satisfied clauses and
 from unsitisfied literal in unsitisified clauses. It starts to store the value of the message from the clause j
 to the variable i, the one which has been fixed previously. The member store the survey in survey_cl_to_i_f
 and set in variable i the survey _surveys_cl_to_i[j] equl to 1. Then check if the clause is satisfied or not*/

void Clause::clean(Vertex * &p_V, unsigned int &j){/*clean the clasue*/
    bool flag_deg=true;
    survey_cl_to_i_f[*imag(p_V->_I_am_in_cl_at_init[j])]=p_V->_surveys_cl_to_i[j];/*store clause to variable survey in the clause*/
    if(cp_lit[*imag(p_V->_I_am_in_cl_at_init[j])]){
        (p_V->_who_I_am)?p_V->_surveys_cl_to_i[j]=0.:p_V->_surveys_cl_to_i[j]=1.;/*set survey clause to variable  into a Vertex object to one*/
        _vb[*imag(p_V->_I_am_in_cl_at_init[j])]=p_V->_who_I_am;/*update bool vector  _vb*/
    }else{
        (!p_V->_who_I_am)?p_V->_surveys_cl_to_i[j]=0.:p_V->_surveys_cl_to_i[j]=1.;/*set survey clause to variable  into a Vertex object to one*/
        _vb[*imag(p_V->_I_am_in_cl_at_init[j])]=(!(p_V->_who_I_am));/*update bool vector  _vb*/
    }
    _v_ref[*imag(p_V->_I_am_in_cl_at_init[j])]=_cast_b_to_i(_vb[*imag(p_V->_I_am_in_cl_at_init[j])]);
    _go_forward[*imag(p_V->_I_am_in_cl_at_init[j])]=0;
    _lit.erase(p_V->_lit_list_i[j]);/*erase literal from the clause*/
    V.erase(p_V->where_I_am[j]);/*erase variable form the clause*/
    _it_list_V_begin=V.begin();
    _size_cl--;
    survey_cl_to_i.erase(p_V->_where_surveys_are_in_cl[j]);/*erase the survey clause to i from the clause*/
    survey_begin=survey_cl_to_i.begin();
    if(I_am_a_cl_true){
        flag_deg=false;
    }
    I_am_a_cl_true=_logic_operator();/*check if the clause has been satisfied or not*/
    if(flag_deg and I_am_a_cl_true){
        /*if the clause has been satisfied then save surveys, set the other variables surveys to 0  and */
        for (unsigned int i=0; i<survey_cl_to_i_f.size(); ++i) {
            if(survey_cl_to_i_f[i]==-1.){
                survey_cl_to_i_f[i]=*survey_cl_to_i_f_ptr[i];
                *survey_cl_to_i_f_ptr[i]=0.;
            }
        }
        /*update the variable nodes degree of the other variables that have not been fixed yet.*/
        for (list<Vertex *>::iterator __it=V.begin(); __it!=V.end(); ++__it) {
            (*__it)->_degree_i--;
            if((*__it)->_degree_i<0){
                cout<<"Error degree node j less than 0"<<endl;
                cout<<"Il grado e':"<<(*__it)->_degree_i<<" "<<(*__it)->_vertex<<endl;
                exit(-1);
            }
        }
        
    }
}


/*public member class Clause. This member builds the clauses unsitisfied by the new assignment into the graph.*/
void Clause::build(Vertex * &p_V, unsigned int &j){/*build the clasue*/
    /*update the value of the survey in the clause to the last one*/
    /* and rebuild the clause*/
    this->_vb[*imag(p_V->_I_am_in_cl_at_init[j])]=false;/*set to default value*/
    this->I_am_a_cl_true=_logic_operator();/*check if the clause has been satisfied or not by another assignment*/
    this->_v_ref[*imag(p_V->_I_am_in_cl_at_init[j])]=-1;/*set to default value*/
    this->V.push_front(v_V[*imag(p_V->_I_am_in_cl_at_init[j])]);/*build variable into the clause*/
    this->_lit.push_front(this->cp_lit[*imag(p_V->_I_am_in_cl_at_init[j])]);/*build literal into the clause*/
    p_V->where_I_am[j]=(this->V.begin());/*store iterator value*/
    p_V->_lit_list_i[j]=(this->_lit.begin());/*store iterator value*/
    this->_it_list_V_begin=V.begin();/*store iterator value*/
    this->_size_cl++;/*update clause size value*/
    this->_go_forward[*imag(p_V->_I_am_in_cl_at_init[j])]=1;/*set to default value*/
    this->survey_cl_to_i.push_front(this->v_survey_cl_to_i[*imag(p_V->_I_am_in_cl_at_init[j])]);/*build the survey clause to i from the clause*/
    p_V->_where_surveys_are_in_cl[j]=(this->survey_cl_to_i.begin());/*update iterator value*/
    this->survey_begin=survey_cl_to_i.begin();/*update iterator value*/
    if(!(this->I_am_a_cl_true)){
        p_V->_surveys_cl_to_i[j]=this->survey_cl_to_i_f[*imag(p_V->_I_am_in_cl_at_init[j])];
        /*update the variable nodes degree of the other variables that have not been fixed yet.*/
        if(!(this->_I_am_in_list_unsat)){
            for (unsigned int i=0; i<this->survey_cl_to_i_f.size(); ++i) {/*restore all surveys of the clause*/
                if(this->survey_cl_to_i_f[i]!=-1. and this->_go_forward[i]==1){
                    *(this->survey_cl_to_i_f_ptr[i])=this->survey_cl_to_i_f[i];
                    *(this->v_survey_cl_to_i[i])=this->survey_cl_to_i_f[i];
                    this->survey_cl_to_i_f[i]=-1;
                }
            }
            for (list<Vertex *>::iterator __it=this->V.begin(); __it!=this->V.end(); ++__it) {
                (*__it)->_degree_i++;/*update the variable nodes degree of the other variables that have not been fixed yet.*/
            }
        }else{
            /*The litteral in an unsat clause is re-build*/
            *(this->survey_cl_to_i_f_ptr[*imag(p_V->_I_am_in_cl_at_init[j])])=this->survey_cl_to_i_f[*imag(p_V->_I_am_in_cl_at_init[j])];
            *(this->v_survey_cl_to_i[*imag(p_V->_I_am_in_cl_at_init[j])])=this->survey_cl_to_i_f[*imag(p_V->_I_am_in_cl_at_init[j])];
            this->survey_cl_to_i_f[*imag(p_V->_I_am_in_cl_at_init[j])]=-1.;/*update clause to variable survey*/
            p_V->_degree_i++;/*update the variable node degree */
        }
        
    }else{
        /*set the survey in the ture clause at 0., it will be set to the last value when the clause will be unsat*/
        p_V->_surveys_cl_to_i[j]=0.;
    }
    
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


/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/********************************** START SP ***************************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


/*public member class Graph which describe unit propagation algorithm. This member is called when a clause is
 composed only by one literal, and therefore has to be satisfied, otherwise we obtain a contraddiction*/
void Graph::unit_propagation(unsigned int  &c){ /*fix the variable into clause c to true and clean the graph*/
    //cout<<"unit propagation"<<endl;
    _unit_prop++;/*update counter unit propagation*/
    if(_cl[c].size()>1 and _new!=1)exit(-1);/*check that we have the right clause*/
    if(_cl[c].empty())exit(-1);/*check that we have the right clause*/
    if (*_cl[c]._lit.begin()){
        (*_cl[c].V.begin())->_who_I_am=true; /*set the variable to the value that satisfied the literal into the clause*/
        (*_cl[c].V.begin())->_sT=1.;/*set the associated survey to one*/
        (*_cl[c].V.begin())->_sF=0.;
        (*_cl[c].V.begin())->_sI=0.;
    }else {
        (*_cl[c].V.begin())->_who_I_am=false;/*set the variable to the value that satisfied the literal into the clause*/
        (*_cl[c].V.begin())->_sT=0.;
        (*_cl[c].V.begin())->_sF=1.;/*set the associated survey to one*/
        (*_cl[c].V.begin())->_sI=0.;
    }
    (*_cl[c].V.begin())->_sC=1.1;/*set the certitude to 1.1 to be sure that in the sort the right variable shows up*/
    (*_cl[c].V.begin())->_degree_i=0;/*set variable node degree to 0*/
    Vertex * cp=(*_cl[c].V.begin()); /*variable node copy*/
    sort(ptrV.begin()+(_list_fixed_element.size()), ptrV.end(), _Vertex_greater_pred());/*sort the vector*/
    (*_cl[c].V.begin())->_sC=1.;/*set the certitude to 1*/
    _list_fixed_element.push_back(cp);/*store the Vertex in the listt of fixed element*/
    cp->_it_list_fixed_elem=--_list_fixed_element.end();/*save its address into the varibale node*/
    cp->_I_am_a_fixed_variable=true;/*fix the variable*/
    /*clean the graph*/
    /*
     Clean the graph means:
     
     1) erasing from the factor graph all satisfied clauses;
     2) erasing literals associated to variable i, which are present in a clause that is not satisfied by the variable node assignement.
     */
    clean(cp);
    _N_t=_N-static_cast<unsigned int>(_list_fixed_element.size());/*update the number of un-fixed variable nodes*/
}

/*public member class Graph which helps us to fix the variables that have the highest value of certitude*/
void Graph::choose_var_to_fix_and_clean(){
    /*set the rangee over variables unfixed are into vertex ptrV*/
    _M_t=0;
    unsigned int _size=(unsigned int)(frac*((double)_N_t))+(unsigned int)_list_fixed_element.size();
    unsigned int _size_init=(unsigned int)_list_fixed_element.size();
    /*set unit propagation counter to zero*/
    _unit_prop=0;
    /*condition for erasing only one variable at each convergence when (frac*((double)_N_t)<1*/
    if (_size<=_size_init) {
        _size=1+_size_init;
    }
    unsigned int _counter_dec_var=0;
    for (unsigned int i=_size_init; i<_size; ++i) {
        _list_fixed_element.push_back(ptrV[i]);/*store the variable node into fixed element list*/
        ptrV[i]->_it_list_fixed_elem=--_list_fixed_element.end();/*save its address into the varibale node*/
        ptrV[i]->_I_am_a_fixed_variable=true;/*fix the variable*/
        ptrV[i]->fix_var_i();/*set the variable to the value predicted by the rule described in main.cpp*/
        if(_last_certitude>ptrV[i]->_sC)_last_certitude=ptrV[i]->_sC;/*store last certitude*/
        
        /*clean the graph*/
        /*
         Clean the graph means:
         
         1) erasing from the factor graph all satisfied clauses;
         2) erasing literals associated to variable i, which are present in clauses not satisfied by the variable node assignement.
         */
        clean(ptrV[i]);
        _counter_dec_var++;
        //cout<<_m_t_m_1<<" "<<i<<endl;
        //cout<<"il grado e': "<<ptrV[i]->_degree_i<<endl;
        ptrV[i]->_degree_i=0;/*set to 0 the degree of the variable node*/
    }
    
    _m_t_m_1=_counter_dec_var;
    _N_t=_N-static_cast<unsigned int>(_list_fixed_element.size());/*update the number of un-fixed variable nodes*/
    //update_products();
}

void Graph::clean(Vertex * &V_to_clean){
    unsigned int k=0;
    unsigned int temp;
    for (unsigned int j=0; j<V_to_clean->_surveys_cl_to_i.size(); ++j) {
        k=*real(V_to_clean->_I_am_in_cl_at_init[j]);
        _cl[k].clean(V_to_clean, j);/*clean the graph*/
        if(_cl[k].I_am_a_cl_true and _cl[k]._I_am_in_list_unsat){
            /*At this point the algorithm  picks a clause sat and set it on the top of the vector vec_list*/
            /*The vector vec_list_cl contains pointers to clauses sat and unsat. The value _m divides the sat clause, from 0 to _m-1, to unsat clause,
             from _m to  _M*/
            
            temp=_cl[k]._l;
            swap(vec_list_cl[_cl[k]._l], vec_list_cl[_m]);
            (vec_list_cl[_cl[k]._l])->_l=temp;
            (vec_list_cl[_m])->_l=_m;
            _cl[k]._I_am_in_list_unsat=false;
            _cl_list.erase(_cl[k]._it_list);
            _m++;
        }
    }
}

/*public member class Graph. This memeber sorts in descending order, using as predicate the certitude, vertex
 vector ptrV. The first x components of the vector are not sorted because they contain fixed variables*/
void Graph::sort_V_Dec_move(){
    sort(ptrV.begin()+(_list_fixed_element.size()), ptrV.end(), _Vertex_greater_pred());
}



/*public memeber class Graph. This memeber computes the surveys for each variable node.
 It is called when a convergence of all messages from a clause to variable is found.
 Moreover this member computes the variable complexity associated to each variable node*/
void Graph::surveys(){/*compute surveys for each variable node*/
    complexity_variables=0.;/*set complexity variable to 0*/
    unsigned int _size_init=static_cast<unsigned int>(_list_fixed_element.size());
    for (unsigned int i=_size_init; i<_N; ++i) {
        ptrV[i]->compute_s();/*compute surveys for each variable node*/
        complexity_variables+=(static_cast<double>(ptrV[i]->_degree_i)-1.)*log(ptrV[i]->complexity_variable);/*update graph variable complexity*/
    }
    complexity=complexity_clauses-complexity_variables;/*compute graph total complexity*/
    if(_numb_of_dec_moves==1 and _numb_of_back_moves==1) _comp_init=complexity;
    if(complexity_variables==0.) complexity=0;
    if((_numb_of_back_moves/_numb_of_dec_moves)<_R_BSP){
        ++_numb_of_back_moves;
        fl_bsp=true;/*update values for BSP ratio choice.*/
        //if(complexity<1.e-6 and complexity>0)fl_bsp=false; /*The algorithm is close to call walksat, and for safety reasons it makes only decimations*/
        /*this checks can be removed, because does not affect the algorithm*/
    }else{
        ++_numb_of_dec_moves;
        fl_bsp=false;
    }
}


void Graph::unit_propagation(){
    unsigned long __k;
    unsigned int C;
    /*unit propagation*/
    for(__k=_m; __k<_M; __k++){
        C=(vec_list_cl[__k])->_c;
        if(_cl[C]._I_am_in_list_unsat and _cl[C].empty()){/*check if it is empty*/
            cout<<"Contradiction found"<<endl; /*contradiction found*/
            cout<<"I quit from convergence_messages function"<<endl;
            cout<<"The number of unsat clauses is: "<<_cl_list.size()<<endl;/*Cambiato il 13/02/2025*/
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start_TIME;
            cout << "Tempo di esecuzione totale: " << elapsed.count() << " secondi" << endl;
            exit(-1);/*exit failure*/
        }
        if(_cl[C].size()==1){
            /*set the variable to 1 otherwise you will have contradiction*/
            unit_propagation(C); /*fix the variable into clause c to true and clean the graph*/
            _counter_conv=0;/*set to zero variable counter convergence*/
            complexity_clauses=0.;/*set to zero clause complexity*/
            __k=_m-1;
        }
    }
    for(__k=_m; __k<_M; __k++){
        C=(vec_list_cl[__k])->_c;
        if(_cl[C].size()==1){
            cout<<"there is an unit propagation not found"<<endl;
        }
    }
    _unit_prop=0;
    cout<<"END UNIT PROPAGATION"<<endl;
}

/*public member class Graph. This member update all messages from clauses to variables and stops if:
 a) a convergence is found : SUCCESS;
 b) a contradiction is found : exit FAILURE;
 c) no convergence is found after t_max iteration: exit FAILURE
 */
void Graph::convergence_messages(){/*compute convergence messages for message passing algorithm*/
    bool conv_f=false;
    unsigned long i,l, __k;
    unsigned int C;
START:
    /*unit propagation*/
    for(__k=_m; __k<_M; __k++){
        C=(vec_list_cl[__k])->_c;
        if(_cl[C]._I_am_in_list_unsat and _cl[C].empty()){/*check if it is empty*/
            cout<<"Contradiction found"<<endl; /*contradiction found*/
            cout<<"I quit from convergence_messages function"<<endl;
            cout<<"The number of unsat clauses is: "<<_cl_list.size()<<endl;/*Cambiato il 13/02/2025*/
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start_TIME;
            cout << "Tempo di esecuzione totale: " << elapsed.count() << " secondi" << endl;
            exit(-1);/*exit failure*/
        }
        if(_cl[C]._I_am_in_list_unsat and _cl[C].size()==1){
            /*set the variable to 1 otherwise you will have contradiction*/
            unit_propagation(C); /*fix the variable into clause c to true and clean the graph*/
            _counter_conv=0;/*set to zero variable counter convergence*/
            complexity_clauses=0.;/*set to zero clause complexity*/
            __k=_m-1;
        }
    }
    update_products();
    /*convergence*/
    for (unsigned int t=0; t<t_max; ++t) {
        conv_f=false;
        _counter_conv=0;/*set counter convergence to zero*/
        complexity_clauses=0.;/*set clauses complexity to zero*/
        i=0;
        l=0;
        for(__k=_m; __k<_M; ++__k){/*for on clause objects*/
            C=(vec_list_cl[__k])->_c;
            s=_cl[C]._size_cl_init;
            i=0;
            while (1) {
                if (_cl[C]._go_forward[i]) {
                    _new=1.;
                    norm=1.;
                    l=0;
                    while (1) {
                        if(i!=l && _cl[C]._go_forward[l]){
                            _prod_S=_Pr_S(_cl[C].v_V[l],_cl[C].v_lit[l], _cl[C].div_s[l]);
                            _prod_U=_Pr_U(_cl[C].v_V[l],_cl[C].v_lit[l]);
                            _new*=__pu();/*new message from cl to variable is computed*/
                            norm*=__norm();
                        }
                        ++l;
                        if(l==s)break;
                    }
                    _new=compute_message(_new,norm);/*set to 0 a message iff the message is smaller than 1e-16*/
                    if(_new==1.){
                        unit_propagation(C); /*fix the variable into clause c to true and clean the graph*/
                        cout<<"The Instance is not really random, I have a survey equal to 1, which gives me nan."<<endl;
                        cout<<"I try to use unit propagation and fix the variable to the best value for satisfying the clause"<<endl;
                        cout<<"This may bring us to a contradiction"<<endl;
                        cout<<"UP"<<endl;
                        goto START;
                    }
                    _cl[C].old_s[i]=_cl[C].update[i];
                    _cl[C].update[i]=((1.-_phi)*_new)+(_phi*_cl[C].update[i]);/*update new message value*/
                    if(_counter_conv==0){
                        if(_conv(_cl[C].update[i], _cl[C].old_s[i])){
                            ++_counter_conv;
                        }
                    }
                    ++i; 
                }else{
                    ++i;
                }
                if(i==s)break;
            }
            i=0;
            while (1) {
                *_cl[C].v_survey_cl_to_i[i]=_cl[C].update[i];
                _cl[C].div_s[i]=Div_s(_cl[C].update[i]);
                ++i;
                if(i==s)break;
            }
        }
        update_products();/*update products into vertex node for speeding up the algorithm*/
        if(_counter_conv==0){/*if counter convergence is zero, a convergence is found*/
            _M_t=static_cast<unsigned int>(_cl_list.size());
            // cout<<"I found a convergence at "<<t<<" "<<_m<<endl;
            _time_conv_print=t;
            update_complexity_clauses();/*update complexity_clause*/
            conv_f=true;
            break;
        }
    }
    if(!conv_f){/*if after t_max iterations no convergence is found, the algorithm return exit failure */  /*Cambiato il 13/02/2025*/
        _M_t=static_cast<unsigned int>(_cl_list.size());
        // cout<<"I found a convergence at "<<t<<" "<<_m<<endl;
        _time_conv_print=t_max;
        update_complexity_clauses();/*update complexity_clause*/
       // cout<<"SP does not find any fixed points -> SP does not converge."<<endl;
       // cout<<"I am sorry I quit :("<<endl;
       // cout<<"The number of unsat clauses is: "<<_cl_list.size()<<endl;
       // auto end = chrono::high_resolution_clock::now();
       // chrono::duration<double> elapsed = end - start_TIME;
       // cout << "Tempo di esecuzione totale: " << elapsed.count() << " secondi" << endl;
        //exit(-1);
    }
}

/*public member class Graph. This member  updates only clause complexity.*/
void Graph::update_complexity_clauses(){
    double _ps=1.,_pu=1.;/*products for clause complexity*/
    unsigned long j;
    unsigned int C;
    //cout<<"This is the value where I start: "<<_M-_m<<endl;
    for(unsigned int __k=_m; __k<_M; ++__k) {
        C=(vec_list_cl[__k])->_c;
        _ps=1.;/*initialize to one product __ps*/
        _pu=1.;/*initialize to one product __pu*/
        /*updating complexity clauses*/
        for (j=0; j<_cl[C]._size_cl_init; ++j) { /*for each literal in a clause clauses complexity is updated */
            if(_cl[C]._go_forward[j]==1){
                _prod_S=_Pr_S(_cl[C].v_V[j],_cl[C].v_lit[j], _cl[C].div_s[j]);
                _prod_U=_Pr_U(_cl[C].v_V[j],_cl[C].v_lit[j]);
                _ps*=__norm();/*update product for clause complexity*/
                _pu*=__pu();/*update product for clause complexity*/
            }
        }
        complexity_clauses+=log(_ps-_pu);/*update clause complexity*/
    }
    
}

/*public member class Graph which splits the global factor graph information in different objects and vectors.*/
void Graph::split_and_collect_information(){/*split the graph in different vectors and lists*/
    /*create clauses*/
    vec_list_cl.resize(_M);
    list<Vertex *>::iterator _it; /*iterator list of Vertex pointers*/
    double _rn=0.; /*random number*/
    unsigned int pos=0; /*position into a clause*/
    _cl[0].I_am_a_cl_true=false;/*set to false bool variable I_am_a_cl_true*/
    _cl_list.push_front(&_cl[0]);/*insert pointer of clause into the unsitisfied clauses list*/
    _cl[0]._it_list=_cl_list.begin();/*store the iterator of the unsitisfied clauses list*/
    _cl[0]._I_am_in_list_unsat=true;/*set to true the boolean varibale I_am_in_list_unsat*/
    vec_list_cl[0]=&_cl[0]; /*pointer to an element of a vector of Clause objects*/
    _m=0;
    for (unsigned int i=0, l=0; i<_ivec.size(); ++i) {
        if(_ivec[i]==0){
            pos=0; /*set position to 0*/
            ++l; /*clause label increment*/
            if(l==_M)break;
            _cl[l].I_am_a_cl_true=false;/*set to false bool variable I_am_a_cl_true*/
            _cl_list.push_front(&_cl[l]);/*insert pointer of clause into the unsitisfied clauses list*/
            _cl[l]._it_list=_cl_list.begin();/*store the iterator of the unsitisfied clauses list*/
            _cl[l]._I_am_in_list_unsat=true;/*set to true the boolean varibale I_am_in_list_unsat*/
            vec_list_cl[l]=&_cl[l];
        }else{
            _cl[l]._c=l;/*clause labeling*/
            _cl[l]._l=l;/*position in vec_listy*/
            _cl[l].survey_cl_to_i_f_ptr.push_back(NULL); /*initialization vector of frozen survey pointers*/
            _cl[l].survey_cl_to_i_f.push_back(-1.);/*initialization vector of frozen surveys*/
            _cl[l]._vb.push_back(false); /*initialization Boolean vector*/
            _cl[l]._v_ref.push_back(-1);
            _cl[l].V.push_back(ptrV[abs(_ivec[i])-1]);/*Vertex pointer stored in list V*/
            _cl[l]._it_list_V_begin=_cl[l].V.begin();/*iterator stored into the clause*/
            _cl[l]._size_cl++; /*size of the clause*/
            _cl[l]._size_cl_init++; /*size of the clause at the beginning*/
            _cl[l].cpV.push_back(ptrV[abs(_ivec[i])-1]);/*copy of V in a vertex vector*/
            _cl[l]._vecpos.push_back(pos);/*vector of literal position in the clause*/
            _it=--_cl[l].V.end();/*pick iterator of V in cl*/
            ptrV[abs(_ivec[i])-1]->where_I_am.push_back(_it);/*store iterator*/
            ptrV[abs(_ivec[i])-1]->_degree_i++;/*update degree vertex i*/
            /*initialization surveys, real t, imag t-1*/
            _rn=random()/(RAND_MAX+1.0);/*choose a random number between 0-1*/
            ptrV[abs(_ivec[i])-1]->_surveys_cl_to_i.push_back(_rn);/*survey random initialization*/
            _cl[l].old_s.push_back(_rn);
            _cl[l].update.push_back(_rn);
            _cl[l].div_s.push_back(Div_s(_rn));
            pos++;/*update position*/
            if(_ivec[i]>0){/*check if literal is negated or not*/
                _cl[l]._lit.push_back(true);/*store literal in Boolean list _lit*/
                _cl[l].cp_lit.push_back(true);/*store literal in a boolean vector*/
                _cl[l].cp_lit_int.push_back(1);/*store literal as integer*/
            }else{
                _cl[l]._lit.push_back(false);/*store literal in Boolean list _lit*/
                _cl[l].cp_lit.push_back(false);/*store literal in a boolean vector*/
                _cl[l].cp_lit_int.push_back(0);/*store literal as integer*/
                
            }
            ptrV[abs(_ivec[i])-1]->_lit_list_i.push_back(--_cl[l]._lit.end());/*store iterator in Vertex i*/
        }
    }
    /*To readers: please do not chenge this part, because we are using dynamic vectors and when one needs to store pointers of these vectors one needs to make safty choices. More precisely, each time that vector<T> constructor is called for allocating new memory, pointers can change and problems appear*/
    bool flag_s=true;
    _cl[0].v_survey_cl_to_i.resize(_cl[0].size());
    _cl[0]._go_forward.resize(_cl[0].size());
    _cl[0].v_lit.resize(_cl[0].size());
    _cl[0].v_V.resize(_cl[0].size());
    _cl[0]._var.resize(_cl[0].size());
    for (unsigned int i=0,k=0, l=0; i<_ivec.size(); ++i) {
        if(_ivec[i]==0){
            k=0;
            ++l; /*increment of one clause label*/
            flag_s=true;
            if(l==_M)break;
            _cl[l]._go_forward.resize(_cl[l].size());
            _cl[l].v_survey_cl_to_i.resize(_cl[l].size());
            _cl[l].v_lit.resize(_cl[l].size());
            _cl[l].v_V.resize(_cl[l].size());
            _cl[l]._var.resize(_cl[l].size());
        }else{
            
            _cl[l].v_V[k]=ptrV[abs(_ivec[i])-1];/*Vertex pointer stored in list V*/
            _cl[l]._go_forward[k]=1;
            ptrV[abs(_ivec[i])-1]->_I_am_in_cl_at_init.push_back(complex<unsigned int *>(_cl[l]._ptr_c(),_cl[l]._ptr_pos(k)));/*store initial position of Vertex i into clause l*/
            ptrV[abs(_ivec[i])-1]->_bvec_lit.push_back(_cl[l]._ptr_lit_int(k));/*pointers vector literal variablee node*/
            _cl[l].survey_cl_to_i.push_back(ptrV[abs(_ivec[i])-1]->ptr_survey());/*store pointer of survey in cl l*/
            _cl[l].v_survey_cl_to_i[k]=ptrV[abs(_ivec[i])-1]->ptr_survey();
            _cl[l]._var[k]=_ivec[i];
            if(flag_s)_cl[l].survey_begin=_cl[l].survey_cl_to_i.begin();
            flag_s=false;
            if(_ivec[i]>0){
                _cl[l].v_lit[k]=true;
                ptrV[abs(_ivec[i])-1]->_surveys_cl_to_i_plus.push_back(ptrV[abs(_ivec[i])-1]->ptr_survey());
                
            }else{
                _cl[l].v_lit[k]=false;
                ptrV[abs(_ivec[i])-1]->_surveys_cl_to_i_minus.push_back(ptrV[abs(_ivec[i])-1]->ptr_survey());
            }
            _cl[l].survey_cl_to_i_f_ptr[k++]=ptrV[abs(_ivec[i])-1]->ptr_survey();
            ptrV[abs(_ivec[i])-1]->_where_surveys_are_in_cl.push_back(--_cl[l].survey_cl_to_i.end());
            ptrV[abs(_ivec[i])-1]->_i++;
        }
    }
    
    update_products();/*update products*/
}

void Graph::update_products(){/*update products*/
    unsigned int _size_init=static_cast<unsigned int>(_list_fixed_element.size());
    for (unsigned int i=_size_init; i<_N; ++i) {
        //        if(!ptrV[i]->_I_am_a_fixed_variable){
        ptrV[i]->make_products();/*update products*/
        ptrV[i]->_i=0/*set variable _i in Vertex i to 0*/;
        //        }
    }
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/************************************ END SP ***************************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/



/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***************************** START BACKTRACKING **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

/*public member class Graph. This memeber sorts the element of the list in ascending order, using as predicate the certitude, The first x components of the vector are not sorted because they contain fixed variables*/
void Graph::sort_V_Back_move(){
    sort(ptrV.begin(), ptrV.begin()+(_list_fixed_element.size()), _Vertex_greater_pred());
}


/*public member calss Graph. This member builds up the clauses that are not anymore satisfied by the assignment of the variables*/
void Graph::build(Vertex * &V_to_build){
    unsigned int k=0;
    unsigned int temp;
    for (unsigned int j=0; j<V_to_build->_surveys_cl_to_i.size(); ++j) {
        k=*real(V_to_build->_I_am_in_cl_at_init[j]);/*pick the clause that should be unfixed*/
        _cl[k].build(V_to_build, j);/*build the clause in the graph*/
        if(!_cl[k].I_am_a_cl_true and !_cl[k]._I_am_in_list_unsat){/*check if the clasue is false and if it is in the list of un-sat clauses*/
            _cl[k]._I_am_in_list_unsat=true;
            _cl_list.push_front(&_cl[k]);/*set the clause into the list of un-sat clasues*/
            _cl[k]._it_list=_cl_list.begin();/*store the iterator of the list*/
            _M_t++;/*update index of un-sat clauses*/
            _m--; /*update index for fast calcualtion*/
            temp=_cl[k]._l;
            swap(vec_list_cl[_cl[k]._l], vec_list_cl[_m]);
            (vec_list_cl[_cl[k]._l])->_l=temp;
            (vec_list_cl[_m])->_l=_m;
        }
    }
}



/*public member class Graph. This member computes all operation for backtracking moves.*/
void Graph::backtrack(){
    //cout<<"I make a backtrack move"<<endl;
    sort_V_Back_move(); /*sort the elements in ptV vector, the ones in position 0, _m-1; in ascending order*/
    
    unsigned long _size=_m_t_m_1+_unit_prop;
    _unit_prop=0;
    _m_t_m_1=0;
    unsigned long _size_init=_list_fixed_element.size();
    for (unsigned long i=_size_init; i>_size_init-_size;) {
        if((ptrV[--i])->_sC != 1.){ // se sto usando NN questo if deve essere >0 altrimenti !=1
            /*build the graph*/
            /*
             build the graph means:
             
             1) introducing into the factor graph all unsatisfied clauses, given by the fact that variable i is not anymore fixed;
             2) introducing literals associated to variable i, which are present in clauses not satisfied.
             */
            
            _list_fixed_element.erase(ptrV[i]->_it_list_fixed_elem);/*erase the variable node in fixed element list*/
            ptrV[i]->_it_list_fixed_elem = _list_fixed_element.end(); /*set the iterator to _list_fixed_element.end() by default*/
            ptrV[i]->reset_value_default_var_i();/*reset values into the node*/
            build(ptrV[i]);/*build clauses*/
            
        }else{
            break;
        }
        
    }
    _N_t=_N-static_cast<unsigned int>(_list_fixed_element.size());/*update the number of un-fixed variable nodes*/
    //update_products();/*update products into vertex node for speeding up the algorithm*/
    
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/****************************** END BACKTRACKING  **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

/*public member class Graph. This member prints on file the residual CNF formula.*/
void Graph::print_on_file_residual_formula(){
    ostringstream seed;
    seed<<_seed;/*real seed of the graph in the residual formula title*/
    const string s=seed.str();
    const string title="residualformula_seed";
    const string txt=".cnf";
    string rf=directory;
    rf+=title;
    rf+=s;
    rf+=txt;
    ofstream outfile(rf.c_str());
    if (!outfile) {
        cerr<<"Error file output does not exist"<<endl;
        exit(-1);
    }else{

        outfile << "c seed=1234567"<<endl;
        outfile << "p cnf";
        outfile << ' ' << _N << ' ' << _cl_list.size() << endl;
        for (list<Clause *>::iterator i=_cl_list.begin(); i!=_cl_list.end(); ++i) {
            outfile<<_cl[(*i)->_c];/*print on file cluases*/
        }
        
        
    }
    
}

/*public member class graph. This member calls WalkSAT function and computes the solution for the residual formula.
 Moreover, it builds the complete solution for the problem and also checks if all solutions are composed by frozen variables or not.*/
bool Graph::WalkSAT(){
    sort(ptrV.begin(), ptrV.end(), _Vertex_smaller_labeled_vertex_pred());/*simple ascending sort for labeled vertex.*/
    vector <vector<bool> > sol;
    ostringstream seed;
    seed<<_seed;
    const string s=seed.str();
    const string title="residualformula_seed";
    const string txt=".cnf";
    string rf=directory;
    myrand=1234567;/*seed for walksat*/
    rf+=title;
    rf+=s;
    rf+=txt;
    int argc=10;
    char * argv[10];
    for (int i=0;i<argc;i++) argv[i]=(char *)malloc(1000);
    strcpy (argv[0],"test" );
    strcpy (argv[1],"-solcnf" );
    strcpy (argv[2],"-cutoff");
    strcpy (argv[3],"10000000" );
    strcpy (argv[4],"-best" );
    strcpy (argv[5],"-seed" );
    strcpy (argv[6],"1234567");
    strcpy (argv[7],"-numsol" );
    strcpy (argv[8],"10" );
    strcpy (argv[9],rf.c_str());
    bool flag=false;
    for (int i=1;i<argc;i++)fprintf(stderr, "Input %s\n", argv[i]);
    if(_cl_list.empty()){
       sol.resize(1);
       for (unsigned int i=0; i<_N; ++i) {
           sol[0].push_back(ptrV[i]->_who_I_am);
       }
       unsigned int couneter_unsat=0;
       
       for(unsigned int c=0; c<_M;++c){    /*problem check*/
        if (!_cl[c]._check()) {
            couneter_unsat++;
        }
    }
    cout<<"The number of unsat clauses is: "<<couneter_unsat<<endl;
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start_TIME;
    cout << "Tempo di esecuzione totale: " << elapsed.count() << " secondi" << endl;
    exit(-1);
    }else{
        WalkSat(sol,argc,argv);/*call WalkSAT*/
    }
    vector<long int> mysol;
    cout<<"Check and white a solution"<<endl;
    cout<<"time of whitening    freezing probability"<<endl;
    for (unsigned int i=0;i<sol.size();i++){
        if(!mysol.empty())mysol.clear();
        mysol.resize(_N+1);
        /*initialization procedure for whitening procedure*/
        for (unsigned int l=0; l<_M; ++l) {
            _cl[l].I_am_white=false;
        }
        for (unsigned int l=0; l<_N; ++l) {
            ptrV[l]->_I_am_white=false;
        }
        
        /*save SP solutions in vector mysol*/
        for (list<Vertex*>::iterator __it=_list_fixed_element.begin(); __it!=_list_fixed_element.end(); ++__it) {
            unsigned int var=(*__it)->_vertex;
            if((*__it)->_who_I_am)mysol[(*__it)->_vertex]=static_cast<long int>(var);
            else mysol[(*__it)->_vertex]=-(static_cast<long int>(var));
        }
        
        /*build a complete solution for the problem*/
        for (long int j=0; j<sol[i].size(); ++j) {
            if (!sol[i][j] and mysol[j+1]==0) {
                mysol[j+1]=-(j+1);
                ptrV[j]->_who_I_am=false;
                ptrV[j]->_I_am_a_fixed_variable=true;
            }else if(sol[i][j] and mysol[j+1]==0){
                mysol[j+1]=(j+1);
                ptrV[j]->_who_I_am=true;
                ptrV[j]->_I_am_a_fixed_variable=true;
            }
        }
        
        /*check if the global solution is correct*/
        for(unsigned int c=0; c<_M;++c){    /*problem check*/
            if (!_cl[c]._check()) {
                cout<<_cl[c];
                cout<<"NO SOLUTION ERROR"<<endl;/*NO SOLUTION FOUND*/
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = end - start_TIME;
                cout << "Tempo di esecuzione totale: " << elapsed.count() << " secondi" << endl;
                exit(-1);
            }
        }
        if(i==0)print_only_one_sol();
        if(!flag)flag=true;
        ostringstream alpha_str; //printing solution on file
        ostringstream var_str;
        ostringstream    i_str;
        ostringstream seed_str;
        string mystring;
        const string title="Solution_tot";
        const string seed="seed=";
        const string under="_";
        const string dat=".txt";
        
        alpha_str<<_alpha;
        var_str<<_N;
        i_str<<i;
        seed_str<<_seed;
        mystring=directory;
        mystring+=title;
        mystring+=alpha_str.str();
        mystring+=under;
        mystring+=var_str.str();
        mystring+=under;
        mystring+=i_str.str();
        mystring+=under;
        mystring+=seed;
        mystring+=seed_str.str();
        mystring+=dat;
        ofstream outfile5(mystring.c_str());
        for (int i=1; i<mysol.size(); ++i) {
            outfile5<<mysol[i]<<endl;
        }
        Whitening_Solution();/*whitening procedure on a complete solution*/
    }
    return flag;
}

/*Frozen and unfrozen variables and whitening procedure*/

/*The complexity of finding solutions close to SAT-UNSAT threshold could be correlated
 to the presence of frozen cluster. Frozen clusters are those clusters which contain an exponential
 number of frozen solutions. Frozen solutions are those solutions that contain frozen variables,
 and frozen variables are those variables that have the same values in all the solutions which in a cluster.
 Unforzen variables are those variables that can have different  values in the solutions
 that are in a forzen cluster. In other words, it is possible going from a frozen solution to another,
 both belonging to the same cluster, with a local rearangment of unfrozen variables.
 However, going from frozen solution to another solution belonging in different clusters
 needs a global rearrangment of all variables.
 For checking if a variable is frozen or not, and therefore if a solution is forzen or not,
 physicists have defined the whitening procedure, and for a SAT problem is described in the following way:
 
 start with a solution and assign iteratively a ”∗” (joker or white color) to variables which belong only
 to clauses which are already satisfied by another variable or already contain a "∗" variable.
 
 If a finite fraction of variable is not assigned to a "*" value, then all those variables are frozen variables
 and the solution  is called frozen. However, in our experience we have not met frozen solutions for K-SAT problems
 for N large enough. We have met them only for smaller values of N.
 
 */

/*public member class Graph.*/
void Graph::Whitening_Solution(){/*whitening procedure*/
    string mystring;/*string for printing file*/
    const string title="whitening";
    const string dat=".txt";
    ostringstream alpha_str;
    ostringstream var_str;
    
    alpha_str<<_alpha;
    var_str<<_N;
    mystring=directory;
    mystring+=title;
    mystring+=var_str.str();
    mystring+=alpha_str.str();
    mystring+=dat;
    ofstream outfilew(mystring.c_str(), ios_base::app );
    bool flag_white=true;/*Boolean flag for white*/
    double counter_white_var=0.;/*counter for white variables*/
    unsigned int time_white=0;/*time for whitening a solution*/
    for (unsigned int t=0; t<1024; ++t) {/*time loop*/
        time_white=t;
        for (unsigned int i=0; i<_N; ++i) {/*loop on all variable nodes*/
            if(!ptrV[i]->_I_am_white){
                flag_white=true;
                ptrV[i]->_who_I_am=!ptrV[i]->_who_I_am;/*flip the variable and check if the new configuration is still a solution*/
                for (unsigned j=0; j<ptrV[i]->_I_am_in_cl_at_init.size(); ++j) {
                    if(!_cl[*real(ptrV[i]->_I_am_in_cl_at_init[j])].I_am_white){/*check in all the clause where the variable appears*/
                        flag_white=flag_white and _cl[*real(ptrV[i]->_I_am_in_cl_at_init[j])]._check();
                        if(!flag_white)break;
                    }
                }
                if(flag_white){
                    for (unsigned j=0; j<ptrV[i]->_I_am_in_cl_at_init.size(); ++j) {
                        if(!_cl[*real(ptrV[i]->_I_am_in_cl_at_init[j])].I_am_white){/*if the variable is a "*" joker variable*/
                            _cl[*real(ptrV[i]->_I_am_in_cl_at_init[j])].I_am_white=true;/*clauses where joker variables appear become "white"*/
                        }
                    }
                    ptrV[i]->_I_am_white=true;
                    counter_white_var++;
                }
                ptrV[i]->_who_I_am=!ptrV[i]->_who_I_am;/*flip the variable to the correct value*/
                if(counter_white_var==static_cast<double>(_N)){
                    t=1025;
                    break;
                }
            }
        }
    outfilew<<_seed<<" "<<_N<<" "<<_alpha<<" "<<time_white<<" "<<counter_white_var<<endl;/*print on file*/
    }
    cout<<time_white<<"\t \t \t \t"<<counter_white_var/(double)(_N)<<endl;/*print on terminal*/
}

/*public member class Graph. This member prints the variable surveys at the first iteration*/
void Graph::print_only_one_sol(){
    /*output file*/
    const string title="Sol_SP_";
    const string txt=".txt";
    string str;
    ostringstream _seeds, _Ks, _sN, _salpha;
    _seeds<<_seed;
    _Ks<<_K;
    _salpha<<_alpha;
    _sN<<_N;
    str=directory;
    str=title;
    str+="_seed_";
    str+=_seeds.str();
    str+="K=";
    str+=_Ks.str();
    str+="N=";
    str+=_sN.str();
    str+="alpha=";
    str+=_salpha.str();
    str+=txt;
    ofstream outfilesol_(str.c_str(), ios_base::app);
    vector<long int> mysol;
    mysol.resize(_N+1);
    for (list<Vertex*>::iterator __it=_list_fixed_element.begin(); __it!=_list_fixed_element.end(); ++__it) {
        unsigned int var=(*__it)->_vertex;
        if((*__it)->_who_I_am)mysol[(*__it)->_vertex]=static_cast<long int>(var);
        else mysol[(*__it)->_vertex]=-(static_cast<long int>(var));
    }
    for (unsigned i=1; i<=_N; ++i) {
        if(mysol[i]!=0)outfilesol_<<mysol[i]<<" ";
    }
    outfilesol_<<_seed<<" "<<_comp_init/(double)_N<<" "<<endl;
    
    
}



/*public member class Graph. This member prints the variable surveys at the first iteration*/
void Graph::print_surveys(){
    /*output file*/
    const string title="Surveys";
    const string txt=".dat";
    string str;
    ostringstream _seeds, _Ks, _sN, _salpha;
    _seeds<<_seed;
    _Ks<<_K;
    _salpha<<_alpha;
    _sN<<_N;
    str=directory;
    str+=title;
    str+="_seed_";
    str+=_seeds.str();
    str+="K=";
    str+=_Ks.str();
    str+="N=";
    str+=_sN.str();
    str+="alpha=";
    str+=_salpha.str();
    //str+=sat;
    //str+=_seeds.str();
    //str+=_Ks.str();
    //str+=sat;
    //str+=_seeds.str();
    str+=txt;
    ofstream outfilewff(str.c_str(), ios_base::app);
    vector<long int> mysol;
    mysol.resize(_N+1);
    for (list<Vertex*>::iterator __it=_list_fixed_element.begin(); __it!=_list_fixed_element.end(); ++__it) {
        unsigned int var=(*__it)->_vertex;
        if((*__it)->_who_I_am)mysol[(*__it)->_vertex]=static_cast<long int>(var);
        else mysol[(*__it)->_vertex]=-(static_cast<long int>(var));
    }
    for (unsigned i=0; i<_N; ++i) {
        if(mysol[i+1]!=0)outfilewff<<_v_sT[i]<<" "<<_v_sI[i]<<" "<<_v_sF[i]<<" ";
    }
    outfilewff<<_seed<<" "<<_comp_init/(double)_N<<endl;
    
    
}
/*public member class Graph. This member prints the values on file*/
void Graph::print(){
    const string title="Value_Anal_Compl";
    const string txt=".txt";
    string str;
    ostringstream _seeds, _Ks, _sN, _salpha;
    _seeds<<_seed;
    _Ks<<_K;
    _salpha<<_alpha;
    _sN<<_N;
    str=directory;
    str=title;
    str+="_seed_";
    str+=_seeds.str();
    str+="K=";
    str+=_Ks.str();
    str+="N=";
    str+=_sN.str();
    str+="alpha=";
    str+=_salpha.str();
    //str+=sat;
    //str+=_seeds.str();
    //str+=_Ks.str();
    //str+=sat;
    //str+=_seeds.str();
    str+=txt;
    ofstream outfilecomp_(str.c_str(), ios_base::app);
    for (unsigned i=0; i<_v_Nt.size(); ++i) {
        outfilecomp_<<_v_Nt[i]<<" "<<_v_M_t[i]<<" "<<_v_c[i]<<" "<<_v_time[i]<<" "<<_N<<" "<<_R_BSP<<endl;
    }
    
}

/*public member class Graph. In this member an instance of the problem is built and is printed on file*/
void Graph::write_on_file_graph(){/*build a graph and write a CNF formula*/
    /* get the values from INPUT and store them in the right place*/
    _N=static_cast<unsigned int>(stoul(_argv[_argc-1],nullptr,0));
    _alpha=stod(_argv[_argc-2],nullptr);
    double m=((double)_N*_alpha);
    _M=static_cast<unsigned int>(m);
    while(static_cast<double>(_M)<m){
        if(((double)_M)==m)break;
        _M++;
    }
    _alpha=(double)_M/(double)_N;
    _K=static_cast<unsigned int>(stoul(_argv[_argc-3],nullptr,0));
    cout<<setprecision(9);
    cout<<"I am going to build an instance with N: "<<_N<<" variables and "<<_M<<" clauses with a clause density equal to "<<_alpha<<endl;
    _ivec.reserve(_K*_M);
    _N_t=_N;
    _M_t=0;
    /*string for output file*/
    const string title="Formula_CNF";
    const string sat="-SAT_seed=";
    const string txt=".cnf";
    string str;
    ostringstream _seeds, _Ks, _sN, _salpha;
    _seeds<<_seed;
    _Ks<<_K;
    _salpha<<_alpha;
    _sN<<_N;
    str=directory;
    str+=title;
    str+="K=";
    str+=_Ks.str();
    str+="N=";
    str+=_sN.str();
    str+="alpha=";
    str+=_salpha.str();
    str+=sat;
    str+=_seeds.str();
    str+=txt;
    
    /*I am modifing how to write the output file*/
#ifdef PRINT_FORMULA_CNF
    /*output file*/
    ofstream outfilewff(str.c_str(), ios_base::app);
    outfilewff << "c seed="<<_seed<<endl;
    outfilewff << "p cnf";
    outfilewff << ' ' << _N << ' ' << _M << endl;
#endif
    /******************************************************/
    /******************************************************/
    /********************** START *************************/
    /******** Henry Kautz & Bart Selman makewff.c *********/
    /******************************************************/
    /******************************************************/
    /******************************************************/
    
    /*minimal modifiications for compatibility in c++*/
    
    int i, j, k;
    int lit;
    int cl[MAX_CLEN];
    bool dup;
    
    /*build the CNF instance*/
    for (i=0; i<_M; i++){
        
        for (j=0; j<_K; j++){
            
            do {
                
                
                lit =  static_cast<int>(random() % _N) + 1;
                
                
                dup = false;
                
                for (k=0; k<j; k++)
                    
                    if (lit == cl[k]) dup = true;
                
            } while(dup);
            
            cl[j] = lit;
            
        }
        /* flip the literal*/
        for (j=0; j<_K; j++){
            
            if (_flip()) cl[j] *= -1;
            
            _ivec.push_back(cl[j]);
            
        }
        _ivec.push_back(0);
        /******************************************************/
        /******************************************************/
        /******************************************************/
        /******** Henry Kautz & Bart Selman makewff.c *********/
        /*********************** END **************************/
        /******************************************************/
        /******************************************************/
        
        /* print on file*/
#ifdef PRINT_FORMULA_CNF
        for (j=0; j<_K; j++)outfilewff<<cl[j]<<" ";
        
        outfilewff<<"0"<<" "<<endl;
#endif
    }
    
}

/*public member class Graph. This member reads an instance of a problem from a file*/
void Graph::read_from_file_graph(){/*read an instance given as INPUT*/
    /*open file to read*/
    ifstream infile(_argv[_argc-1].c_str());
    if (!infile) {
        cout<<_argv[_argc-1]<<endl;
        cout <<"File not found"<<endl;
        cout<<"Please check if the name is correct or if the file exists."<<endl;
        exit(-1);
    }else{
        string ogg1;
        int ogg2, ogg3;
        bool flag=false;
        int var1;
        /*read file and store variables in _ivec*/
        /* A CNF formula is composed by by +- numbers. 0 identifies the end of a clause*/
        while (!infile.eof()) {
            if(infile.eof())break;
            if (!flag) {
                infile>>ogg1;
                if (ogg1=="c"){
                    infile>>ogg1;
                    //string str;
                    //str=ogg1.substr(5);
                    //_seed_out=atoi(str.c_str());
                }
                if (ogg1== "cnf") {
                    infile>>ogg2>>ogg3;
                    _N=ogg2;
                    _M=ogg3;
                    _N_t=_N;
                    _M_t=_M;
                    _ivec.reserve(_M*10);
                    flag=true;
                }
            }
            if (flag) {
                infile>>var1;
                if(infile.eof())break;
                _ivec.push_back(var1);
            }
        }
        vector<long int>(_ivec).swap(_ivec);
    }
}

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/******************************  END GRAPH CLASS  **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


