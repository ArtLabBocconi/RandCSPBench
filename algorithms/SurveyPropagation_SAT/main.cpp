//
//  main.cpp
//  TheBackTrackingSurveyPropagation
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
//  Created by Raffaele Marino on 2018-08-25.
//  Copyright © 2018 Raffaele Marino. All rights reserved.
//


                                /*******************************************/
                                /*******************************************/
                                /*IL PERDER TEMPO A CHI PIU' SA PIU' SPIACE*/
                                /*******************************************/
                                /*******************************************/


#define MAIN
#define VERSION "BSP_SAT_VERSION_2018"
#include "Header.h"
#include "Vertex.hpp"
#include "Graph.hpp"
#define UNIX 1
#if UNIX
#define random() rand()
#define srandom(seed) srand(seed)
#endif    

/*This code describes message passing algorithms for solving random SAT problems.
 
 
 
 ******************************************************************************************************************
 At this moment the code is able to solve:
 
    1) Random K-SAT problems with Survey Propagation, with backtracking strategy if needed, for any K.
 
 ******************************************************************************************************************
 
 
 This code is an open source code, it can be modified and improved with new strategies.
 For any question about this code (and if you will find bugs), please, contact me at:
 raffaele.marino@epfl.ch or marino.raffaele@mail.huji.ac.il or marinoraffaele.nunziatella@gmail.com */




void help();/*help function*/

int main(int argc,  char * const argv[]) {
    chrono::time_point<std::chrono::high_resolution_clock> start_TIME = chrono::high_resolution_clock::now();
    Graph G(argc,argv);/*declaration object Graph*/
    vector<Vertex> V;/*declaration vertex vector*/
    
    cout<<"START ALGORITHM:"<<endl;
    
    /***********************************************************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    /***************************** START WRITE OR LOAD *********************************/
    /***********************************************************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    
    /* At this point the code choses between loading a SAT CNF instance or building one. */
    int c;
    while ((c = getopt (argc, argv, "l:w:h?")) != -1)/* write or load an instance K-SAT*/
        switch (c) {
            case 'l':/* load an instance K-SAT*/
                switch (argc) {
                    case 3:
                        G.read_from_file_graph(); /*read CNF instance from file*/
                        break;
                    default:
                        help(); /*describe how to give INPUT from terminal*/
                        exit(-1);
                        break;
                }
                break;
            case 'w':/* write an instance for SAT problem*/
                G.write_on_file_graph(); /*build and write a CNF instance on file*/
                exit(-1);
                break;
                
            case 'h':/* help function is called*/
                
            default:
                
                fprintf(stderr, "%s [options]\n"
                        "\n"
                        "-l= reading input file"
                        "\n"
                        "-w= writing input file"
                        "\n"
                        "+++++++++++++++++++++++"
                        "\n"
                        ,argv[0]);
                help();/*help function*/
                exit(-1);
                break;
        }
    /***********************************************************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    /****************************** END WRITE OR LOAD **********************************/
    /***********************************************************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    
    /*At this point the code initializes the vertex Vector. */
    V.resize(G.N());/*Initialization Vector V*/
    for (unsigned int i=0; i<G.N(); ++i) {
        V[i]._vertex=i+1; /*label each variable node with a number form 1 to N*/
        V[i]._vertex_lli=(long int)(i+1);
    }
    
    /*In graph G the vector V is stored in a vector of pointers*/
    G.get_V(V);/*pass vector V to object G*/
    
    /*An instance of SAT, K-SAT, or NAESAT, is given as a conjunction of one or more clauses,
     where a clause is a disjunction of literals. For the XORSAT problem, instead, a clause takes the form of
     a "XOR" function of literals. Each instance can be represented by a bipartite graph, called factor graph.
     Each clause corresponds to a function node, each variable to a variable node,
     and an edge connects a function node and a variable node if and only if the clause contains the variable.
     All these information are collected in split_and_collect_information(). This public member of class Graph
     helps to split a CNF instance into Vertex objects and Clause objects.
     These objects store all types of information about the factor graph.
     The reader can see the specific for each class member in Graph.hpp, Vertex.hpp and Graph.cpp,
     Vertex.cpp files.*/
    
    G.split_and_collect_information();/*split and collect information into graph G*/
    if(_R_BSP!=0.)cout<<"START BSP WITH r="<<_R_BSP<<":"<<endl;
    else cout<<"START SID:"<<endl;
    /*Check if we have to use unit propagation*/
    G.unit_propagation();
    
    goto SP;
    /***********************************************************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    /********************************* START BSP ***************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    
    
    /*The heart of message passing procedures are iterative equations.
     Survey propagation (SP) algorithm is based on equations  derived by cavity method.
     
     For cavity method and analytic formulation of SP equations we refer to :
     
     [1] Marino, Raffaele, Giorgio Parisi, and Federico Ricci-Tersenghi. "The backtracking survey propagation algorithm for solving random K-SAT problems." Nature communications 7 (2016): 12996.
     [2] Mézard, Marc, Giorgio Parisi, and Riccardo Zecchina. "Analytic and algorithmic solution of random satisfiability problems." Science 297.5582 (2002): 812-815.
     [3] Braunstein, Alfredo, and Riccardo Zecchina. "Survey propagation as local equilibrium equations." Journal of Statistical Mechanics: Theory and Experiment 2004.06 (2004): P06007.
     [4] Mézard, Marc, and Giorgio Parisi. "The cavity method at zero temperature." Journal of Statistical Physics 111.1-2 (2003): 1-34.
     [5] Mézard, Marc, and Riccardo Zecchina. "Random k-satisfiability problem: From an analytic solution to an efficient algorithm." Physical Review E 66.5 (2002): 056126.
     [6] Parisi, Giorgio. "On the survey-propagation equations for the random K-satisfiability problem." arXiv preprint cs/0212009 (2002).
     [7] Parisi, Giorgio. "A backtracking survey propagation algorithm for K-satisfiability." arXiv preprint cond-mat/0308510 (2003).
     [8] Aurell, Erik, Uri Gordon, and Scott Kirkpatrick. "Comparing beliefs, surveys, and random walks." Advances in Neural Information Processing Systems (2005).
     
     
     SP equations return messages that go from each caluse to each variable node,
     and compute for each variable node surveys. A message sent from a clause c to a variable i
     informs the variable what is the best choice to do for satisfying clause c. This message is computed
     from the messages recived by remaining variables j into clause c, but distinct from i.
     All messages are computed iteratively till a fixed point shows up.
     In this code this procedure is obtained by the public member of class Graph convergence_messages().
     For details, we refer to technical specifications into class Graph.hpp and .cpp .
     Once a convergece is found, surveys for each variable node are computed. They are three,
     and each of them describes the probability that a variable node can be assigned to
     a true value, s_T, to a false value, s_F, or can be indeterminate s_I.
     This calculation, in this code, is performed by the public member surveys().
     For details, we refer to technical specifications in class Graph and class Vertex.
     Moreover, we also compute the certitude, defined as:
     
                                                s_C=(1.-min(s_T,s_F).
     
     This simple bias will help us to decide which variable(s) can be fixed or relased
     during decimation or backtracking procedure.*/
SP:
    
    G.convergence_messages();/*find messages convergence*/
    G.surveys();/*compute surveys for variable nodes*/
    
    /*The backtracking survey propagation (BSP) algorithm proceeds similarly to survey inspired decimation (SID), by alternating decimation or backtracking steps on a fraction f of variables, in order to keep the algorithm efficient. The choice between a decimation or a backtracking step is taken accordingly to a stochastic rule, where the parameter r ∈ [0,1) represents the ratio between backtracking steps to decimation steps [1]. When r=0 one obtains survey inspired decimation (SID), while when r!=0 one works with backtracking survey propagation. In this code r=_R_BSP into Header.h file.
     */
    if (!G.fl_bsp) {    /*choice between backtracking strategy or decimation  strategy*/
        
                                            /*decimation strategy*/
        
       /* The main step in decimation procedure consists in starting from a problem with N variables and assign
        a variable node to reduce the order of the factor graph and its size, with the hope to make it simpler.
        This strategy proceeds in the following way. One chooses a variable node i with the largest value of
        certitude s_C, assigns this variable to true or false using the rule:
        
                                            i is TRUE if s_T>s_F;
                                            i if FALSE if s_F>s_T,
        
        and removes it from the factor graph. Then one removes, in according to the problem
        (i.e. if is a SAT, XORSAT or NAESAT), all the clauses that are satisfied by the assignment done, and in all
        cluases not sitisfied by the assignment of i, the literal associated is removed.
        In a faster version one can decimate L variable nodes with maximal s_C, where L = fN with f a small number,
        instead of decimate only one variable node each time. If one is not too near the critical point and f < .01
        the results are only weakly dependent on f [3,7].
        This procedure is described by two public members in class Graph, i.e. sort_V_Dec_move() and
        choose_var_to_fix_and_clean(). sort_V_Dec_move() helps us to pick up variable nodes with the highest value
        of certitude. It is a sort and the computational complexity is O(NlogN). choose_var_to_fix_and_clean()
        fixes variable nodes and cleans the graph, as described above. It has a computational complexity O(logN).
        Onece decimation step is terminated a new convergence of clause to variable messages is needed.
        This strategy proceeds iteratively till one of these conditions appears:
        
                1) all messages computed by SP equations are trivial, i.e. all of them are 0. In this case one is arrived to an "easy" region, called by physicist paramagnetic phase, where the residual graph can be solved using greedy or heuristic (polynomial)algorithms.
                2) SP equations does not find any fixed point. In this case we claim that the algorithm fails.
        */
        cout<<G;/*print on terminal complexity*/
        G.save();
        if(G.complexity==0)goto PARAPHASE; /*go to PARAPHASE for building final solution*/
        if(G.complexity<-2. && 1!=1){/*Cambiato il 13/02/2025*/
            /*By experience we know that when N is large, e.g. (> 1000), and SP
             gives a negative complexity, with probability equal to one SP will not converge. For this
             reason we have set an exit strategy here.*/
            /*This point should be analyzed carefully from the scientific community*/
            cout<<"Negative complexity: I am not able to find a solution ;("<<endl;
            cout<<"I am sorry I quit"<<endl;
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start_TIME;
            cout << "Tempo di esecuzione totale: " << elapsed.count() << " secondi" << std::endl;
            exit(-1);
        }
        G.sort_V_Dec_move();/*sort for picking up variable nodes with the highest value of s_C*/
        G.choose_var_to_fix_and_clean();/*fix varaible nodes and clean the graph*/
        goto SP;/*go to SP*/
    
    }else{
      
                                            /*backtracking strategy*/
        
        /*The backtracking survey propagation (BSP) algorithms introduces a backtracking step,
         where a variable already assigned can be released and eventually re-assigned in a future decimation step.
         It is not difficult to understand when it is worth releasing a variable.
         The bias of a not assigned variable node i, which we recall to be defined as certitude:
         
                                                s_C=(1.-min(s_T,s_F),
         
         can be compared to the certitude of another variable node j already assigned:
         if the bias s_C associated to variable node j is smaller than the bias s_C associated to i
         then it is useful releasing variable j. Indeed, releasing variable j may help to correct
         mistakes made during previous decimation steps.
         In other words, decimation steps and backtracking steps choose variables,
         respectively, to  be fixed and to be relased, according to their biases s_C:
         variables to be fixed have the largest biases and variables to be released have the smallest biases.*/
        
        
        
        G.backtrack();/*bactrack move*/
        goto SP;/*go to SP*/
    }
    
    
PARAPHASE:
    
    /*When SP equations find trivial solutions, i.e. all messages from clauses to variables are null,
     the algorithm enter in a paramagnetic phase. The peculiarity of this phase is that the residual CNF formula
     obtaained by BSP is relatively simple for deterministic algorithms.
     We choose Walksat for the reason that is fast.
     In WalkSAT member in class Graph we also analyse if a solution is composed by frozen variable or not.
     This procedure will be described in detail in class Graph.*/
    
#ifdef WALKSAT
    G.print_on_file_residual_formula();/*print on file residual formula un-satisfied*/
    if(G.WalkSAT()){
        G.print();
    };/*run walksat on residual formula*/
#endif

    /***********************************************************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    /*********************************** END BSP ***************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    /***********************************************************************************/
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start_TIME;
    cout << "Tempo di esecuzione totale: " << elapsed.count() << " secondi" << endl;
    return 0;
}


void help(){    /*help function*/
    
    cout << "************** Help Function **************"<<endl;
    
    cout << "\n"<<endl;
    
    cout << "************************************"<<endl;
    
    cout << "\n"<<endl;
    
    cout << "If you use -w: [K] [alpha] [N variables]"<<endl;
    
    cout << "\n"<<endl;
    
    cout << "If you use -l: [formula_INPUT_File_Name.cnf]"<<endl;
    
}
