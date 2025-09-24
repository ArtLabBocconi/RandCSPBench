//
//  Vertex.hpp
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

#ifndef Vertex_hpp
#define Vertex_hpp
#include "Header.h"

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***************************** START VERTEX CLASS **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


/*a=_sT, b=_sF, c=_sI*/

/*Here anyone can create its own bias __H(*)*/

#ifdef POL
#define __H(a,b,c) = abs(a-b) /*polarization*/
#endif

#ifdef CERT
#define __H(a,b,c)  1.-min(a,b) /*certitude 1.-min(s_T,s_F)*/
#endif

#ifdef P_M
#define __H(a,b,c)  max(1.-b,1.-a) /*new certitude max(1.-s_F,1.-s_T)*/
#endif

#ifdef I_C
#define __H(a,b,c)  c /*only s_I*/
#endif

/*This class describes variable nodes in a factor graph. It collects all information that a variable node shall have, e.g. label vertex, degree of the node etc. Moreover,  this class also compute all surveys that a variable node sends to a factor node, using message passing algorithms*/

/*class vertex*/

class Vertex{
public:
    
    /*public members*/
    
    Vertex(void){ /*default constructor*/
        _vertex=0;/*vertex label*/
        _i=0;/*counter index vectors*/
        _degree_i=0;/*variable node degree*/
        _who_I_am=false;/*value of variable node*/
        _I_am_a_fixed_variable=false;/*tell if a variables has beeen fixed*/
        complexity_variable=0.;/*variable node complexity*/
        prod_V_plus=0.;/*products un-negated literal messages*/
        prod_V_minus=0.;/*product negated literal messages*/
        _sT=0.;/*survey sT variable node*/
        _sI=0.;/*survey sI variable node*/
        _sF=0.;/*survey sF variable node*/
        _sC=0.;/*certitude*/
        _I_am_white=false;/*white variable*/
    };
    
    
    ~Vertex(){};/*default destructor*/
    
    void make_products(); /*update products and surveys*/
    
    void compute_s(); /*compute surveys for each variable node*/
    
    void fix_var_i(); /*fix variable i to true or false*/
    
    void reset_value_default_var_i(); /*reset value default variable node*/
    
    double * ptr_survey(); /*pointer of  vector _surveys_cl_to_i*/
    
    double * ptr_survey(unsigned int &j); /*pointer of  vector _surveys_cl_to_i*/
    
    double _Pr_U(bool &b); /* product of unsitisfied messages*/
    
    double _Pr_S(bool &b, double &s); /* product of sitisfied messages*/
    
    double _p_PLUS(); /*compute value p_plus*/
    
    double _p_MINUS(); /*compute value p_minus*/
    
    double _p_IND(); /*compute value p_indeterminate*/
    
    double S(double &a, double &b, double &c); /*compute sT,sF*/
    
    double S_C(); /*compute certitude*/
    
    long int _var_int(bool f){ /*variable node fixed and expressed as int */
        return (f==true) ? (_vertex_lli):(-1*_vertex_lli);
    }
    
    friend ostream& operator<<(ostream &out, Vertex V){ /*friend member for print object public variables*/
        out<<V._vertex<<" "<<V._degree_i<<" "<<V.prod_V_plus<<" "<<V.prod_V_minus<<" "<<endl;
        for (unsigned int i=0; i<V._surveys_cl_to_i.size(); ++i) {
            out<<V._surveys_cl_to_i[i]<<" "<<&(V._surveys_cl_to_i[i])<<" "<<*real(V._I_am_in_cl_at_init[i])<<endl;
        }
        return out;
    }
    
    friend ostream& operator<<(ostream &out, Vertex* V){ /*friend member for print object public variables*/
        out<<V->_vertex<<" "<<V->_degree_i<<" "<<V->prod_V_plus<<" "<<V->prod_V_minus<<" "<<endl;
        for (unsigned int i=0; i<V->_surveys_cl_to_i.size(); ++i) {
            out<<V->_surveys_cl_to_i[i]<<" "<<&(V->_surveys_cl_to_i[i])<<" "<<*real(V->_I_am_in_cl_at_init[i])<<endl;
        }
        return out;
    }
    
    /*public variables*/
    vector<list<Vertex * >::iterator> where_I_am; /*store where i is in _cl*/
    vector<list<double *>::iterator>  _where_surveys_are_in_cl;/*store position into a list*/
    vector<double > _surveys_cl_to_i; /* store surveys, image is time t+1*/
    vector<complex<unsigned int *> > _I_am_in_cl_at_init; /*store where i is in _cl*/
    vector<list<bool >::iterator> _lit_list_i; /*store where i is in _cl*/
    list<Vertex *>::iterator _it_list_fixed_elem; /*store where the element is into the list of fixed elements*/
    vector<int *> _bvec_lit;/*list of literal in clause*/
    vector<double *> _surveys_cl_to_i_plus; /* store surveys associated to plus literal, real  is time t, image is time t+1*/
    vector<double *> _surveys_cl_to_i_minus; /* store surveys associated to plus literal, real is time t, image is time t+1*/
    double _sT; /*survey of i true*/
    double _sI; /*survey of i indeterminate*/
    double _sF; /*survey of i false*/
    double _sC; /*certitude*/
    double prod_V_plus; /*product (1-message) in V_plus*/
    double prod_V_minus; /*product (1-message) in V_minus*/
    double complexity_variable;/*variable node complexity*/
    unsigned int _vertex; /*vertex label*/
    long int _vertex_lli; /*vertex label for literal*/
    unsigned int _i; /*counter index vectors*/
    int _degree_i; /*degree of i*/
    bool _I_am_a_fixed_variable; /*fixed variable during decimation*/
    bool _who_I_am; /*variable node value after fixing during deciamtion*/
    bool _I_am_white; /*I am a white variable*/
    
private:
    
    /*private members*/
    double _one_minus(unsigned int &i);/*compute 1-survey stored into the class vector*/
};

/*private memeber  which returns the value  of 1-survey from the clause i to (this) variable*/
inline double Vertex::_one_minus(unsigned int &i){/*compute 1-survey stored into the class vector*/
    return (1.-_surveys_cl_to_i[i]);
}

/*public memeber which fixes the variable node to true or false, depending by surveys value _sT and _sF*/
inline void Vertex::fix_var_i(){/*fix variable node to rue or false*/
    if(_sT>_sF) _who_I_am=true;
    else _who_I_am=false;
}

/*public memeber which resets the variable node*/
inline void Vertex::reset_value_default_var_i(){
    _who_I_am=false;/*value of variable node*/
    _I_am_a_fixed_variable=false;/*tell if a variables has beeen fixed*/
}


/*public member which returns, depending by the literal into a clause, the right value of products of (1-message) for SP and BP equations*/
inline double Vertex::_Pr_U(bool &b){/* product of unsitisfied messages*/
    return (b)?(prod_V_minus):prod_V_plus;
}

/*public member which returns, depending by the literal into a clause, the right value of products of (1-message) for SP and BP equations*/
inline double Vertex::_Pr_S(bool &b, double &s){ /*product of sitisfied messages*/
    return (b)?(prod_V_plus*s):(prod_V_minus*s);
}

/*public member which returns the pointer of a vector where surveys are stored*/
inline double * Vertex::ptr_survey(){/*returns address vector surveys*/
    return &_surveys_cl_to_i[_i];
}

/*public member which returns the pointer, given an index j, of a vector where surveys are stored*/
inline double * Vertex::ptr_survey(unsigned int &j){/*returns address vector surveys*/
    return &_surveys_cl_to_i[j];
}

/*public member which returns value pi_plus for computing surveys sI,sT,SF*/
inline double Vertex::_p_PLUS(){/*compute  _p_plus variable node*/
    return ((1.-prod_V_plus)*prod_V_minus);
}

/*public member which returns value pi_minus for computing surveys sI,sT,SF*/
inline double Vertex::_p_MINUS(){/*compute _p_minus variable node*/
    return ((1.-prod_V_minus)*prod_V_plus);
}

/*public member which returns value pi_indeterminate for computing surveys sI,sT,SF*/
inline double Vertex::_p_IND(){/*compute _p_indeterminate variable node*/
    return (prod_V_plus*prod_V_minus);
}

/*public member which returns the survey _sT or sF*/
inline double Vertex::S(double &a, double &b, double &c){ /*compute sT,sF*/
    return (a/(a+b+c));
}

/*public member which describe how to compute the variable node certitude*/
inline double Vertex::S_C(){/*compute certitude survey*/
    return __H(_sT, _sF, _sI);
}

#endif /* Vertex_hpp */

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/******************************  END VERTEX CLASS  **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/

