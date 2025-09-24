//
//  Vertex.cpp
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

/*In this file public, non-inline, members for class Vertex are defined.*/

#include "Vertex.hpp"

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/***************************** START VERTEX CLASS **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/


/*public member which computes surveys sT, sF, sI and certitude for each variable node.
 Moreover, the function computes the own associated variable complexity.*/
void Vertex::compute_s(){/*compute surveys variable node*/
    double _p_plus=0.,_p_minus=0.,_p_I=0.;/*declaration local variables*/
    complexity_variable=0.;/*set variable complexity to zero*/
    _p_plus=_p_PLUS();/*compute bias _pi_plus*/
    _p_minus=_p_MINUS();/*compute bias _pi_minus*/
    _p_I=_p_IND();/*compute bias _pi_indeterminate*/
    complexity_variable=_p_I+_p_minus+_p_plus;/*update variable node complexity*/
    _sT=S(_p_plus, _p_minus, _p_I);/*compute survey _sT variable node*/
    _sF=S(_p_minus, _p_plus, _p_I);/*compute survey _sF variable node*/
    _sI=1.-_sT-_sF;/*compute survey _sI variable node*/
    _sC=S_C();/*compute certitude variable node i*/
}

/*public member which updates products (1-survey) for V_plus and V_minus sets. These products are usful for computing SP and BP equations in an easy way.*/
void Vertex::make_products(){/*make products for set V_plus and V_minus*/
    
    prod_V_plus=1.;/*set the product of V_plus to 1*/
    for (unsigned long i=0; i<_surveys_cl_to_i_plus.size(); ++i) {/*loop for updating vector, where are stored  messages, and products (1-survey) for V_plus and V_minus sets.*/
        prod_V_plus*=(1.- *_surveys_cl_to_i_plus[i]);/*compute products for V_plus*/
    }
    prod_V_minus=1.;/*set the product of V_minus to 1*/
    for (unsigned long i=0; i<_surveys_cl_to_i_minus.size(); ++i) {/*loop for updating vector, where are stored  messages, and products (1-survey) for V_plus and V_minus sets.*/
        prod_V_minus*=(1.- *_surveys_cl_to_i_minus[i]);/*compute products for V_minus*/
    }
}


/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/******************************  END VERTEX CLASS  **********************************/
/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
