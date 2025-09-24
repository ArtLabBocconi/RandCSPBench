//
//  Header.h
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
//  Created by Raffaele Marino on 17/10/2018.
//  Copyright © 2018 Raffaele Marino. All rights reserved.
//

#ifndef Header_h
#define Header_h
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <vector>
#include <iterator>
#include <algorithm>
#include <valarray>
#include <numeric>
#include <complex>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <iomanip>
#include <list>
#include <chrono> // Necessario per misurare il tempo
#include "random.h"

//#define NeurNET
#define WALKSAT
#define BP
#define ZERO (1.0E-10)/*all values less than 1.0e-10 are zero*/
#define PRINT_FORMULA_CNF
//#define POL
#define CERT
//#define P_M
//#define I_C


#define INITLEN 32
/*if SAT is un-commented will be a sat-solver*/
#define SAT
/*if XORSAT is un-commented will be a xorsat-solver*/
//#define XORSAT
/*if NAESAT is un-commented will be a naesat-solver*/
//#define NAESAT



using namespace std;
const string directory="./";/*directory for output files*/
enum{t_max=1024}; /*maximum number of iteration for a message passing algorithm*/
const double rho_SP=1.;/*from SP to BP when it is set to 0*/
const double frac=0.00125; /*fraction of variables for decimation*/
const double e=2.7182818284590452353602874713527;/*constant e*/
const double pi=3.141592653589793238462643383279;/*constant p*/
const double epsilon=0.01;/*convergence epsilon value*/
const double _R_BSP=0.0;/*r values for BSP, it goes from [0, 1).  When it is equal to 0, one gets a SID algorithms, while when it is different to 0 one gets the BSP. */
const double _phi=0.0;/*phi value for the BSP*/
#ifdef WALKSAT
int WalkSat(vector <vector<bool> > & sol,int argc, char * argv[]); // WalkSat function
#endif

#endif /* Header_h */






